"""
SmolVLM2 KV260 PYNQ driver for the current reused LLM/ViT overlay.

This file is intentionally import-safe on a non-PYNQ machine: the `pynq`
package is imported only when `SmolVLM2Accelerator` is constructed on board.
The current PL contains reused mode-aware IPs. Connector, text embedding,
image patch/position embedding, ViT final post-LayerNorm, and image/text merge
are PS-side software steps implemented through `ps_float.py`.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

CORE_DIR = os.path.dirname(__file__)
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
_local_constants = os.path.abspath(os.path.join(CORE_DIR, "constants.py"))
_loaded_constants = sys.modules.get("constants")
if _loaded_constants is not None and os.path.abspath(getattr(_loaded_constants, "__file__", "")) != _local_constants:
    del sys.modules["constants"]

from constants import (
    AXILITE_BASE,
    AXI_PARAM_BIAS,
    AXI_PARAM_NORM,
    AXI_STATE_LAYER,
    AXI_STATE_REPLAY_TOKEN,
    AXI_STATE_WRITE_CLS,
    BIN_DIR,
    BITFILE,
    CLS_Y_ELEMS,
    C,
    CACHE_UPDATE_FULL_TILE,
    CONNECTOR_OUT_TOKENS,
    DECODER_STATE_ELEMS,
    DEFAULT_DECODER_STATE_SCALE,
    DEFAULT_FCLK0_MHZ,
    DEFAULT_VIT_INPUT_SCALE,
    DEFAULT_VIT_STATE_SCALE,
    HWHFILE,
    IMAGE_TOKEN_ID,
    LLAMA_L,
    LLM_CLS_RUN_MASK,
    LLM_CLS_WRITE_RUN_MASK,
    LLM_LAYER_RUN_MASK,
    MODE_LLM,
    MODE_VIT,
    PL_BIN_FILES,
    REGISTER_MAP,
    T,
    VIT_C,
    VIT_IMG_SIZE,
    VIT_L,
    VIT_LAYER_RUN_MASK,
    VIT_PATCH_SIZE,
    VIT_PARAM_NORM_RUN_MASK,
    VIT_SEQ_LEN,
    VIT_STATE_ELEMS,
    VIT_A_BYTES,
    VIT_STAGE_PROBE_PAIRS,
    VIT_XM_BYTES,
    KV_CACHE_BYTES,
    XSAFILE,
)
from ip import AXI_IP
from ps_float import PSFloatAssets


WEIGHT_BIN_KEYS = ("decoder_w", "cls_w", "vit_w")
VIT_ONLY_BIN_KEYS = {"vit_w", "vit_bias", "vit_lnw", "vit_lnb"}


def _add_timing(timings: dict[str, float] | None, key: str, value: float) -> None:
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + float(value)


def _ensure_xrt_env() -> None:
    """为非交互 SSH shell 补齐 PYNQ/XRT 设备发现所需环境变量。"""
    if "XILINX_XRT" in os.environ:
        return
    for root in ("/usr", "/opt/xilinx/xrt"):
        if os.path.exists(os.path.join(root, "lib", "libxrt_core.so")):
            # Ubuntu/KV260 镜像把 XRT runtime 装在 /usr；没有这个变量时
            # pynq._3rdparty.xrt 不会加载 libxrt_core.so，导致 No Devices Found。
            os.environ["XILINX_XRT"] = root
            return


def _import_pynq():
    _ensure_xrt_env()
    try:
        import pynq  # type: ignore
        from pynq import Clocks, Overlay  # type: ignore
    except ImportError as exc:
        raise RuntimeError("This driver must be constructed on a PYNQ board with the pynq package installed") from exc
    return pynq, Overlay, Clocks


def _download_bitfile_with_fpgautil(bitfile: str) -> tuple[bool, str]:
    """Fallback probe for PYNQ images whose Overlay metadata parser rejects .bit."""
    if bitfile.endswith(".xsa"):
        bitfile = os.path.splitext(bitfile)[0] + ".bit"
    candidates = []
    binfile = os.path.splitext(bitfile)[0] + ".bin"
    if os.path.exists(binfile):
        candidates.append(binfile)
    candidates.append(bitfile)

    outputs = []
    # fpgautil 通常需要 raw .bin；即使它能成功下载，demo 仍优先依赖 PYNQ Overlay 提供 DDR memory metadata。
    for candidate in candidates:
        try:
            proc = subprocess.run(
                ["fpgautil", "-b", candidate],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            return False, str(exc)
        output = (proc.stdout or "").strip()
        outputs.append(f"{candidate}: rc={proc.returncode}, output={output!r}")
        failed = proc.returncode != 0 or "failed" in output.lower() or "i/o error" in output.lower()
        if not failed:
            return True, "\n".join(outputs)
    return False, "\n".join(outputs)


def _overlay_failure_message(bitfile: str, hwhfile: str, overlay_exc: Exception, fpgautil_output: str) -> str:
    return (
        "FPGA overlay was not downloaded. PYNQ Overlay failed before DDR allocation, "
        "and the fpgautil fallback did not provide a usable PYNQ default memory. "
        f"bitfile={bitfile}, hwhfile={hwhfile}, "
        f"overlay_error={type(overlay_exc).__name__}: {overlay_exc}, "
        f"fpgautil_output={fpgautil_output!r}. "
        "Fix the PYNQ Overlay/HWH load path or provide a board-loadable .bin plus an explicit "
        "PYNQ memory allocation target before running demo.py."
    )


def _buffer_bytes(buf: Any) -> int:
    return int(np.asarray(buf).nbytes)


def _shape_bytes(shape: tuple[int, ...], dtype: Any) -> int:
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


def _mib(nbytes: int) -> float:
    return nbytes / (1024.0 * 1024.0)


def _flush(buf: Any) -> None:
    if hasattr(buf, "flush"):
        buf.flush()


def _invalidate(buf: Any) -> None:
    if hasattr(buf, "invalidate"):
        buf.invalidate()


@dataclass
class BufferInfo:
    name: str
    address: int
    bytes: int


class SmolVLM2Accelerator:
    """Board-side helper for overlay loading, DDR allocation, and coarse scheduling."""

    def __init__(
        self,
        bitfile: str = BITFILE,
        hwhfile: str = HWHFILE,
        bin_dir: str = BIN_DIR,
        axilite_base: int = AXILITE_BASE,
        fclk0_mhz: float | None = DEFAULT_FCLK0_MHZ,
        load_overlay: bool = True,
        load_weights: bool = True,
        load_ps_float: bool = True,
        enable_vit: bool = True,
        load_ps_vision: bool | None = None,
        load_text_fp16: bool | None = None,
        load_text_i16: bool = True,
        ps_vision_backend: str = "numpy",
        vit_scratch_banks: int = 2,
    ):
        self.bitfile = bitfile
        self.hwhfile = hwhfile
        self.bin_dir = bin_dir
        self.axilite_base = axilite_base
        self.fclk0_mhz = fclk0_mhz
        self.enable_vit = bool(enable_vit)
        self.vit_scratch_banks = max(1, int(vit_scratch_banks)) if self.enable_vit else 1
        self._current_vit_scratch_bank = 0
        if load_ps_vision is None:
            load_ps_vision = self.enable_vit
        if load_text_fp16 is None:
            # 当前 text-only 和 multimodal merge 都使用 PYNQ 同款 int16
            # embedding 表；除非调用方显式需要浮点 text lookup，否则不要
            # 额外加载 94MB fp16 text embedding。
            load_text_fp16 = False
        self.pynq, overlay_cls, clocks = _import_pynq()
        self._overlay_cls = overlay_cls
        self._clocks = clocks

        # PYNQ 3.0.1 在 KV260 镜像上可构造 XSA Overlay，但可能不能从 .bit 自动关联同名 HWH；
        # 因此优先用 Vivado 导出的 XSA，失败后再探测 .bit/.hwh 和 fpgautil 路径。
        self.overlay = None
        if load_overlay:
            self.overlay_load_method = "none"
            self.overlay_errors: list[str] = []
            self.fpgautil_output = ""
            self._load_overlay(download_bitfile=bitfile, hwhfile=hwhfile)
            if fclk0_mhz is not None:
                # 2026-05-23：完整 decoder layer golden 在 166/214MHz 会 mismatch；
                # 默认固定到 150MHz，避免沿用 overlay/平台默认高频。
                clocks.fclk0_mhz = fclk0_mhz
        else:
            self.overlay_load_method = "skipped"
            self.overlay_errors = []
            self.fpgautil_output = ""

        self._manual_xrt_memory = None
        self.hw = AXI_IP(axilite_base, [(name, offset, np.uint64) for name, offset in REGISTER_MAP])
        self.buffers: dict[str, Any] = {}
        self.ps_float = (
            PSFloatAssets(
                bin_dir,
                load_vision=bool(load_ps_vision),
                load_text_fp16=bool(load_text_fp16),
                load_text_i16=bool(load_text_i16),
                vision_backend=ps_vision_backend,
            )
            if load_ps_float
            else None
        )

        self._allocate_buffers()
        if load_weights:
            self.load_pl_bins()
        self.write_base_addresses()
        self.reset_runtime_registers()

    def _load_overlay(self, download_bitfile: str, hwhfile: str) -> None:
        """下载当前 overlay；供初始化和 PL 内部状态清空复用。"""
        overlay_candidates = []
        if os.path.exists(XSAFILE):
            overlay_candidates.append(XSAFILE)
        overlay_candidates.append(download_bitfile)
        last_exc: Exception | None = None
        overlay_errors: list[str] = []

        for overlay_path in dict.fromkeys(overlay_candidates):
            if not os.path.exists(overlay_path):
                last_exc = FileNotFoundError(overlay_path)
                overlay_errors.append(f"{overlay_path}: {type(last_exc).__name__}: {last_exc}")
                continue
            try:
                if overlay_path.endswith(".xsa"):
                    self.overlay = self._overlay_cls(overlay_path, download=True)
                else:
                    try:
                        kwargs = {"download": True}
                        if os.path.exists(hwhfile):
                            kwargs["hwh_file"] = hwhfile
                        self.overlay = self._overlay_cls(overlay_path, **kwargs)
                    except TypeError as exc:
                        if "hwh_file" not in str(exc):
                            raise
                        self.overlay = self._overlay_cls(overlay_path, download=True)
                self.bitfile = overlay_path
                self.overlay_load_method = "pynq_overlay"
                self.overlay_errors = []
                self.fpgautil_output = ""
                return
            except Exception as exc:
                last_exc = exc
                overlay_errors.append(f"{overlay_path}: {type(exc).__name__}: {exc}")
                self.overlay = None

        overlay_exc = RuntimeError("; ".join(overlay_errors)) if overlay_errors else last_exc or RuntimeError("no overlay candidate was attempted")
        fpgautil_ok, fpgautil_output = _download_bitfile_with_fpgautil(download_bitfile)
        self.overlay_errors = overlay_errors
        self.fpgautil_output = fpgautil_output
        if not fpgautil_ok:
            raise RuntimeError(
                _overlay_failure_message(download_bitfile, hwhfile, overlay_exc, fpgautil_output)
            ) from overlay_exc
        # PYNQ 3.0.1 在当前 KV260 镜像上不能从本项目 XSA 取出 t.xclbin，
        # 但 fpgautil 能加载 .bin，且控制寄存器用 /dev/mem、DDR 用默认
        # XRT allocator；因此 fallback 成功时继续执行板上 smoke。
        self.overlay_load_method = "fpgautil_bin"
        self.bitfile = os.path.splitext(download_bitfile)[0] + ".bin"

    def reload_overlay(self) -> None:
        """
        重新下载 PL overlay 以清空 HLS/AXIS 内部状态，保留现有 DDR buffers。

        这是 2026-05-23 连续 LLM layer golden 的临时 correctness workaround：
        DDR 中的 state/KV/weights 不动，只重置 PL 内部 FIFO/FSM，并重新写地址寄存器。
        """
        bitfile = self.hwhfile.replace(".hwh", ".bit")
        self._load_overlay(download_bitfile=bitfile, hwhfile=self.hwhfile)
        if self.fclk0_mhz is not None:
            self._clocks.fclk0_mhz = self.fclk0_mhz
        self.write_base_addresses()
        self.reset_runtime_registers()

    def _allocate_array(self, name: str, shape: tuple[int, ...], dtype: Any) -> Any:
        nbytes = _shape_bytes(shape, dtype)
        try:
            return self.pynq.allocate(shape=shape, dtype=dtype)
        except Exception as exc:
            message = str(exc)
            if "Overlay is not downloaded" in message:
                try:
                    return self._allocate_array_with_manual_xrt_memory(shape, dtype)
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"Allocate failed for {name}: {nbytes} bytes ({_mib(nbytes):.1f} MiB); "
                        f"default allocator error={type(exc).__name__}: {exc}; "
                        f"manual XRT DDR fallback error={type(fallback_exc).__name__}: {fallback_exc}"
                    ) from fallback_exc
            raise RuntimeError(f"Allocate failed for {name}: {nbytes} bytes ({_mib(nbytes):.1f} MiB)") from exc

    def _allocate_array_with_manual_xrt_memory(self, shape: tuple[int, ...], dtype: Any) -> Any:
        # PYNQ 3.0.1 在当前 KV260 镜像上不能从本项目 XSA 取出 t.xclbin；
        # fpgautil 可下载 bit/bin，但默认 allocate 仍会因缺少 Overlay
        # metadata 拒绝分配。这里显式构造 PS DDR bank0 的 XRT memory
        # target，只改变 buffer 分配路径，不改变 PL 地址或调度语义。
        if self._manual_xrt_memory is None:
            from pynq.pl_server.device import Device  # type: ignore
            from pynq.pl_server.embedded_device import EmbeddedXrtMemory  # type: ignore

            desc = {"idx": 0, "base_address": 0, "size": 0x80000000}
            self._manual_xrt_memory = EmbeddedXrtMemory(Device.active_device, desc)
        return self.pynq.allocate(shape=shape, dtype=dtype, target=self._manual_xrt_memory)

    def _allocate_u8_file_buffer(self, key: str, *, size: int | None = None, label: str | None = None) -> Any:
        path = os.path.join(self.bin_dir, PL_BIN_FILES[key])
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        nbytes = os.path.getsize(path) if size is None else size
        return self._allocate_array(label or key, (nbytes,), np.uint8)

    def _allocate_buffers(self) -> None:
        """分配当前硬件需要的 DDR 区域；不再分配旧 vision_input/connector_w。"""
        self.buffers["decoder_state"] = self._allocate_array("decoder_state", (DECODER_STATE_ELEMS,), np.int32)
        if self.enable_vit:
            # 全量 VLM 路径保持原先已验证过的分配顺序，避免 XRT/CMA 返回的
            # 物理地址布局变化影响多 M_AXI master 的 board-side golden 结果。
            self.buffers["vit_state"] = self._allocate_array("vit_state", (VIT_STATE_ELEMS,), np.int32)
            self.buffers["cls_y"] = self._allocate_array("cls_y", (CLS_Y_ELEMS,), np.int32)
            self.buffers["kv_cache"] = self._allocate_array("kv_cache", (KV_CACHE_BYTES,), np.uint8)
            for bank in range(self.vit_scratch_banks):
                suffix = "" if bank == 0 else f"_{bank}"
                # VIT-02: A_REORDER/XM_REORDER 仍使用同一套 RTL 接口，通过
                # 每层切换现有 base-address 寄存器选择 DDR scratch bank。
                self.buffers[f"vit_a{suffix}"] = self._allocate_array(f"vit_a{suffix}", (VIT_A_BYTES,), np.uint8)
                self.buffers[f"vit_xm{suffix}"] = self._allocate_array(f"vit_xm{suffix}", (VIT_XM_BYTES,), np.uint8)
        else:
            # ViT DDR 窗口只在图像路径需要。text-only demo 不提前分配这些
            # 连续 buffer，避免碎片化后影响 decoder_w_hi 这类大权重分配。
            self.buffers["cls_y"] = self._allocate_array("cls_y", (CLS_Y_ELEMS,), np.int32)
            self.buffers["kv_cache"] = self._allocate_array("kv_cache", (KV_CACHE_BYTES,), np.uint8)

        for key in PL_BIN_FILES:
            if not self.enable_vit and key in VIT_ONLY_BIN_KEYS:
                continue
            if key in WEIGHT_BIN_KEYS:
                path = os.path.join(self.bin_dir, PL_BIN_FILES[key])
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
                size = os.path.getsize(path)
                if size % 2 != 0:
                    raise ValueError(f"{path}: split weight file size must be even, got {size}")
                half = size // 2
                # WEIGHT_AXI 本来就是 LO/HI 双 M_AXI 端口；拆成两个 DDR buffer 可降低单次连续分配需求。
                self.buffers[f"{key}_lo"] = self._allocate_u8_file_buffer(key, size=half, label=f"{key}_lo")
                self.buffers[f"{key}_hi"] = self._allocate_u8_file_buffer(key, size=half, label=f"{key}_hi")
            else:
                self.buffers[key] = self._allocate_u8_file_buffer(key)

        self.reset_kv_cache()

    @staticmethod
    def _load_file_slice(path: str, buf: Any, offset: int, size: int) -> None:
        with open(path, "rb") as f:
            f.seek(offset)
            arr = np.fromfile(f, dtype=np.uint8, count=size)
        if arr.nbytes != size:
            raise ValueError(f"{path}: expected {size} bytes at offset {offset}, got {arr.nbytes}")
        if arr.nbytes != _buffer_bytes(buf):
            raise ValueError(f"{path}: got {arr.nbytes} bytes, buffer has {_buffer_bytes(buf)} bytes")
        buf[:] = arr
        _flush(buf)

    def load_pl_bins(self, keys: tuple[str, ...] | list[str] | None = None) -> None:
        """把 packed PL 权重/参数加载到 PYNQ buffer，并 flush 到 DDR。"""
        selected = PL_BIN_FILES.keys() if keys is None else keys
        for key in selected:
            filename = PL_BIN_FILES[key]
            if not self.enable_vit and key in VIT_ONLY_BIN_KEYS:
                # text-only 路径不会启动 ViT mode；对应 DDR 地址稍后写 0。
                continue
            path = os.path.join(self.bin_dir, filename)
            if key in WEIGHT_BIN_KEYS:
                lo = self.buffers[f"{key}_lo"]
                hi = self.buffers[f"{key}_hi"]
                half = _buffer_bytes(lo)
                if _buffer_bytes(hi) != half:
                    raise ValueError(f"{key}: LO/HI buffer sizes differ")
                if os.path.getsize(path) != half * 2:
                    raise ValueError(f"{path}: file size does not match split buffers")
                self._load_file_slice(path, lo, 0, half)
                self._load_file_slice(path, hi, half, half)
            else:
                arr = np.fromfile(path, dtype=np.uint8)
                buf = self.buffers[key]
                if arr.nbytes != _buffer_bytes(buf):
                    raise ValueError(f"{path}: got {arr.nbytes} bytes, buffer has {_buffer_bytes(buf)} bytes")
                buf[:] = arr
                _flush(buf)

    def flush_pl_bins(self, keys: tuple[str, ...] | list[str] | None = None) -> None:
        """把已加载的 PL 权重/参数 buffer 重新 flush，处理 cache coherency 边界。"""
        selected = PL_BIN_FILES.keys() if keys is None else keys
        for key in selected:
            if not self.enable_vit and key in VIT_ONLY_BIN_KEYS:
                continue
            if key in WEIGHT_BIN_KEYS:
                _flush(self.buffers[f"{key}_lo"])
                _flush(self.buffers[f"{key}_hi"])
            elif key in self.buffers:
                _flush(self.buffers[key])

    def buffer_info(self) -> list[BufferInfo]:
        return [
            BufferInfo(name, int(buf.device_address), _buffer_bytes(buf))
            for name, buf in sorted(self.buffers.items())
            if hasattr(buf, "device_address")
        ]

    def write_base_addresses(self) -> None:
        """把 DDR 物理地址写入 Spinal Manager 冻结寄存器。"""
        b = self.buffers
        def addr(name: str) -> int:
            return int(b[name].device_address) if name in b else 0

        decoder_state_addr = int(b["decoder_state"].device_address)
        self.hw.MEMORY_DECODER_X = decoder_state_addr
        self.hw.MEMORY_DECODER_Y = decoder_state_addr
        self.hw.MEMORY_DECODER_STATE = decoder_state_addr
        self.hw.MEMORY_CLS_Y = int(b["cls_y"].device_address)
        self.hw.MEMORY_K_CACHE = int(b["kv_cache"].device_address)

        self.hw.MEMORY_DECODER_W_LO = int(b["decoder_w_lo"].device_address)
        self.hw.MEMORY_DECODER_W_HI = int(b["decoder_w_hi"].device_address)
        self.hw.MEMORY_CLS_W_LO = int(b["cls_w_lo"].device_address)
        self.hw.MEMORY_CLS_W_HI = int(b["cls_w_hi"].device_address)
        self.hw.MEMORY_VIT_W_LO = addr("vit_w_lo")
        self.hw.MEMORY_VIT_W_HI = addr("vit_w_hi")

        self.hw.MEMORY_LLM_LNW = int(b["llm_lnw"].device_address)
        self.hw.MEMORY_CLS_LNW = int(b["cls_lnw"].device_address)
        self.hw.MEMORY_VIT_BIAS = addr("vit_bias")
        self.hw.MEMORY_VIT_LNW = addr("vit_lnw")
        self.hw.MEMORY_VIT_LNB = addr("vit_lnb")

        self.hw.MEMORY_VIT_STATE = addr("vit_state")
        if self.enable_vit:
            self._select_vit_scratch_bank(self._current_vit_scratch_bank)
        else:
            self.hw.MEMORY_VIT_A = 0
            self.hw.MEMORY_VIT_XM = 0

    @staticmethod
    def _vit_scratch_name(base: str, bank: int) -> str:
        return base if int(bank) == 0 else f"{base}_{int(bank)}"

    def _select_vit_scratch_bank(self, bank: int) -> int:
        """选择当前 ViT layer 使用的 A/XM DDR scratch bank。"""
        self._require_vit_enabled()
        selected = int(bank) % self.vit_scratch_banks
        a_name = self._vit_scratch_name("vit_a", selected)
        xm_name = self._vit_scratch_name("vit_xm", selected)
        self.hw.MEMORY_VIT_A = int(self.buffers[a_name].device_address)
        self.hw.MEMORY_VIT_XM = int(self.buffers[xm_name].device_address)
        self._current_vit_scratch_bank = selected
        return selected

    def _require_vit_enabled(self) -> None:
        if not self.enable_vit:
            raise RuntimeError("This accelerator was constructed with enable_vit=False; re-create it for image/Vision paths")

    def reset_runtime_registers(self) -> None:
        self.hw.L_BEGIN = 0
        self.hw.L_CLOSE = 0
        self.hw.POS = 0
        self.hw.MODE = MODE_LLM
        self.hw.PARAM_OP = AXI_PARAM_NORM
        self.hw.STATE_OP = AXI_STATE_LAYER
        self.hw.RUN_MASK = 0

    def reset_kv_cache(self) -> None:
        self.buffers["kv_cache"][:] = 0
        _flush(self.buffers["kv_cache"])

    def refresh_llm_runtime_buffers(self, copy_state: bool = True, copy_kv: bool = True) -> None:
        """
        重新分配 LLM 运行时 DDR buffer，并保留大权重/参数 buffer。

        连续多层 PYNQ golden 显示，单纯 reload overlay 不能清掉某些跨层残留；
        换一组 decoder_state/cls_y/kv_cache BO 后再重写地址，可测试是否为运行时
        buffer/AXI adapter 状态边界，而不必重新加载几百 MB 权重。
        """
        old_state = None
        old_kv = None
        if copy_state:
            _invalidate(self.buffers["decoder_state"])
            old_state = np.asarray(self.buffers["decoder_state"], dtype=np.int32).copy()
        if copy_kv:
            _invalidate(self.buffers["kv_cache"])
            old_kv = np.asarray(self.buffers["kv_cache"], dtype=np.uint8).copy()

        self.buffers["decoder_state"] = self._allocate_array("decoder_state_refresh", (DECODER_STATE_ELEMS,), np.int32)
        self.buffers["cls_y"] = self._allocate_array("cls_y_refresh", (CLS_Y_ELEMS,), np.int32)
        self.buffers["kv_cache"] = self._allocate_array("kv_cache_refresh", (KV_CACHE_BYTES,), np.uint8)

        if old_state is not None:
            self.buffers["decoder_state"][:] = old_state
            _flush(self.buffers["decoder_state"])
        else:
            self.buffers["decoder_state"][:] = 0
            _flush(self.buffers["decoder_state"])
        self.buffers["cls_y"][:] = 0
        _flush(self.buffers["cls_y"])
        if old_kv is not None:
            self.buffers["kv_cache"][:] = old_kv
            _flush(self.buffers["kv_cache"])
        else:
            self.reset_kv_cache()

        self.write_base_addresses()
        self.reset_runtime_registers()

    def wait_busy_then_idle(
        self,
        timeout_s: float = 30.0,
        poll_s: float = 0.001,
        start_timeout_s: float | None = None,
        allow_fast_complete: bool = False,
    ) -> None:
        # ViT norm preload 这类 M_AXI-only 阶段在硬件上可能只持续几十微秒；
        # Python 轮询可能错过 IDLE=0 窗口。对这类已知短阶段允许“未观察到 busy，
        # 但最终仍为 idle”作为已完成，避免预取 FIFO 被误判后遗留给下一样本。
        if start_timeout_s is None:
            start_timeout_s = 0.01 if allow_fast_complete else 1.0
        start_deadline = time.time() + min(float(start_timeout_s), timeout_s)
        while time.time() < start_deadline:
            if int(self.hw.IDLE) == 0:
                break
            time.sleep(poll_s)
        else:
            if allow_fast_complete and int(self.hw.IDLE) != 0:
                return
            raise TimeoutError("accelerator did not leave idle after trigger")
        self.wait_idle(timeout_s=timeout_s, poll_s=poll_s)

    def wait_idle(self, timeout_s: float = 30.0, poll_s: float = 0.001) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if int(self.hw.IDLE) != 0:
                return
            time.sleep(poll_s)
        raise TimeoutError(f"accelerator did not become idle in {timeout_s:.3f}s")

    def trigger(
        self,
        run_mask: int,
        mode: int,
        l_begin: int,
        l_close: int,
        pos: int = 0,
        param_op: int = AXI_PARAM_NORM,
        state_op: int = AXI_STATE_LAYER,
        wait: bool = True,
        timeout_s: float = 30.0,
        allow_fast_complete: bool = False,
        cache_update_mode: int = CACHE_UPDATE_FULL_TILE,
    ) -> None:
        """
        启动一次闭合硬件阶段。

        `RUN_MASK` 必须覆盖所有会互相消费 AXIS 的 IP；不要单独启动 producer。
        LLM 调度不启动 A/XM_REORDER，ViT 调度不启动 KV_CACHE。
        """
        self.hw.L_BEGIN = int(l_begin)
        self.hw.L_CLOSE = int(l_close)
        self.hw.POS = int(pos)
        self.hw.MODE = int(mode)
        self.hw.PARAM_OP = int(param_op)
        self.hw.STATE_OP = int(state_op)
        self.hw.RUN_MASK = int(run_mask)
        # 当前 HLS 已回到量化 K/V cache 并删除 cache_update_mode 端口；
        # 参数保留给旧调试脚本兼容，但不再写 AXI-Lite stale register。
        _ = cache_update_mode
        self.hw.T = 1
        if wait:
            self.wait_busy_then_idle(timeout_s=timeout_s, allow_fast_complete=allow_fast_complete)

    def run_vit_norm_preload(self, layer: int, timeout_s: float = 10.0) -> None:
        """ViT 每层 bias/gamma/beta 参数搬运阶段。"""
        self._require_vit_enabled()
        self.trigger(
            VIT_PARAM_NORM_RUN_MASK,
            MODE_VIT,
            layer,
            layer + 1,
            param_op=AXI_PARAM_NORM,
            state_op=AXI_STATE_LAYER,
            timeout_s=timeout_s,
            allow_fast_complete=True,
        )

    def run_vit_bias_preload(self, layer: int, timeout_s: float = 10.0) -> None:
        """保留接口占位：ViT bias 不能单独预取，必须随主链由 DEMUX 消费。"""
        raise RuntimeError("ViT bias is streamed during run_vit_layer(); do not run M_AXI bias standalone")

    def run_vit_layer(self, layer: int, pos: int = 0, timeout_s: float = 60.0) -> None:
        """ViT layer 主链：包含 A_REORDER/XM_REORDER scratch，不包含 KV_CACHE。"""
        self._require_vit_enabled()
        self._select_vit_scratch_bank(layer)
        self.trigger(
            VIT_LAYER_RUN_MASK,
            MODE_VIT,
            layer,
            layer + 1,
            pos=pos,
            param_op=AXI_PARAM_BIAS,
            state_op=AXI_STATE_LAYER,
            timeout_s=timeout_s,
        )

    def run_llm_layer(
        self,
        layer: int,
        pos: int,
        timeout_s: float = 30.0,
        cache_update_mode: int = CACHE_UPDATE_FULL_TILE,
    ) -> None:
        """LLM decoder layer 主链：包含 KV_CACHE，不启动 ViT-only reorder IP。"""
        # 旧 raw-V/current-only 参数在当前 bitstream 中无效，保留函数签名只为脚本兼容。
        self.trigger(
            LLM_LAYER_RUN_MASK,
            MODE_LLM,
            layer,
            layer + 1,
            pos=pos,
            param_op=AXI_PARAM_NORM,
            state_op=AXI_STATE_LAYER,
            timeout_s=timeout_s,
            cache_update_mode=cache_update_mode,
        )

    def run_llm_cls_with_trace(
        self,
        pos: int,
        timeout_s: float = 30.0,
        write_delay_s: float = 0.005,
    ) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
        """运行 final RMSNorm + lm_head，并记录两阶段 CLS 调度边界寄存器。"""
        self.buffers["cls_y"][:] = 0
        _flush(self.buffers["cls_y"])
        trace = {"before_main": self.dump_registers()}
        # CLS/lm_head 与 Spinal 顶层仿真一致：先启动主链并让 STATE_AXI 做
        # token-major replay 喂 final RMSNorm；随后单独重启 STATE_AXI 消费 DEMUX
        # argmax 写回。不能把主链直接配成 WRITE_CLS，否则没有 x_stream replay。
        self.trigger(
            LLM_CLS_RUN_MASK,
            MODE_LLM,
            LLAMA_L,
            LLAMA_L + 1,
            pos=pos,
            param_op=AXI_PARAM_NORM,
            state_op=AXI_STATE_REPLAY_TOKEN,
            wait=False,
            timeout_s=timeout_s,
        )
        time.sleep(float(write_delay_s))
        trace["before_write"] = self.dump_registers()
        self.trigger(
            LLM_CLS_WRITE_RUN_MASK,
            MODE_LLM,
            LLAMA_L,
            LLAMA_L + 1,
            pos=pos,
            param_op=AXI_PARAM_NORM,
            state_op=AXI_STATE_WRITE_CLS,
            timeout_s=timeout_s,
            allow_fast_complete=True,
        )
        trace["after_write"] = self.dump_registers()
        _invalidate(self.buffers["cls_y"])
        return np.asarray(self.buffers["cls_y"], dtype=np.int32).copy(), trace

    def run_llm_cls(self, pos: int, timeout_s: float = 30.0, write_delay_s: float = 0.005) -> np.ndarray:
        """运行 final RMSNorm + lm_head，返回当前 chunk 的 int32 argmax token id。"""
        got, _ = self.run_llm_cls_with_trace(pos=pos, timeout_s=timeout_s, write_delay_s=write_delay_s)
        return got

    def write_decoder_state_from_float(
        self,
        embeddings: np.ndarray,
        scale: float = DEFAULT_DECODER_STATE_SCALE,
    ) -> None:
        """把 PS 浮点 embedding 按 O_res=12 默认尺度量化写入 LLM state。"""
        if scale is None:
            raise ValueError("decoder state scale must be explicit")
        x = np.asarray(embeddings, dtype=np.float32).reshape(-1, C)
        if x.shape[0] > T:
            raise ValueError(f"decoder chunk supports at most {T} tokens, got {x.shape[0]}")
        q = np.rint(x / float(scale)).clip(np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)
        buf = self.buffers["decoder_state"]
        buf[:] = 0
        buf[: q.size] = q.reshape(-1)
        _flush(buf)

    def write_decoder_state_from_int32(self, embeddings: np.ndarray) -> None:
        """把 PS 已量化好的 int32 embedding/state 直接写入 LLM state。"""
        x = np.asarray(embeddings, dtype=np.int32).reshape(-1, C)
        if x.shape[0] > T:
            raise ValueError(f"decoder chunk supports at most {T} tokens, got {x.shape[0]}")
        buf = self.buffers["decoder_state"]
        buf[:] = 0
        buf[: x.size] = x.reshape(-1)
        _flush(buf)

    def write_vit_state_from_float(
        self,
        embeddings: np.ndarray,
        scale: float = DEFAULT_VIT_INPUT_SCALE,
    ) -> None:
        """把单个子图的 PS patch/position embedding 按 O_res=12 写入 ViT state。"""
        self._require_vit_enabled()
        if scale is None:
            raise ValueError("vit input state scale must be explicit")
        x = np.asarray(embeddings, dtype=np.float32).reshape(VIT_SEQ_LEN, VIT_C)
        q = np.rint(x / float(scale)).clip(np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)
        buf = self.buffers["vit_state"]
        buf[:] = q.reshape(-1)
        _flush(buf)

    def write_vit_state_from_int32(self, embeddings: np.ndarray) -> None:
        """把已量化的 ViT patch/state 直接写入 ViT state，供 CUDA/BF16 软件 patch 对齐调试。"""
        self._require_vit_enabled()
        x = np.asarray(embeddings, dtype=np.int32).reshape(VIT_SEQ_LEN, VIT_C)
        buf = self.buffers["vit_state"]
        buf[:] = x.reshape(-1)
        _flush(buf)

    def read_vit_state(self) -> np.ndarray:
        self._require_vit_enabled()
        _invalidate(self.buffers["vit_state"])
        return np.asarray(self.buffers["vit_state"], dtype=np.int32).copy()

    def read_vit_state_into(self, out: np.ndarray) -> None:
        """Invalidate and copy the current ViT state into a preallocated int32 array."""
        self._require_vit_enabled()
        target = np.asarray(out, dtype=np.int32)
        if target.size != VIT_STATE_ELEMS:
            raise ValueError(f"output buffer must hold {VIT_STATE_ELEMS} int32 elements, got {target.size}")
        _invalidate(self.buffers["vit_state"])
        np.copyto(target.reshape(-1), np.asarray(self.buffers["vit_state"], dtype=np.int32))

    def ps_text_embeddings(self, input_ids: np.ndarray | list[int]) -> np.ndarray:
        if self.ps_float is None:
            raise RuntimeError("PS float assets were not loaded")
        return self.ps_float.text_lookup(input_ids)

    def ps_text_embeddings_i32(self, input_ids: np.ndarray | list[int]) -> np.ndarray:
        if self.ps_float is None:
            raise RuntimeError("PS assets were not loaded")
        return self.ps_float.text_lookup_i32(input_ids)

    def ps_image_features_from_vit_state(
        self,
        vit_state_scale: float = DEFAULT_VIT_STATE_SCALE,
        vit_state: np.ndarray | None = None,
        timings: dict[str, float] | None = None,
    ) -> np.ndarray:
        if self.ps_float is None:
            raise RuntimeError("PS float assets were not loaded")
        if vit_state is None:
            t0 = time.perf_counter()
            vit_state = self.read_vit_state()
            _add_timing(timings, "vision_read_vit_state", time.perf_counter() - t0)
        t0 = time.perf_counter()
        features = self.ps_float.image_features_from_vit_state(vit_state, vit_state_scale)
        _add_timing(timings, "vision_final_ln_connector", time.perf_counter() - t0)
        return features

    def run_vit_encoder_from_embeddings(
        self,
        patch_embeddings: np.ndarray,
        vit_input_scale: float = DEFAULT_VIT_INPUT_SCALE,
        layer_timeout_s: float = 60.0,
        norm_timeout_s: float = 10.0,
        timings: dict[str, float] | None = None,
    ) -> None:
        """
        单个 512x512 子图：写入初始 ViT state 后串行跑 12 层 ViT。

        当前硬件 ViT 一次只覆盖 `T=1024`，因此多子图必须在上层循环多次调用本函数。
        """
        self._require_vit_enabled()
        t0 = time.perf_counter()
        self.write_vit_state_from_float(patch_embeddings, vit_input_scale)
        _add_timing(timings, "vision_write_vit_state", time.perf_counter() - t0)
        for layer in range(VIT_L):
            t0 = time.perf_counter()
            self.run_vit_norm_preload(layer, timeout_s=norm_timeout_s)
            _add_timing(timings, "vision_norm_preload", time.perf_counter() - t0)
            t0 = time.perf_counter()
            self.run_vit_layer(layer, timeout_s=layer_timeout_s)
            elapsed = time.perf_counter() - t0
            _add_timing(timings, "vision_vit_layers", elapsed)
            _add_timing(timings, f"vision_vit_layer_{layer}", elapsed)

    def run_single_subimage_features(
        self,
        pixel_values: np.ndarray,
        patch_attention_mask: np.ndarray | None,
        vit_input_scale: float = DEFAULT_VIT_INPUT_SCALE,
        vit_state_scale: float = DEFAULT_VIT_STATE_SCALE,
        layer_timeout_s: float = 60.0,
        norm_timeout_s: float = 10.0,
        timings: dict[str, float] | None = None,
    ) -> np.ndarray:
        """单个真实子图 -> PL ViT -> PS final post-LN/Connector -> `[1,64,960]`。"""
        self._require_vit_enabled()
        if self.ps_float is None:
            raise RuntimeError("PS float assets were not loaded")
        patches_per_side = VIT_IMG_SIZE // VIT_PATCH_SIZE
        pixels = np.asarray(pixel_values, dtype=np.float32).reshape(1, 3, VIT_IMG_SIZE, VIT_IMG_SIZE)
        patch_mask = None if patch_attention_mask is None else np.asarray(patch_attention_mask).reshape(1, patches_per_side, patches_per_side)
        t0 = time.perf_counter()
        patch_embeddings = self.ps_float.patch_position_embedding(pixels, patch_mask)[0]
        _add_timing(timings, "vision_patch_position", time.perf_counter() - t0)
        self.run_vit_encoder_from_embeddings(
            patch_embeddings,
            vit_input_scale=vit_input_scale,
            layer_timeout_s=layer_timeout_s,
            norm_timeout_s=norm_timeout_s,
            timings=timings,
        )
        features = self.ps_image_features_from_vit_state(vit_state_scale, timings=timings)
        return features.reshape(1, CONNECTOR_OUT_TOKENS, C)

    def run_image_features_serial(
        self,
        pixel_values: np.ndarray,
        pixel_attention_mask: np.ndarray | None,
        vit_input_scale: float = DEFAULT_VIT_INPUT_SCALE,
        vit_state_scale: float = DEFAULT_VIT_STATE_SCALE,
        layer_timeout_s: float = 60.0,
        norm_timeout_s: float = 10.0,
        image_batch: Any | None = None,
        timings: dict[str, float] | None = None,
    ) -> np.ndarray:
        """
        多子图串行 Vision 路径。

        `pixel_values` 来自 processor，通常为 `[B,N_images,3,512,512]`。这里过滤 padding
        全黑子图，按 processor 顺序复用同一块 ViT DDR scratch，输出 `[N_real,64,960]`。
        """
        self._require_vit_enabled()
        if self.ps_float is None:
            raise RuntimeError("PS float assets were not loaded")
        if image_batch is None:
            t0 = time.perf_counter()
            batch = self.ps_float.flatten_processor_images(pixel_values, pixel_attention_mask)
            _add_timing(timings, "vision_flatten_processor_images", time.perf_counter() - t0)
        else:
            batch = image_batch

        num_subimages = int(batch.pixel_values.shape[0])
        if num_subimages <= 0:
            return np.empty((0, CONNECTOR_OUT_TOKENS, C), dtype=np.float32)

        def patch_mask_for(idx: int) -> np.ndarray | None:
            if batch.patch_attention_mask is None:
                return None
            return np.asarray(batch.patch_attention_mask[idx : idx + 1])

        def compute_patch(idx: int) -> tuple[np.ndarray, float]:
            t_patch = time.perf_counter()
            patch = self.ps_float.patch_position_embedding(
                batch.pixel_values[idx : idx + 1],
                patch_mask_for(idx),
            )[0]
            return patch, time.perf_counter() - t_patch

        def compute_features(state: np.ndarray) -> tuple[np.ndarray, float]:
            t_features = time.perf_counter()
            features = self.ps_float.image_features_from_vit_state(state, vit_state_scale)
            return features.reshape(CONNECTOR_OUT_TOKENS, C), time.perf_counter() - t_features

        patch_future: Future[tuple[np.ndarray, float]]
        feature_futures: list[Future[tuple[np.ndarray, float]]] = []
        features: list[np.ndarray | None] = [None] * num_subimages

        with ThreadPoolExecutor(max_workers=1) as patch_executor, ThreadPoolExecutor(max_workers=1) as feature_executor:
            patch_future = patch_executor.submit(compute_patch, 0)
            for idx in range(num_subimages):
                t0 = time.perf_counter()
                patch_embedding, patch_elapsed = patch_future.result()
                _add_timing(timings, "vision_patch_position_batch", time.perf_counter() - t0)
                _add_timing(timings, "vision_patch_position_async_compute", patch_elapsed)

                if idx + 1 < num_subimages:
                    # VIT-02(a): 下一子图的 PS patch/position embedding 在当前
                    # 子图 PL ViT 运行期间准备；第一张图仍需同步等待。
                    patch_future = patch_executor.submit(compute_patch, idx + 1)

                t_sub = time.perf_counter()
                self.run_vit_encoder_from_embeddings(
                    patch_embedding,
                    vit_input_scale=vit_input_scale,
                    layer_timeout_s=layer_timeout_s,
                    norm_timeout_s=norm_timeout_s,
                    timings=timings,
                )
                t0 = time.perf_counter()
                state = np.empty((VIT_SEQ_LEN, VIT_C), dtype=np.int32)
                self.read_vit_state_into(state)
                _add_timing(timings, "vision_read_vit_state", time.perf_counter() - t0)
                _add_timing(timings, "vision_subimage_pl_total", time.perf_counter() - t_sub)

                # VIT-02(a): 当前子图 final post-LN/Connector 交给后台执行，
                # 主线程继续启动下一子图 PL；输出按 idx 收集，保持 processor 顺序。
                feature_futures.append(feature_executor.submit(compute_features, state))

            for idx, future in enumerate(feature_futures):
                t0 = time.perf_counter()
                feature, feature_elapsed = future.result()
                _add_timing(timings, "vision_final_ln_connector_wait", time.perf_counter() - t0)
                _add_timing(timings, "vision_final_ln_connector", feature_elapsed)
                features[idx] = feature

        return np.stack([feature for feature in features if feature is not None], axis=0).reshape(
            -1,
            CONNECTOR_OUT_TOKENS,
            C,
        )

    def build_merged_embeddings(
        self,
        input_ids: np.ndarray | list[int],
        pixel_values: np.ndarray,
        pixel_attention_mask: np.ndarray | None,
        vit_input_scale: float = DEFAULT_VIT_INPUT_SCALE,
        vit_state_scale: float = DEFAULT_VIT_STATE_SCALE,
        layer_timeout_s: float = 60.0,
        norm_timeout_s: float = 10.0,
        image_batch: Any | None = None,
        timings: dict[str, float] | None = None,
    ) -> np.ndarray:
        """文本 embedding + 多子图串行 image features merge，供后续 LLM prefill 使用。"""
        self._require_vit_enabled()
        if self.ps_float is None:
            raise RuntimeError("PS float assets were not loaded")
        ids = np.asarray(input_ids, dtype=np.int64)
        # 多模态路径也使用 text-only 已验证过的 int16 embedding 表。PS
        # Connector 仍是浮点，但在 merge 前量化成同一 O_res=12 的 int32，
        # 使 LLM 入口与 `merged_i32` 调试路径完全一致。
        t0 = time.perf_counter()
        text_embeds_i32 = self.ps_float.text_lookup_i32(ids)
        _add_timing(timings, "prepare_text_lookup_i32", time.perf_counter() - t0)
        t0 = time.perf_counter()
        image_features = self.run_image_features_serial(
            pixel_values,
            pixel_attention_mask,
            vit_input_scale=vit_input_scale,
            vit_state_scale=vit_state_scale,
            layer_timeout_s=layer_timeout_s,
            norm_timeout_s=norm_timeout_s,
            image_batch=image_batch,
            timings=timings,
        )
        _add_timing(timings, "vision_image_features_total", time.perf_counter() - t0)
        expected_image_tokens = image_features.shape[0] * CONNECTOR_OUT_TOKENS
        actual_image_tokens = int((ids == IMAGE_TOKEN_ID).sum())
        if actual_image_tokens != expected_image_tokens:
            raise ValueError(
                f"<image> token count {actual_image_tokens} != real subimages {image_features.shape[0]} "
                f"* {CONNECTOR_OUT_TOKENS}"
            )
        t0 = time.perf_counter()
        merged = self.ps_float.merge_embeddings_i32(ids, text_embeds_i32, image_features)
        _add_timing(timings, "prepare_merge_embeddings_i32", time.perf_counter() - t0)
        return merged

    def dump_registers(self) -> dict[str, int]:
        return {name: int(getattr(self.hw, name)) for name, _ in REGISTER_MAP}

    def read_vit_stage_probe(self) -> dict[str, Any]:
        """Read passive ViT stage probe status registers."""
        flags = int(self.hw.VIT_PROBE_FLAGS)
        layer_cycles = int(self.hw.VIT_PROBE_LAYER_CYCLES)
        stages = []
        for stage, invocation, reg_name in VIT_STAGE_PROBE_PAIRS:
            word = int(getattr(self.hw, reg_name))
            start = word & 0xFFFFFFFF
            done = (word >> 32) & 0xFFFFFFFF
            span = done - start if done >= start and done != 0 else 0
            stages.append(
                {
                    "stage": stage,
                    "invocation": int(invocation),
                    "start_cyc": int(start),
                    "done_cyc": int(done),
                    "span_cyc": int(span),
                    "valid": bool(span > 0),
                    "register": reg_name,
                }
            )

        def _span(first: int, second: int) -> int | None:
            return second - first if first > 0 and second > 0 and second >= first else None

        def _delta(first: int, second: int) -> int | None:
            return second - first if first > 0 and second > 0 else None

        permute = [s for s in stages if s["stage"] == "PERMUTE" and s["valid"]]
        permute_gaps = []
        for prev, cur in zip(permute, permute[1:]):
            permute_gaps.append(
                {
                    "from_invocation": prev["invocation"],
                    "to_invocation": cur["invocation"],
                    "gap_cyc": _span(prev["done_cyc"], cur["start_cyc"]),
                }
            )

        a_first_w = int(self.hw.VIT_PROBE_A_FIRST_W)
        a_last_b = int(self.hw.VIT_PROBE_A_LAST_B)
        a_first_ar = int(self.hw.VIT_PROBE_A_FIRST_AR)
        a_last_r = int(self.hw.VIT_PROBE_A_LAST_R)
        xm_first_w = int(self.hw.VIT_PROBE_XM_FIRST_W)
        xm_last_b = int(self.hw.VIT_PROBE_XM_LAST_B)
        xm_first_ar = int(self.hw.VIT_PROBE_XM_FIRST_AR)
        xm_last_r = int(self.hw.VIT_PROBE_XM_LAST_R)
        attention_stages = {"QK_GEMM", "SOFTMAX", "RV_GEMM"}
        attention_valid = [s for s in stages if s["stage"] in attention_stages and s["valid"]]
        attention_span = 0
        if attention_valid:
            attention_span = max(s["done_cyc"] for s in attention_valid) - min(s["start_cyc"] for s in attention_valid)

        return {
            "flags": flags,
            "overflow": bool(flags & 0x1),
            "active": bool(flags & 0x2),
            "layer_cycles": layer_cycles,
            "stages": stages,
            "permute_gaps": permute_gaps,
            "a_reorder_store_span_cyc": _span(a_first_w, a_last_b),
            "a_reorder_replay_span_cyc": _span(a_first_ar, a_last_r),
            "a_reorder_barrier_gap_cyc": _delta(a_last_b, a_first_ar),
            "xm_reorder_store_span_cyc": _span(xm_first_w, xm_last_b),
            "xm_reorder_replay_span_cyc": _span(xm_first_ar, xm_last_r),
            "xm_reorder_barrier_gap_cyc": _delta(xm_last_b, xm_first_ar),
            "a_first_w_cyc": a_first_w,
            "a_last_b_cyc": a_last_b,
            "a_first_ar_cyc": a_first_ar,
            "a_last_r_cyc": a_last_r,
            "xm_first_w_cyc": xm_first_w,
            "xm_last_b_cyc": xm_last_b,
            "xm_first_ar_cyc": xm_first_ar,
            "xm_last_r_cyc": xm_last_r,
            "attention_span_cyc": attention_span,
            "sum_stage_spans_cyc": int(sum(s["span_cyc"] for s in stages if s["valid"])),
        }


def create_accelerator(**kwargs) -> SmolVLM2Accelerator:
    return SmolVLM2Accelerator(**kwargs)


if __name__ == "__main__":
    acc = create_accelerator(load_weights=True, load_ps_float=False)
    for info in acc.buffer_info():
        print(f"{info.name}: addr=0x{info.address:016x}, bytes={info.bytes}")
    print(acc.dump_registers())
