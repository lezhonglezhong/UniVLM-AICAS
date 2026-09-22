#!/usr/bin/env python3
"""Probe PYNQ overlay metadata, download, and DDR allocation paths."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from constants import BITFILE, HWHFILE, OVERLAY_DIR, RESULTS_DIR


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def default_output() -> str:
    path = Path(RESULTS_DIR) / f"overlay_probe_{now_tag()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def file_info(path: str) -> dict[str, Any]:
    info: dict[str, Any] = {"path": os.path.abspath(path), "exists": os.path.exists(path)}
    if not os.path.exists(path):
        return info
    data = Path(path).read_bytes()
    info.update(
        {
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "head_hex": data[:16].hex(),
        }
    )
    return info


def record_step(payload: dict[str, Any], name: str, func) -> Any:
    """记录每个加载探针的异常和 traceback，避免一次失败中断整个诊断。"""
    try:
        value = func()
        payload["steps"][name] = {"ok": True, "result": value}
        return value
    except Exception as exc:  # noqa: BLE001 - probe must keep running after failures
        payload["steps"][name] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        return None


def run_cmd(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "output": (proc.stdout or "").strip(),
    }


def run_fpgautil(cmd: list[str]) -> dict[str, Any]:
    result = run_cmd(cmd)
    output = str(result.get("output", ""))
    semantic_fail = result["returncode"] != 0 or "failed" in output.lower() or "i/o error" in output.lower()
    result["semantic_ok"] = not semantic_fail
    return result


def fpga_manager_snapshot() -> dict[str, Any]:
    """采集 Linux FPGA manager 状态和最近内核日志，用于定位 bitstream 下载失败。"""
    paths = [
        "/sys/class/fpga_manager/fpga0/name",
        "/sys/class/fpga_manager/fpga0/state",
        "/sys/class/fpga_manager/fpga0/status",
        "/sys/class/fpga_manager/fpga0/firmware",
        "/sys/class/fpga_region/region0/state",
    ]
    snapshot: dict[str, Any] = {"sysfs": {}, "firmware_candidates": []}
    for path in paths:
        entry: dict[str, Any] = {"exists": os.path.exists(path)}
        if os.path.exists(path):
            try:
                entry["value"] = Path(path).read_text(encoding="utf-8", errors="replace").strip()
            except Exception as exc:  # noqa: BLE001 - sysfs nodes may be write-only
                entry["error"] = f"{type(exc).__name__}: {exc}"
        snapshot["sysfs"][path] = entry

    firmware_dir = Path("/lib/firmware")
    if firmware_dir.exists():
        for path in sorted(firmware_dir.glob("smolvlm2*")):
            try:
                snapshot["firmware_candidates"].append(
                    {"path": str(path), "size": path.stat().st_size}
                )
            except OSError as exc:
                snapshot["firmware_candidates"].append(
                    {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
                )

    dmesg = subprocess.run(["dmesg"], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = (dmesg.stdout or "").splitlines()
    snapshot["dmesg_tail"] = lines[-120:]
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe PYNQ overlay/HWH/fpgautil behavior.")
    parser.add_argument("--output", default=None, help="Result JSON path. Defaults under results/.")
    parser.add_argument("--download", action="store_true", help="Actually let PYNQ Overlay download the bitstream.")
    parser.add_argument("--fpgautil", action="store_true", help="Also try fpgautil with .bit/.bin files.")
    args = parser.parse_args()

    output = args.output or default_output()
    binfile = os.path.join(OVERLAY_DIR, "smolvlm2_kv260.bin")
    xsafile = os.path.join(OVERLAY_DIR, "smolvlm2_kv260.xsa")
    payload: dict[str, Any] = {
        "script": "core/overlay_probe.py",
        "output_path": os.path.abspath(output),
        "download": bool(args.download),
        "fpgautil": bool(args.fpgautil),
        "files": {
            "bit": file_info(BITFILE),
            "hwh": file_info(HWHFILE),
            "bin": file_info(binfile),
            "xsa": file_info(xsafile),
        },
        "steps": {},
    }

    def import_pynq() -> dict[str, Any]:
        import pynq  # type: ignore
        from pynq import Overlay  # type: ignore

        return {
            "pynq_version": getattr(pynq, "__version__", "unknown"),
            "overlay_signature": str(inspect.signature(Overlay)),
        }

    record_step(payload, "import_pynq", import_pynq)

    # 直接测试 HWH 是否可被 metadata 前端解析；若 head_hex 不是 3c3f786d6c，优先怀疑 BOM/文件格式。
    def parse_hwh() -> str:
        from pynqmetadata.frontends import Metadata  # type: ignore

        Metadata(input=HWHFILE)
        return "parsed"

    record_step(payload, "parse_hwh_metadata", parse_hwh)
    record_step(payload, "fpga_manager_before", fpga_manager_snapshot)

    def overlay_with_hwh_file() -> str:
        from pynq import Overlay  # type: ignore

        Overlay(BITFILE, hwh_file=HWHFILE, download=args.download)
        return "constructed"

    record_step(payload, "overlay_bit_hwh_file", overlay_with_hwh_file)

    def overlay_auto_hwh() -> str:
        from pynq import Overlay  # type: ignore

        Overlay(BITFILE, download=args.download)
        return "constructed"

    record_step(payload, "overlay_bit_auto_hwh", overlay_auto_hwh)

    def overlay_xsa() -> str:
        from pynq import Overlay  # type: ignore

        Overlay(xsafile, download=args.download)
        return "constructed"

    record_step(payload, "overlay_xsa", overlay_xsa)

    if args.fpgautil:
        if os.path.exists(BITFILE):
            record_step(payload, "fpgautil_bit", lambda: run_fpgautil(["fpgautil", "-b", BITFILE]))
        if os.path.exists(binfile):
            record_step(payload, "fpgautil_bin", lambda: run_fpgautil(["fpgautil", "-b", binfile]))
        record_step(payload, "fpga_manager_after_fpgautil", fpga_manager_snapshot)

    # 只有至少一条 PYNQ Overlay 路径成功且允许下载时才测试 DDR 分配，避免无意义地触发 default_memory 报错。
    overlay_ok = any(
        payload["steps"].get(name, {}).get("ok")
        for name in ("overlay_bit_auto_hwh", "overlay_xsa")
    )
    if args.download and overlay_ok:
        def allocate_smoke() -> dict[str, Any]:
            import pynq  # type: ignore

            buf = pynq.allocate(shape=(16,), dtype=np.uint8)
            result = {"device_address": int(buf.device_address), "nbytes": int(np.asarray(buf).nbytes)}
            buf.close()
            return result

        record_step(payload, "pynq_allocate_smoke", allocate_smoke)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"result_json: {output}")


if __name__ == "__main__":
    main()
