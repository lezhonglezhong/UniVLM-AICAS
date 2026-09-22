"""
Current PYNQ local inference helpers.

This module intentionally targets the current reused-IP overlay.  It does not
call llama-server/OpenAI APIs and does not use the obsolete Qwen-style
`generate/run_decode/put_embedding` methods.  `generate_once()` now runs a
minimal text prefill/decode loop over the reused LLM hardware, while multimodal
inputs still depend on the PS-side image preprocessing path.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Patch embedding and connector are large OpenBLAS GEMMs on the KV260 PS.
# Four threads matched the 4-core ARM and was fastest in the 250 MHz N<=5 run.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from constants import (
    BIN_DIR,
    C,
    CACHE_UPDATE_FULL_TILE,
    CONNECTOR_OUT_TOKENS,
    DEFAULT_DECODER_STATE_SCALE,
    DEFAULT_FCLK0_MHZ,
    DEFAULT_MAX_IMAGE_SPLITS,
    IMAGE_TOKEN_ID,
    LLAMA_L,
    MODEL_DIR,
    PL_BIN_FILES,
    RESULTS_DIR,
    S,
    T,
)
from fpga_smolvlm2 import create_accelerator


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def results_path(name: str) -> str:
    path = Path(RESULTS_DIR) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def default_output(prefix: str) -> str:
    return results_path(f"{prefix}_{now_tag()}.json")


def write_json(path: str, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_timing(timings: dict[str, Any] | None, key: str, value: float) -> None:
    if timings is not None:
        timings[key] = float(timings.get(key, 0.0)) + float(value)


def add_count(timings: dict[str, Any] | None, key: str, value: int = 1) -> None:
    if timings is not None:
        timings[key] = int(timings.get(key, 0)) + int(value)


def extract_input_ids(encoded: Any) -> Any:
    # 兼容 transformers 5.x BatchEncoding、普通 dict 和直接返回 ndarray/list 的情况。
    try:
        return encoded["input_ids"]
    except Exception:
        return encoded


def _fast_smolvlm_image_inputs(
    image: Any,
    image_processor: Any,
    max_splits: int | None,
) -> tuple[np.ndarray, np.ndarray, list[list[int]], list[list[int]]]:
    del max_splits  # The already-patched image_processor owns the N<=5 policy.
    max_size_cfg = getattr(image_processor, "max_image_size", None) or {"longest_edge": 512}
    max_edge = int(max_size_cfg.get("longest_edge", 512))
    resample = getattr(image_processor, "resample", 1)
    rescale_factor = float(getattr(image_processor, "rescale_factor", 1.0 / 255.0))
    image_mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
    image_std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])

    image_array = image_processor.process_image(
        image,
        do_convert_rgb=getattr(image_processor, "do_convert_rgb", True),
    )
    resized = image_processor.resize(image=image_array, size=image_processor.size, resample=resample)
    split_ready = image_processor.resize_for_vision_encoder(resized, max_edge, resample=resample)
    frames, rows, cols = image_processor.split_images(
        split_ready,
        max_image_size={"longest_edge": max_edge},
        resample=resample,
    )
    processed = [
        image_processor.normalize(image_processor.rescale(frame, rescale_factor), image_mean, image_std)
        for frame in frames
    ]

    padded = []
    masks = []
    for frame in processed:
        padded_frame, pixel_mask = image_processor.pad(frame, (max_edge, max_edge))
        padded.append(padded_frame)
        masks.append(pixel_mask)
    return np.array([padded]), np.array([masks]), [[int(rows)]], [[int(cols)]]


@dataclass
class LocalGenerationResult:
    status: str
    mode: str
    implementation: str = "pynq_text_prefill_decode"
    prompt: str | None = None
    image_path: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    output_ids: list[int] = field(default_factory=list)
    output_text: str = ""
    timings_s: dict[str, float] = field(default_factory=dict)
    input_info: dict[str, Any] = field(default_factory=dict)
    fclk0_mhz: float | None = None
    max_image_splits: int | None = None
    overlay_reload_interval: int | None = None
    overlay_reloads: int = 0
    register_dump: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalPynqRunner:
    """Small board-side runner shared by demo and evaluation scripts."""

    def __init__(
        self,
        dry_run: bool = False,
        load_overlay: bool = True,
        load_weights: bool = True,
        load_ps_float: bool = True,
        enable_vit: bool = True,
        load_ps_vision: bool | None = None,
        load_text_fp16: bool | None = None,
        overlay_reload_interval: int | None = None,
        layer_timeout_s: float = 60.0,
        cls_timeout_s: float = 30.0,
        ps_vision_backend: str = "numpy",
        fclk0_mhz: float | None = DEFAULT_FCLK0_MHZ,
        max_image_splits: int | None = DEFAULT_MAX_IMAGE_SPLITS,
        preload_processor: bool = False,
    ):
        self.dry_run = dry_run
        self.layer_timeout_s = layer_timeout_s
        self.cls_timeout_s = cls_timeout_s
        self.overlay_reload_interval = int(overlay_reload_interval) if overlay_reload_interval else None
        self.fclk0_mhz = fclk0_mhz
        self.max_image_splits = int(max_image_splits) if max_image_splits else None
        self.overlay_reload_count = 0
        self.last_input_info: dict[str, Any] = {}
        self.last_prepare_timings: dict[str, float] = {}
        self.last_setup_timings: dict[str, float] = {}
        self.acc = None
        self.processor = None
        self.tokenizer = None

        if dry_run:
            return

        self.acc = create_accelerator(
            load_overlay=load_overlay,
            load_weights=load_weights,
            load_ps_float=load_ps_float,
            enable_vit=enable_vit,
            load_ps_vision=load_ps_vision,
            load_text_fp16=load_text_fp16,
            ps_vision_backend=ps_vision_backend,
            fclk0_mhz=fclk0_mhz,
        )
        self._load_tokenizer()
        if preload_processor:
            self._load_processor(self.last_setup_timings)

    def _load_tokenizer(self) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required on board for tokenizer/processor paths") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

    def _load_processor(self, timings: dict[str, float] | None = None) -> None:
        if self.processor is not None:
            return
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("transformers is required on board for multimodal processor paths") from exc
        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        if self.max_image_splits is not None:
            patch_image_processor_max_splits(self.processor, self.max_image_splits)
        if self.tokenizer is None:
            self.tokenizer = getattr(self.processor, "tokenizer", None)
        elapsed = time.perf_counter() - t0
        self.last_setup_timings["processor_load"] = elapsed
        if timings is not None:
            timings["processor_load"] = elapsed

    def _text_inputs(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        if self.tokenizer is None:
            raise RuntimeError("tokenizer is not initialized")
        timings: dict[str, float] = {}
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        t0 = time.perf_counter()
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="np",
        )
        timings["text_tokenizer_chat_template"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        inputs = extract_input_ids(inputs)
        input_ids = np.asarray(inputs, dtype=np.int64).reshape(-1)
        timings["text_input_ids_numpy"] = time.perf_counter() - t0
        # text-only 路径对齐 Qwen：优先查 int16 固定点 embedding 表，
        # 直接形成 PL decoder state 的 int32 表示，避免板上实时 fp16->int 转换误差。
        t0 = time.perf_counter()
        embeddings = self.acc.ps_text_embeddings_i32(input_ids)
        timings["text_embedding_lookup_i32"] = time.perf_counter() - t0
        self.last_prepare_timings = timings
        return input_ids, embeddings

    def _image_inputs(self, image_path: str, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        timings: dict[str, float] = {}
        self._load_processor(timings)
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required on board for image evaluation paths") from exc

        t0 = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        timings["processor_open_image"] = time.perf_counter() - t0
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        t0 = time.perf_counter()
        prompt_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        timings["processor_chat_template"] = time.perf_counter() - t0
        use_fast_processor = os.environ.get("VLM_USE_FAST_IMAGE_PROCESSOR", "1") != "0"
        processor_t0 = time.perf_counter()
        if use_fast_processor:
            t0 = time.perf_counter()
            pixel_values, pixel_attention_mask, image_rows, image_cols = _fast_smolvlm_image_inputs(
                image,
                self.processor.image_processor,
                self.max_image_splits,
            )
            timings["processor_fast_image"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            expanded_text = self.processor.expand_text_with_image_tokens(
                [prompt_text],
                image_rows=image_rows,
                image_cols=image_cols,
            )
            tokenized = self.tokenizer(expanded_text, return_tensors="np")
            timings["processor_fast_tokenizer"] = time.perf_counter() - t0
            inputs = {
                "input_ids": tokenized["input_ids"],
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_attention_mask,
            }
            timings["processor_fast_path"] = 1.0
        else:
            inputs = self.processor(
                text=prompt_text,
                images=[image],
                return_tensors="np",
            )
            timings["processor_fast_path"] = 0.0
        timings["processor_call"] = time.perf_counter() - processor_t0
        t0 = time.perf_counter()
        input_ids = np.asarray(inputs["input_ids"], dtype=np.int64).reshape(-1)
        pixel_values = np.asarray(inputs["pixel_values"], dtype=np.float32)
        pixel_attention_mask = inputs.get("pixel_attention_mask")
        image_tokens = int((input_ids == IMAGE_TOKEN_ID).sum())
        batch = self.acc.ps_float.flatten_processor_images(pixel_values, pixel_attention_mask)
        timings["processor_parse_and_flatten"] = time.perf_counter() - t0
        self.last_input_info = {
            "max_image_splits": self.max_image_splits,
            "processor_pixel_values_shape": list(pixel_values.shape),
            "real_subimages": int(batch.pixel_values.shape[0]),
            "real_indices": [int(x) for x in batch.real_indices.tolist()],
            "image_tokens": image_tokens,
            "expected_image_tokens": int(batch.pixel_values.shape[0]) * CONNECTOR_OUT_TOKENS,
        }
        t0 = time.perf_counter()
        embeddings = self.acc.build_merged_embeddings(
            input_ids,
            pixel_values,
            pixel_attention_mask,
            image_batch=batch,
            timings=timings,
        )
        timings["build_merged_embeddings_total"] = time.perf_counter() - t0
        self.last_prepare_timings = timings
        return input_ids, embeddings

    @staticmethod
    def _embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings)
        if x.size % C != 0:
            raise ValueError(f"embedding element count {x.size} is not divisible by C={C}")
        if np.issubdtype(x.dtype, np.integer):
            return x.astype(np.int32, copy=False).reshape(-1, C)
        return x.astype(np.float32, copy=False).reshape(-1, C)

    @staticmethod
    def _chunk_for_pos(embeddings: np.ndarray, pos: int) -> np.ndarray:
        # decoder_state 只保存当前 8-token tile；lane0 对应该 tile 的绝对起始位置。
        x = LocalPynqRunner._embedding_matrix(embeddings)
        if pos < 0 or pos >= x.shape[0]:
            raise ValueError(f"position {pos} is outside embeddings length {x.shape[0]}")
        chunk_start = (pos // T) * T
        valid = pos - chunk_start + 1
        chunk = np.zeros((T, C), dtype=x.dtype)
        chunk[:valid] = x[chunk_start : chunk_start + valid]
        return chunk

    @staticmethod
    def _token_from_cls(cls_ids: np.ndarray, pos: int) -> int:
        # lm_head 对当前 tile 的 8 个 lane 都输出 argmax；只取当前绝对 pos 所在 lane。
        ids = np.asarray(cls_ids, dtype=np.int64).reshape(-1)
        lane = pos % T
        if ids.size <= lane:
            raise ValueError(f"CLS output has {ids.size} ids, cannot read lane {lane}")
        return int(ids[lane])

    def _decode_ids(self, ids: list[int]) -> str:
        if self.tokenizer is None:
            return " ".join(str(x) for x in ids)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _recover_accelerator_after_error(self) -> None:
        # 真实 timeout 后 PL 内部 FIFO/FSM 可能仍停在半次事务中。评测循环继续跑
        # 下一条样本前先重载 overlay，避免一个失败样本污染后续所有样本。
        if self.acc is None:
            return
        self.acc.reload_overlay()
        self.overlay_reload_count += 1

    def _run_llm_tile(
        self,
        embeddings: np.ndarray,
        pos: int,
        cache_update_mode: int = CACHE_UPDATE_FULL_TILE,
        timings: dict[str, Any] | None = None,
        phase: str = "llm",
    ) -> None:
        tile_t0 = time.perf_counter()
        # 当前硬件按 tile 工作；partial tile 用 pos 的 chunk_pos 屏蔽未来 token。
        # cache_update_mode 是 raw-V/current-only 实验遗留参数；当前 HLS 已忽略。
        chunk = self._chunk_for_pos(embeddings, pos)
        t0 = time.perf_counter()
        if np.issubdtype(chunk.dtype, np.integer):
            self.acc.write_decoder_state_from_int32(chunk)
        else:
            self.acc.write_decoder_state_from_float(chunk)
        add_timing(timings, f"{phase}_write_decoder_state", time.perf_counter() - t0)
        layers_t0 = time.perf_counter()
        for layer in range(LLAMA_L):
            layer_t0 = time.perf_counter()
            self.acc.run_llm_layer(
                layer,
                pos=pos,
                timeout_s=self.layer_timeout_s,
                cache_update_mode=cache_update_mode,
            )
            layer_elapsed = time.perf_counter() - layer_t0
            add_timing(timings, f"{phase}_layer_{layer:02d}", layer_elapsed)
            add_timing(timings, f"{phase}_layer_total", layer_elapsed)
            add_count(timings, f"{phase}_layer_calls")
            if (
                self.overlay_reload_interval is not None
                and layer + 1 < LLAMA_L
                and (layer + 1) % self.overlay_reload_interval == 0
            ):
                # 当前 bitstream 连续多层后会残留内部 AXIS/HLS 状态；reload PL
                # 只清硬件内部状态，不改 DDR 中的 state/KV/weights。
                reload_t0 = time.perf_counter()
                self.acc.reload_overlay()
                self.overlay_reload_count += 1
                add_timing(timings, f"{phase}_overlay_reload", time.perf_counter() - reload_t0)
        add_timing(timings, f"{phase}_layers_total", time.perf_counter() - layers_t0)
        add_timing(timings, f"{phase}_tile_total", time.perf_counter() - tile_t0)
        add_count(timings, f"{phase}_tile_runs")

    def _run_lm_head_for_pos(
        self,
        pos: int,
        timings: dict[str, Any] | None = None,
        phase: str = "llm",
    ) -> int:
        t0 = time.perf_counter()
        cls_ids = self.acc.run_llm_cls(pos=pos, timeout_s=self.cls_timeout_s)
        add_timing(timings, f"{phase}_cls_total", time.perf_counter() - t0)
        add_count(timings, f"{phase}_cls_calls")
        return self._token_from_cls(cls_ids, pos)

    def _prefill_prompt(self, embeddings: np.ndarray, timings: dict[str, Any] | None = None) -> int:
        # prefill 阶段使用 FULL_TILE：一个 8-token chunk 只跑一次，并写入该 chunk 的全部有效 KV。
        # 最后一块 partial tile 用 pos 的 chunk_pos 限制有效 lane，未来 padding 不落 cache。
        total = int(self._embedding_matrix(embeddings).shape[0])
        if total <= 0:
            raise ValueError("prompt produced no input tokens")
        for chunk_start in range(0, total, T):
            pos = min(chunk_start + T - 1, total - 1)
            self._run_llm_tile(
                embeddings,
                pos,
                cache_update_mode=CACHE_UPDATE_FULL_TILE,
                timings=timings,
                phase="prefill",
            )
        return self._run_lm_head_for_pos(total - 1, timings=timings, phase="prefill")

    def _annotate_decode_roofline(self, timings: dict[str, Any], generated_tokens: int) -> None:
        # DEC-01 诊断：当前 bitstream 不能安全单独启动 WEIGHT_AXI producer。
        # 因此板上实测完整 layer/CLS 时间，同时用 packed 权重文件大小估算
        # decode 阶段的有效权重 DDR 带宽。
        decoder_w_bytes = int(os.path.getsize(os.path.join(BIN_DIR, PL_BIN_FILES["decoder_w"])))
        cls_w_bytes = int(os.path.getsize(os.path.join(BIN_DIR, PL_BIN_FILES["cls_w"])))
        prefill_tiles = int(timings.get("prefill_tile_runs", 0))
        prefill_cls = int(timings.get("prefill_cls_calls", 0))
        decode_tiles = int(timings.get("decode_tile_runs", 0))
        decode_cls = int(timings.get("decode_cls_calls", 0))

        timings["roofline_decoder_w_bytes"] = decoder_w_bytes
        timings["roofline_cls_w_bytes"] = cls_w_bytes
        timings["prefill_weight_bytes_estimated"] = prefill_tiles * decoder_w_bytes + prefill_cls * cls_w_bytes
        timings["decode_weight_bytes_estimated"] = decode_tiles * decoder_w_bytes + decode_cls * cls_w_bytes
        timings["decode_generated_tokens"] = int(generated_tokens)
        timings["decode_tile_run_tokens"] = decode_tiles
        timings["decode_first_token_from_prefill"] = max(0, int(generated_tokens) - decode_tiles)

        decode_s = float(timings.get("decode", 0.0))
        if decode_s > 0.0:
            timings["decode_effective_weight_gbs_estimated"] = timings["decode_weight_bytes_estimated"] / decode_s / 1e9
            timings["decode_effective_completion_tps"] = float(generated_tokens) / decode_s if generated_tokens else 0.0
            timings["decode_effective_tile_tps"] = float(decode_tiles) / decode_s if decode_tiles else 0.0
        if decode_tiles > 0:
            timings["decode_tile_avg_s"] = float(timings.get("decode_tile_total", 0.0)) / decode_tiles
            timings["decode_layer_avg_s"] = float(timings.get("decode_layer_total", 0.0)) / (decode_tiles * LLAMA_L)
            timings["decode_cls_avg_s"] = float(timings.get("decode_cls_total", 0.0)) / max(1, decode_cls)

    def _generate_from_embeddings(
        self,
        input_ids: np.ndarray,
        embeddings: np.ndarray,
        max_new_tokens: int,
        ignore_eos: bool = False,
    ) -> tuple[list[int], dict[str, float]]:
        if max_new_tokens <= 0:
            return [], {"prefill": 0.0, "decode": 0.0}
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        all_embeddings = self._embedding_matrix(embeddings)
        if ids.size != all_embeddings.shape[0]:
            raise ValueError(f"input_ids length {ids.size} != embeddings length {all_embeddings.shape[0]}")
        if ids.size + max_new_tokens > S:
            raise ValueError(f"KV cache supports at most {S} positions, got prompt {ids.size} + max_new_tokens {max_new_tokens}")

        generated: list[int] = []
        timings: dict[str, Any] = {}

        t0 = time.perf_counter()
        next_id = self._prefill_prompt(all_embeddings, timings=timings)
        timings["prefill"] = time.perf_counter() - t0

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        t0 = time.perf_counter()
        while len(generated) < max_new_tokens:
            generated.append(int(next_id))
            if not ignore_eos and eos_id is not None and int(next_id) == int(eos_id):
                break
            if len(generated) >= max_new_tokens:
                break

            # 生成出的 token 进入下一绝对位置。当前硬件语义回到 8-token tile 重跑：
            # 同一 chunk 内已知 token 会一起重算，并覆盖量化后的 K/V cache chunk。
            token_t0 = time.perf_counter()
            if np.issubdtype(all_embeddings.dtype, np.integer):
                token_embedding = self.acc.ps_text_embeddings_i32(np.asarray([next_id], dtype=np.int64))
            else:
                # 多模态 prompt 含 image features，因此整体 embedding 是 float；
                # 但文本 token 本身仍要走 PYNQ/HLS 对齐过的 int16 embedding 表。
                token_embedding = (
                    self.acc.ps_text_embeddings_i32(np.asarray([next_id], dtype=np.int64)).astype(np.float32)
                    * float(DEFAULT_DECODER_STATE_SCALE)
                )
            add_timing(timings, "decode_token_embedding_lookup", time.perf_counter() - token_t0)
            all_embeddings = np.concatenate([all_embeddings, token_embedding.reshape(1, C)], axis=0)
            pos = all_embeddings.shape[0] - 1
            self._run_llm_tile(
                all_embeddings,
                pos,
                cache_update_mode=CACHE_UPDATE_FULL_TILE,
                timings=timings,
                phase="decode",
            )
            next_id = self._run_lm_head_for_pos(pos, timings=timings, phase="decode")

        timings["decode"] = time.perf_counter() - t0
        self._annotate_decode_roofline(timings, generated_tokens=len(generated))
        return generated, timings

    def generate_once(
        self,
        prompt: str,
        image_path: str | None = None,
        max_new_tokens: int | None = T,
        ignore_eos: bool = False,
    ) -> LocalGenerationResult:
        """Run local PYNQ prefill/decode and return a structured result."""
        mode = "multimodal" if image_path else "text"

        if self.dry_run:
            output_ids = [0]
            return LocalGenerationResult(
                status="dry_run",
                mode=mode,
                prompt=prompt,
                image_path=image_path,
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(output_ids),
                output_ids=output_ids,
                output_text="[dry-run] hardware path not executed",
                timings_s={"total": 0.0},
                fclk0_mhz=self.fclk0_mhz,
                max_image_splits=self.max_image_splits,
            )

        result = LocalGenerationResult(status="running", mode=mode, prompt=prompt, image_path=image_path)
        result.fclk0_mhz = self.fclk0_mhz
        result.max_image_splits = self.max_image_splits
        result.overlay_reload_interval = self.overlay_reload_interval
        result.timings_s.update({f"setup_{k}": v for k, v in self.last_setup_timings.items()})
        t_total0 = time.perf_counter()
        try:
            self.acc.reset_kv_cache()
            self.last_input_info = {}
            self.last_prepare_timings = {}

            t0 = time.perf_counter()
            if image_path:
                input_ids, embeddings = self._image_inputs(image_path, prompt)
            else:
                input_ids, embeddings = self._text_inputs(prompt)
            t1 = time.perf_counter()
            result.prompt_tokens = int(input_ids.size)
            result.timings_s["prepare_inputs"] = t1 - t0
            result.timings_s.update({f"prepare_{k}": v for k, v in self.last_prepare_timings.items()})
            result.input_info = dict(self.last_input_info)
            if max_new_tokens is None:
                max_new_tokens = max(0, S - int(input_ids.size))
                result.input_info["auto_max_new_tokens"] = int(max_new_tokens)
                if ignore_eos:
                    result.input_info["auto_max_new_tokens_policy"] = "decode until KV-cache capacity; EOS is ignored"
                else:
                    result.input_info["auto_max_new_tokens_policy"] = "decode until EOS or KV-cache capacity"
            result.input_info["ignore_eos"] = bool(ignore_eos)

            t0 = time.perf_counter()
            output_ids, llm_timings = self._generate_from_embeddings(
                input_ids,
                embeddings,
                max_new_tokens=max_new_tokens,
                ignore_eos=ignore_eos,
            )
            t1 = time.perf_counter()

            result.output_ids = output_ids
            result.completion_tokens = len(output_ids)
            result.output_text = self._decode_ids(output_ids)
            result.timings_s.update({f"llm_{k}": v for k, v in llm_timings.items()})
            result.timings_s["llm_total"] = t1 - t0
            result.overlay_reloads = int(self.overlay_reload_count)
            result.register_dump = self.acc.dump_registers()
            result.status = "ok"
        except Exception as exc:
            result.status = "error"
            result.error = str(exc)
            result.traceback = traceback.format_exc()
            try:
                if self.acc is not None:
                    result.register_dump = self.acc.dump_registers()
            except Exception:
                pass
            if isinstance(exc, TimeoutError) or "accelerator did" in str(exc):
                try:
                    self._recover_accelerator_after_error()
                except Exception as recovery_exc:
                    result.error = f"{result.error}; recovery failed: {type(recovery_exc).__name__}: {recovery_exc}"
        finally:
            result.overlay_reloads = int(self.overlay_reload_count)
            result.timings_s["total"] = time.perf_counter() - t_total0
        return result


def patch_image_processor_max_splits(processor: Any, max_splits: int) -> None:
    """限制 SmolVLM image processor 每张图最多生成 `max_splits` 个子图。

    该逻辑从软件参考 `software/modeling/__init__.py` 移植到 PYNQ PS 端。
    processor 原本会先 resize 到 512 的倍数，再按 512x512 crop 并额外加
    global image；这里在 resize 后检查 `N = ceil(h/512)*ceil(w/512)+1`，
    超限时选择不超过 `max_splits-1` 个 crop 的最接近宽高比网格重新 resize。
    """
    if max_splits <= 1:
        raise ValueError(f"max_splits must be >= 2 when enabled, got {max_splits}")
    img_proc = processor.image_processor
    if getattr(img_proc, "_vlm_max_splits", None) == int(max_splits):
        return

    max_image_size = int(img_proc.max_image_size["longest_edge"])
    original_resize = img_proc.resize_for_vision_encoder

    def capped_resize_for_vision_encoder(image, *args, **kwargs):
        resized = original_resize(image, *args, **kwargs)

        from transformers.image_utils import get_image_size, infer_channel_dimension_format

        input_data_format = infer_channel_dimension_format(resized, num_channels=(1, 3, 4))
        h, w = get_image_size(resized, channel_dim=input_data_format)
        n_h = math.ceil(h / max_image_size)
        n_w = math.ceil(w / max_image_size)
        current_n = n_h * n_w + 1
        if current_n <= max_splits:
            return resized

        max_crops = max_splits - 1
        aspect_ratio = w / h
        best_nh, best_nw = 1, 1
        best_ar_diff = float("inf")
        for nh in range(1, max_crops + 1):
            for nw in range(1, max_crops + 1):
                if nh * nw > max_crops:
                    continue
                candidate_ar = nw / nh
                ar_diff = abs(candidate_ar - aspect_ratio)
                if ar_diff < best_ar_diff or (ar_diff == best_ar_diff and nh * nw > best_nh * best_nw):
                    best_ar_diff = ar_diff
                    best_nh = nh
                    best_nw = nw

        try:
            from transformers.image_utils import SizeDict

            resize_size = SizeDict(height=best_nh * max_image_size, width=best_nw * max_image_size)
        except Exception:
            resize_size = {"height": best_nh * max_image_size, "width": best_nw * max_image_size}

        return img_proc.resize(
            resized,
            size=resize_size,
            input_data_format=input_data_format,
        )

    img_proc.resize_for_vision_encoder = capped_resize_for_vision_encoder
    img_proc._vlm_max_splits = int(max_splits)
