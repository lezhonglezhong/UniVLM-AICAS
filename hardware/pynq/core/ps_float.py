"""
PS 端浮点 baseline：embedding、ViT final post-LayerNorm、Connector 和 merge。

这些函数只依赖 numpy，可在本机做形状/逻辑验证，也可在 KV260 PS 端运行。
PL 输出的 ViT state 是 int32 定点；进入浮点 final post-LN 前必须显式传入
`vit_state_scale` 还原到近似浮点值。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np

CORE_DIR = os.path.dirname(__file__)
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
_local_constants = os.path.abspath(os.path.join(CORE_DIR, "constants.py"))
_loaded_constants = sys.modules.get("constants")
if _loaded_constants is not None and os.path.abspath(getattr(_loaded_constants, "__file__", "")) != _local_constants:
    del sys.modules["constants"]

from constants import (
    BIN_DIR,
    C,
    CONNECTOR_IN_DIM,
    CONNECTOR_OUT_DIM,
    CONNECTOR_SCALE,
    CONNECTOR_OUT_TOKENS,
    DEFAULT_DECODER_STATE_SCALE,
    IMAGE_TOKEN_ID,
    PS_FLOAT_FILES,
    VIT_C,
    VIT_IMG_SIZE,
    VIT_LN_EPS,
    VIT_PATCH_SIZE,
    VIT_SEQ_LEN,
)


@dataclass
class PSFloatShapes:
    text_embedding: tuple[int, int] = (49280, C)
    patch_w: tuple[int, int, int, int] = (VIT_C, 3, VIT_PATCH_SIZE, VIT_PATCH_SIZE)
    patch_b: tuple[int] = (VIT_C,)
    position_embedding: tuple[int, int] = (VIT_SEQ_LEN, VIT_C)
    post_ln_w: tuple[int] = (VIT_C,)
    post_ln_b: tuple[int] = (VIT_C,)
    connector_w: tuple[int, int] = (CONNECTOR_OUT_DIM, CONNECTOR_IN_DIM)


@dataclass
class ProcessorImageBatch:
    """Processor 输出中过滤 padding 后的真实 ViT 子图。"""

    pixel_values: np.ndarray
    patch_attention_mask: np.ndarray
    real_indices: np.ndarray


class PSFloatAssets:
    """加载和执行 PS 端浮点 baseline 资产。"""

    def __init__(
        self,
        bin_dir: str = BIN_DIR,
        load_vision: bool = True,
        load_text_fp16: bool = True,
        load_text_i16: bool = True,
        vision_backend: str = "numpy",
    ):
        self.bin_dir = bin_dir
        if vision_backend not in ("numpy", "torch"):
            raise ValueError(f"vision_backend must be 'numpy' or 'torch', got {vision_backend!r}")
        self.vision_backend = vision_backend
        self.shapes = PSFloatShapes()
        self.manifest = self._load_manifest()

        # text-only 轻量路径只需要 int16 定点 embedding。多模态 merge 也默认
        # 使用同一张 int16 表；vision/connector 浮点资产按 manifest 读取，
        # 当前为 fp32，用于匹配软件 float32 PS 边界。
        self.text_embedding = self._load("text_embedding", self.shapes.text_embedding) if load_text_fp16 else None
        # Qwen PYNQ 直接查 int16 embedding 表。SmolVLM2 的文本 embedding 按
        # O_res=12 固定点量化后也能放入 int16；text-only 路径优先用它直接写 PL state。
        self.text_embedding_i16 = (
            self._load_i16_optional("text_embedding_i16", self.shapes.text_embedding) if load_text_i16 else None
        )
        if load_vision:
            # 资产按 fp16 存储，计算时提升到 fp32，降低 ARM 端重复转换的误差风险。
            self.patch_w = self._load("patch_w", self.shapes.patch_w).astype(np.float32)
            self.patch_b = self._load("patch_b", self.shapes.patch_b).astype(np.float32)
            self.position_embedding = self._load("position_embedding", self.shapes.position_embedding).astype(np.float32)
            self.post_ln_w = self._load("vision_post_ln_w", self.shapes.post_ln_w).astype(np.float32)
            self.post_ln_b = self._load("vision_post_ln_b", self.shapes.post_ln_b).astype(np.float32)
            self.connector_w = self._load("connector_w", self.shapes.connector_w).astype(np.float32)
        else:
            self.patch_w = None
            self.patch_b = None
            self.position_embedding = None
            self.post_ln_w = None
            self.post_ln_b = None
            self.connector_w = None

    def _torch_dtype(self):
        """PS debug 后端使用 torch.bfloat16，对齐远端 acc_eval 默认 dtype。"""
        import torch

        return torch.bfloat16

    def _torch_tensor(self, arr: np.ndarray):
        """把 NumPy 资产转成 torch BF16；仅 torch debug backend 调用。"""
        import torch

        return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(dtype=self._torch_dtype())

    @staticmethod
    def _torch_pixel_shuffle(x):
        """复现官方 SmolVLMConnector.pixel_shuffle 的 torch 版本。"""
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        scale = CONNECTOR_SCALE
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale), embed_dim * scale)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale), int(height / scale), embed_dim * (scale**2))
        x = x.permute(0, 2, 1, 3)
        return x.reshape(bsz, int(seq / (scale**2)), embed_dim * (scale**2))

    @staticmethod
    def _torch_position_ids(patch_attention_mask, patches_per_side: int):
        """
        复现官方 SmolVLMVisionEmbeddings 的 position id 计算。

        注意 fractional coords 会按官方实现转到 pixel dtype；torch backend
        中 pixel dtype 为 BF16，这是和 NumPy/float32 路径的一个关键差异。
        """
        import torch

        mask = patch_attention_mask.to(dtype=torch.bool)
        batch_size = mask.shape[0]
        boundaries = torch.arange(1 / patches_per_side, 1.0, 1 / patches_per_side, device=mask.device)
        position_ids = torch.full(
            size=(batch_size, patches_per_side * patches_per_side),
            fill_value=0,
            device=mask.device,
            dtype=torch.long,
        )
        nb_h = mask[:, :, 0].sum(dim=1)
        nb_w = mask[:, 0, :].sum(dim=1)
        step_h = 1.0 / nb_h
        step_w = 1.0 / nb_w
        h_indices = torch.arange(patches_per_side, device=mask.device, dtype=torch.float32)
        w_indices = torch.arange(patches_per_side, device=mask.device, dtype=torch.float32)
        coords_h = torch.clamp(h_indices[None, :] * step_h[:, None], max=(1.0 - 1e-6))
        coords_w = torch.clamp(w_indices[None, :] * step_w[:, None], max=(1.0 - 1e-6))
        coords_h = coords_h.to(dtype=torch.bfloat16)
        coords_w = coords_w.to(dtype=torch.bfloat16)
        bucket_h = torch.bucketize(coords_h, boundaries, right=True)
        bucket_w = torch.bucketize(coords_w, boundaries, right=True)
        pos = (bucket_h[:, :, None] * patches_per_side + bucket_w[:, None, :]).reshape(batch_size, -1)
        flat_mask = mask.view(batch_size, -1)
        position_ids[flat_mask] = pos[flat_mask]
        return position_ids

    def _load_manifest(self) -> dict:
        path = self._asset_path("manifest")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _asset_path(self, name: str) -> str:
        """解析 PS 资产路径；允许板上旧 constants.py 暂缺新增的可选整数表 key。"""
        optional_defaults = {
            "text_embedding_i16": "text_embedding_i16.bin",
        }
        filename = PS_FLOAT_FILES.get(name, optional_defaults.get(name))
        if filename is None:
            raise KeyError(name)
        return os.path.join(self.bin_dir, filename)

    def _float_dtype_for_asset(self, name: str, path: str) -> np.dtype:
        """根据 manifest/文件名选择 PS 浮点资产 dtype，兼容旧 fp16 资产。"""
        filename = os.path.basename(path)
        entry = self.manifest.get("assets", {}).get(filename, {})
        export_dtype = entry.get("export_dtype")
        if export_dtype in ("float32", "fp32"):
            return np.dtype("<f4")
        if export_dtype in ("float16", "fp16"):
            return np.dtype("<f2")
        if filename.endswith("_fp32.bin"):
            return np.dtype("<f4")
        if filename.endswith("_fp16.bin"):
            return np.dtype("<f2")
        raise ValueError(f"Cannot infer floating-point dtype for PS asset {name}: {path}")

    def _load(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        path = self._asset_path(name)
        dtype = self._float_dtype_for_asset(name, path)
        arr = np.fromfile(path, dtype=dtype)
        expected = int(np.prod(shape))
        if arr.size != expected:
            raise ValueError(f"{path}: got {arr.size} {dtype} elements, expected {expected}")
        return arr.reshape(shape)

    def _load_i16_optional(self, name: str, shape: tuple[int, ...]) -> np.ndarray | None:
        path = self._asset_path(name)
        if not os.path.exists(path):
            return None
        arr = np.fromfile(path, dtype="<i2")
        expected = int(np.prod(shape))
        if arr.size != expected:
            raise ValueError(f"{path}: got {arr.size} int16 elements, expected {expected}")
        return arr.reshape(shape)

    def text_lookup(self, input_ids: np.ndarray | list[int]) -> np.ndarray:
        """文本 token embedding 查表，输出 float32 [S, 960]。"""
        if self.text_embedding is None:
            raise RuntimeError("text_embedding fp16 asset was not loaded; enable load_text_fp16 for multimodal merge")
        ids = np.asarray(input_ids, dtype=np.int64)
        return self.text_embedding[ids].astype(np.float32)

    def text_lookup_i32(self, input_ids: np.ndarray | list[int]) -> np.ndarray:
        """
        文本 token embedding 查表，输出 PL decoder state 的 int32 固定点值。

        若板上暂未复制 `text_embedding_i16.bin`，回退到旧 fp16 表并按
        `O_res=12` 量化，保持 bring-up 脚本可运行。
        """
        ids = np.asarray(input_ids, dtype=np.int64)
        if self.text_embedding_i16 is not None:
            return self.text_embedding_i16[ids].astype(np.int32)
        if self.text_embedding is None:
            raise RuntimeError("text_embedding_i16.bin is required when text_embedding fp16 asset is not loaded")
        x = self.text_embedding[ids].astype(np.float32)
        return np.rint(x / DEFAULT_DECODER_STATE_SCALE).astype(np.int32)

    def _require_vision_asset(self, name: str) -> np.ndarray:
        arr = getattr(self, name)
        if arr is None:
            raise RuntimeError(f"{name} was not loaded; construct PSFloatAssets with load_vision=True")
        return arr

    @staticmethod
    def flatten_processor_images(
        pixel_values: np.ndarray,
        pixel_attention_mask: np.ndarray | None = None,
    ) -> ProcessorImageBatch:
        """
        复现 SmolVLMModel.get_image_features 的多子图展开和 padding 过滤。

        Processor 对一张原图可能产生多个 512x512 子图，形态通常为
        `[B,N_images,3,512,512]`。PL ViT 一次只处理一个子图，因此这里返回
        flatten 后的真实子图 `[N_real,3,512,512]` 和对应 patch mask
        `[N_real,32,32]`，供 PYNQ 串行调度。
        """
        pixels = np.asarray(pixel_values, dtype=np.float32)
        if pixels.ndim == 5:
            if pixels.shape[2:] != (3, VIT_IMG_SIZE, VIT_IMG_SIZE):
                raise ValueError(f"pixel_values shape must be [B,N,3,{VIT_IMG_SIZE},{VIT_IMG_SIZE}], got {pixels.shape}")
            flat_pixels = pixels.reshape(-1, *pixels.shape[2:])
        elif pixels.ndim == 4:
            if pixels.shape[1:] != (3, VIT_IMG_SIZE, VIT_IMG_SIZE):
                raise ValueError(f"pixel_values shape must be [N,3,{VIT_IMG_SIZE},{VIT_IMG_SIZE}], got {pixels.shape}")
            flat_pixels = pixels
        else:
            raise ValueError(f"pixel_values must be rank 4 or 5, got {pixels.shape}")

        nb_values_per_image = int(np.prod(flat_pixels.shape[1:]))
        real = np.any(flat_pixels != 0.0, axis=(1, 2, 3))
        if real.size == 0:
            raise ValueError("pixel_values contains no images")
        if not real.any():
            real[0] = True

        real_indices = np.flatnonzero(real).astype(np.int64)
        real_pixels = flat_pixels[real]
        patch_mask = PSFloatAssets._patch_mask_from_attention(pixel_attention_mask, pixels.shape, real)
        return ProcessorImageBatch(real_pixels, patch_mask, real_indices)

    @staticmethod
    def _patch_mask_from_attention(
        pixel_attention_mask: np.ndarray | None,
        pixel_values_shape: tuple[int, ...],
        real_mask: np.ndarray,
    ) -> np.ndarray:
        patches_per_side = VIT_IMG_SIZE // VIT_PATCH_SIZE
        if pixel_attention_mask is None:
            return np.ones((int(real_mask.sum()), patches_per_side, patches_per_side), dtype=bool)

        mask = np.asarray(pixel_attention_mask, dtype=bool)
        if mask.ndim == 4:
            flat_mask = mask.reshape(-1, *mask.shape[-2:])
        elif mask.ndim == 3:
            flat_mask = mask
        else:
            raise ValueError(f"pixel_attention_mask must be rank 3 or 4, got {mask.shape}")

        expected_flat = int(np.prod(pixel_values_shape[:2])) if len(pixel_values_shape) == 5 else pixel_values_shape[0]
        if flat_mask.shape[0] != expected_flat:
            raise ValueError(f"pixel_attention_mask first dimension {flat_mask.shape[0]} != flattened images {expected_flat}")

        flat_mask = flat_mask[real_mask]
        if flat_mask.shape[-2:] == (patches_per_side, patches_per_side):
            return flat_mask.astype(bool, copy=False)
        if flat_mask.shape[-2:] != (VIT_IMG_SIZE, VIT_IMG_SIZE):
            raise ValueError(
                "pixel_attention_mask spatial shape must be "
                f"({VIT_IMG_SIZE},{VIT_IMG_SIZE}) or ({patches_per_side},{patches_per_side}), got {flat_mask.shape[-2:]}"
            )

        bsz = flat_mask.shape[0]
        patch_mask = flat_mask.reshape(
            bsz,
            patches_per_side,
            VIT_PATCH_SIZE,
            patches_per_side,
            VIT_PATCH_SIZE,
        )
        patch_mask = patch_mask.transpose(0, 1, 3, 2, 4)
        return patch_mask.any(axis=(-1, -2))

    def patch_position_embedding(self, pixel_values: np.ndarray, patch_attention_mask: np.ndarray | None = None) -> np.ndarray:
        """
        PS 浮点 ViT patch embedding + position embedding。

        `pixel_values` 应来自 SmolVLM2 processor，shape 为 [B, 3, 512, 512]。
        `patch_attention_mask` 若为空，默认全部 32x32 patch 有效。
        """
        x = np.asarray(pixel_values, dtype=np.float32)
        if x.ndim != 4 or x.shape[1:] != (3, VIT_IMG_SIZE, VIT_IMG_SIZE):
            raise ValueError(f"pixel_values shape must be [B,3,{VIT_IMG_SIZE},{VIT_IMG_SIZE}], got {x.shape}")

        if self.vision_backend == "torch":
            return self._torch_patch_position_embedding(x, patch_attention_mask)

        bsz = x.shape[0]
        patches_per_side = VIT_IMG_SIZE // VIT_PATCH_SIZE
        patches = x.reshape(
            bsz,
            3,
            patches_per_side,
            VIT_PATCH_SIZE,
            patches_per_side,
            VIT_PATCH_SIZE,
        )
        # patch 顺序对齐 torch Conv2d flatten(2).transpose(1,2)：row-major patch，再按 C/H/W 展平。
        patches = patches.transpose(0, 2, 4, 1, 3, 5).reshape(bsz, VIT_SEQ_LEN, 3 * VIT_PATCH_SIZE * VIT_PATCH_SIZE)
        patch_w = self._require_vision_asset("patch_w")
        patch_b = self._require_vision_asset("patch_b")
        position_embedding = self._require_vision_asset("position_embedding")
        weight = patch_w.reshape(VIT_C, -1)
        out = patches @ weight.T
        out += patch_b

        pos_ids = self.compute_position_ids(bsz, patch_attention_mask, patches_per_side)
        out += position_embedding[pos_ids]
        return out.astype(np.float32, copy=False)

    def _torch_patch_position_embedding(
        self,
        pixel_values: np.ndarray,
        patch_attention_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """torch/BF16 版 patch+position embedding，用于和官方软件路径做板上 A/B。"""
        import torch
        import torch.nn.functional as F

        self._require_vision_asset("patch_w")
        self._require_vision_asset("patch_b")
        self._require_vision_asset("position_embedding")
        x = torch.from_numpy(np.asarray(pixel_values, dtype=np.float32)).to(dtype=self._torch_dtype())
        weight = self._torch_tensor(self.patch_w)
        bias = self._torch_tensor(self.patch_b)
        patch_embeds = F.conv2d(x, weight, bias=bias, stride=VIT_PATCH_SIZE)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        patches_per_side = VIT_IMG_SIZE // VIT_PATCH_SIZE
        if patch_attention_mask is None:
            mask_np = np.ones((x.shape[0], patches_per_side, patches_per_side), dtype=bool)
        else:
            mask_np = np.asarray(patch_attention_mask).astype(bool)
        mask = torch.from_numpy(mask_np)
        position_ids = self._torch_position_ids(mask, patches_per_side)
        pos_weight = self._torch_tensor(self.position_embedding)
        embeddings = embeddings + F.embedding(position_ids, pos_weight)
        return embeddings.to(dtype=torch.float32).cpu().numpy().astype(np.float32)

    @staticmethod
    def compute_position_ids(bsz: int, patch_attention_mask: np.ndarray | None, patches_per_side: int) -> np.ndarray:
        """复现 SmolVLMVisionEmbeddings 的 2D bucket position id 逻辑。"""
        if patch_attention_mask is None:
            mask = np.ones((bsz, patches_per_side, patches_per_side), dtype=bool)
        else:
            mask = np.asarray(patch_attention_mask, dtype=bool)
            if mask.shape != (bsz, patches_per_side, patches_per_side):
                raise ValueError(f"patch_attention_mask shape must be {(bsz, patches_per_side, patches_per_side)}, got {mask.shape}")

        boundaries = np.arange(1 / patches_per_side, 1.0, 1 / patches_per_side, dtype=np.float32)
        position_ids = np.zeros((bsz, patches_per_side * patches_per_side), dtype=np.int64)

        nb_h = mask[:, :, 0].sum(axis=1).astype(np.float32)
        nb_w = mask[:, 0, :].sum(axis=1).astype(np.float32)
        h_indices = np.arange(patches_per_side, dtype=np.float32)
        w_indices = np.arange(patches_per_side, dtype=np.float32)

        for b in range(bsz):
            step_h = 1.0 / nb_h[b]
            step_w = 1.0 / nb_w[b]
            coords_h = np.clip(h_indices * step_h, None, 1.0 - 1e-6)
            coords_w = np.clip(w_indices * step_w, None, 1.0 - 1e-6)
            bucket_h = np.searchsorted(boundaries, coords_h, side="right")
            bucket_w = np.searchsorted(boundaries, coords_w, side="right")
            pos = (bucket_h[:, None] * patches_per_side + bucket_w[None, :]).reshape(-1)
            flat_mask = mask[b].reshape(-1)
            position_ids[b, flat_mask] = pos[flat_mask]
        return position_ids

    def vit_state_to_float(self, vit_state: np.ndarray, vit_state_scale: float) -> np.ndarray:
        """将 PL int32 ViT state 还原为 PS final post-LN 使用的 float32。"""
        if vit_state_scale is None:
            raise ValueError("vit_state_scale must be explicit for PS floating-point final post-LN")
        state = np.asarray(vit_state).reshape(-1, VIT_C).astype(np.float32, copy=True)
        state *= float(vit_state_scale)
        return state

    def final_post_layernorm(self, vit_hidden: np.ndarray) -> np.ndarray:
        """ViT encoder final post-LayerNorm，输出 float32 [B或1,1024,768] 或 [1024,768]。"""
        x = np.asarray(vit_hidden, dtype=np.float32)
        mean = x.mean(axis=-1, keepdims=True)
        y = x - mean
        var = np.einsum("...c,...c->...", y, y, optimize=True)[..., None] / y.shape[-1]
        y /= np.sqrt(var + VIT_LN_EPS)
        post_ln_w = self._require_vision_asset("post_ln_w")
        post_ln_b = self._require_vision_asset("post_ln_b")
        y *= post_ln_w
        y += post_ln_b
        return y.astype(np.float32, copy=False)

    def connector(self, image_hidden_states: np.ndarray) -> np.ndarray:
        """PS Connector: pixel shuffle + Linear(12288 -> 960)，bias=False。"""
        x = np.asarray(image_hidden_states, dtype=np.float32)
        if x.ndim == 2:
            x = x.reshape(1, VIT_SEQ_LEN, VIT_C)
        if x.shape[1:] != (VIT_SEQ_LEN, VIT_C):
            raise ValueError(f"image_hidden_states shape must be [B,{VIT_SEQ_LEN},{VIT_C}], got {x.shape}")

        x = self.pixel_shuffle(x)
        connector_w = self._require_vision_asset("connector_w")
        y = x @ connector_w.T
        return y.astype(np.float32, copy=False)

    @staticmethod
    def quantize_decoder_state(x: np.ndarray, scale: float = DEFAULT_DECODER_STATE_SCALE) -> np.ndarray:
        """把 PS 端 float embedding 量化为 LLM decoder_state 使用的 int32 定点值。"""
        if scale is None:
            raise ValueError("decoder state scale must be explicit")
        q = np.rint(np.asarray(x, dtype=np.float32) / float(scale))
        return q.clip(np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)

    @staticmethod
    def pixel_shuffle(x: np.ndarray) -> np.ndarray:
        """复现 SmolVLMConnector.pixel_shuffle(scale_factor=4)。"""
        bsz, seq, embed_dim = x.shape
        height = width = int(seq**0.5)
        scale = CONNECTOR_SCALE
        x = x.reshape(bsz, height, width, embed_dim)
        x = x.reshape(bsz, height, width // scale, embed_dim * scale)
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(bsz, width // scale, height // scale, embed_dim * scale * scale)
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(bsz, CONNECTOR_OUT_TOKENS, CONNECTOR_IN_DIM)

    def image_features_from_vit_state(self, vit_state: np.ndarray, vit_state_scale: float) -> np.ndarray:
        """PL ViT int32 state -> float final post-LN -> PS Connector image features。"""
        if self.vision_backend == "torch":
            return self._torch_image_features_from_vit_state(vit_state, vit_state_scale)
        if vit_state_scale is None:
            raise ValueError("vit_state_scale must be explicit for PS numpy vision backend")
        state = np.asarray(vit_state, dtype=np.int32)
        if state.ndim == 1:
            vit_hidden = self.vit_state_to_float(state, vit_state_scale).reshape(1, VIT_SEQ_LEN, VIT_C)
        elif state.ndim == 2:
            if state.shape == (VIT_SEQ_LEN, VIT_C):
                vit_hidden = state.astype(np.float32, copy=True).reshape(1, VIT_SEQ_LEN, VIT_C)
                vit_hidden *= float(vit_state_scale)
            elif state.shape[1] == VIT_C and state.shape[0] % VIT_SEQ_LEN == 0:
                vit_hidden = state.astype(np.float32, copy=True).reshape(-1, VIT_SEQ_LEN, VIT_C)
                vit_hidden *= float(vit_state_scale)
            else:
                raise ValueError(f"vit_state rank-2 shape must be [1024,768] or [N*1024,768], got {state.shape}")
        elif state.ndim == 3 and state.shape[1:] == (VIT_SEQ_LEN, VIT_C):
            vit_hidden = state.astype(np.float32, copy=True)
            vit_hidden *= float(vit_state_scale)
        else:
            raise ValueError(f"vit_state shape must be flat, [1024,768], or [N,1024,768], got {state.shape}")
        vit_ln = self.final_post_layernorm(vit_hidden)
        return self.connector(vit_ln)

    def _torch_image_features_from_vit_state(self, vit_state: np.ndarray, vit_state_scale: float) -> np.ndarray:
        """torch/BF16 版 final post-LN + Connector，用于官方软件路径对齐调试。"""
        import torch
        import torch.nn.functional as F

        self._require_vision_asset("post_ln_w")
        self._require_vision_asset("post_ln_b")
        self._require_vision_asset("connector_w")
        if vit_state_scale is None:
            raise ValueError("vit_state_scale must be explicit for PS torch vision backend")
        raw_state = np.asarray(vit_state, dtype=np.int32)
        if raw_state.ndim == 1:
            state = raw_state.reshape(1, VIT_SEQ_LEN, VIT_C)
        elif raw_state.ndim == 2 and raw_state.shape == (VIT_SEQ_LEN, VIT_C):
            state = raw_state.reshape(1, VIT_SEQ_LEN, VIT_C)
        elif raw_state.ndim == 2 and raw_state.shape[1] == VIT_C and raw_state.shape[0] % VIT_SEQ_LEN == 0:
            state = raw_state.reshape(-1, VIT_SEQ_LEN, VIT_C)
        elif raw_state.ndim == 3 and raw_state.shape[1:] == (VIT_SEQ_LEN, VIT_C):
            state = raw_state
        else:
            raise ValueError(f"vit_state shape must be flat, [1024,768], or [N,1024,768], got {raw_state.shape}")
        x = torch.from_numpy(state.astype(np.int64)).to(dtype=torch.float32)
        x = (x * float(vit_state_scale)).to(dtype=self._torch_dtype())
        weight = self._torch_tensor(self.post_ln_w)
        bias = self._torch_tensor(self.post_ln_b)
        x = F.layer_norm(x, (VIT_C,), weight=weight, bias=bias, eps=VIT_LN_EPS)
        x = self._torch_pixel_shuffle(x)
        connector_w = self._torch_tensor(self.connector_w)
        y = F.linear(x, connector_w, bias=None)
        return y.to(dtype=torch.float32).cpu().numpy().astype(np.float32)

    def merge_embeddings(
        self,
        input_ids: np.ndarray | list[int],
        text_embeds: np.ndarray,
        image_features: np.ndarray,
        image_token_id: int = IMAGE_TOKEN_ID,
    ) -> np.ndarray:
        """把 `<image>` token 对应位置替换成 Connector 输出。"""
        ids = np.asarray(input_ids, dtype=np.int64)
        merged = np.asarray(text_embeds, dtype=np.float32).copy()
        features = np.asarray(image_features, dtype=np.float32).reshape(-1, C)
        if ids.ndim == 1:
            positions = np.flatnonzero(ids == image_token_id)
            if positions.size != features.shape[0]:
                raise ValueError(f"image token count {positions.size} != image feature count {features.shape[0]}")
            merged[positions] = features
            return merged

        if ids.ndim != 2:
            raise ValueError(f"input_ids must be rank 1 or 2, got {ids.shape}")
        if merged.shape[:2] != ids.shape:
            raise ValueError(f"text_embeds leading shape {merged.shape[:2]} does not match input_ids {ids.shape}")

        positions = np.argwhere(ids == image_token_id)
        if positions.shape[0] != features.shape[0]:
            raise ValueError(f"image token count {positions.shape[0]} != image feature count {features.shape[0]}")
        for feature_idx, (batch_idx, token_idx) in enumerate(positions):
            merged[batch_idx, token_idx] = features[feature_idx]
        return merged

    def merge_embeddings_i32(
        self,
        input_ids: np.ndarray | list[int],
        text_embeds_i32: np.ndarray,
        image_features: np.ndarray,
        image_token_id: int = IMAGE_TOKEN_ID,
    ) -> np.ndarray:
        """
        按官方 `inputs_merger` 的 `<image>` token 替换规则生成 LLM 入口 int32 embedding。

        text embedding 已经是 `text_embedding_i16.bin` 查表后的 O_res=12
        定点值；Connector 仍保持浮点实现，只在 merge 前量化一次为同一
        decoder_state 尺度。这样后续 8-token tile 重跑不再重复 float->int round。
        """
        ids = np.asarray(input_ids, dtype=np.int64)
        original_ndim = ids.ndim
        if original_ndim == 1:
            ids_2d = ids.reshape(1, -1)
            text_2d = np.asarray(text_embeds_i32, dtype=np.int32).reshape(1, ids_2d.shape[1], C)
        elif original_ndim == 2:
            ids_2d = ids
            text_2d = np.asarray(text_embeds_i32, dtype=np.int32)
            if text_2d.shape != (ids_2d.shape[0], ids_2d.shape[1], C):
                raise ValueError(f"text_embeds_i32 shape must be {(ids_2d.shape[0], ids_2d.shape[1], C)}, got {text_2d.shape}")
        else:
            raise ValueError(f"input_ids must be rank 1 or 2, got {ids.shape}")

        features = self.quantize_decoder_state(image_features).reshape(-1, CONNECTOR_OUT_TOKENS, C)
        merged = text_2d.copy()
        image_mask = ids_2d == image_token_id
        num_image_tokens = image_mask.sum(axis=1)
        if np.any(num_image_tokens % CONNECTOR_OUT_TOKENS != 0):
            raise ValueError("At least one sample has <image> tokens not divisible by connector output tokens")
        expected_blocks = int(num_image_tokens.sum() // CONNECTOR_OUT_TOKENS)
        if expected_blocks != features.shape[0]:
            raise ValueError(f"image block count {expected_blocks} != image feature blocks {features.shape[0]}")

        blocks_per_sample = num_image_tokens // CONNECTOR_OUT_TOKENS
        offsets = np.pad(np.cumsum(blocks_per_sample), (1, 0), constant_values=0)
        block_offset = offsets[:-1]
        row_cum = image_mask.cumsum(axis=-1)
        chunk_idx = (row_cum - 1) // CONNECTOR_OUT_TOKENS
        local_idx = (row_cum - 1) % CONNECTOR_OUT_TOKENS
        block_idx = block_offset[:, None] + chunk_idx
        merged[image_mask] = features[block_idx[image_mask], local_idx[image_mask], :]

        if original_ndim == 1:
            return merged.reshape(ids_2d.shape[1], C)
        return merged
