"""
SmolVLM2 KV260 PYNQ 常量。

本文件只描述当前复用硬件和 PS 端浮点 baseline 所需的尺寸、寄存器、
RUN_MASK 和文件名。旧 PL Connector/vision input 寄存器不再使用。
"""

from __future__ import annotations

import os


# ======================== 路径 ========================

PYNQ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN_DIR = os.path.join(PYNQ_ROOT, "bin")
MODEL_DIR = os.path.join(PYNQ_ROOT, "model")
OVERLAY_DIR = os.path.join(PYNQ_ROOT, "overlay")
RESULTS_DIR = os.path.join(PYNQ_ROOT, "results")

BITFILE = os.path.join(OVERLAY_DIR, "smolvlm2_kv260.bit")
HWHFILE = os.path.join(OVERLAY_DIR, "smolvlm2_kv260.hwh")
XSAFILE = os.path.join(OVERLAY_DIR, "smolvlm2_kv260.xsa")
AXILITE_BASE = 0xA0000000

# 当前 bitstream 在 PYNQ 近似到 166/214MHz 后 decoder layer golden 会 mismatch；
# 150MHz 是 2026-05-23 板上扫频确认通过的最高稳定档。
DEFAULT_FCLK0_MHZ = 150.0

# SmolVLM processor 默认会按大图尺寸切出多个 512x512 子图，N = crops + global。
# 当前硬件每个子图都要完整跑一次 T=1024 ViT，且 LLM context S=1024 会被
# N*64 个 <image> token 快速占满；PYNQ 侧默认限制 N<=5，对齐软件端
# `max_splits=5` 口径。
DEFAULT_MAX_IMAGE_SPLITS = 5


# ======================== 模型超参数 ========================

VIT_L = 12
VIT_C = 768
VIT_H = 12
VIT_HC = 64
VIT_CM = 3072
VIT_IMG_SIZE = 512
VIT_PATCH_SIZE = 16
VIT_SEQ_LEN = 1024
VIT_LN_EPS = 1e-6

CONNECTOR_SCALE = 4
CONNECTOR_IN_DIM = VIT_C * CONNECTOR_SCALE * CONNECTOR_SCALE
CONNECTOR_OUT_DIM = 960
CONNECTOR_OUT_TOKENS = VIT_SEQ_LEN // (CONNECTOR_SCALE * CONNECTOR_SCALE)

LLAMA_L = 32
C = 960
H = 15
KVH = 5
HC = 64
CM = 2560
VOCAB = 49280
IMAGE_TOKEN_ID = 49190

G = 8
CP = G
TP = 8
T = 8
TT = T // 1
CT = C // CP
# HLS `src/common.h` 固定 `LLAMA_S=1024`。PYNQ 必须按硬件真实 KV cache
# 深度分配 DDR；即使 text-only demo 当前 prompt 较短，也不能用 256 缩小。
S = 1024


# ======================== DDR buffer 尺寸 ========================

DECODER_STATE_ELEMS = T * C
DECODER_STATE_BYTES = DECODER_STATE_ELEMS * 4
VIT_STATE_ELEMS = VIT_SEQ_LEN * VIT_C
VIT_STATE_BYTES = VIT_STATE_ELEMS * 4
CLS_Y_ELEMS = T
CLS_Y_BYTES = CLS_Y_ELEMS * 4

VIT_A_PACKS = VIT_H * VIT_SEQ_LEN * (VIT_HC // CP)
VIT_A_BYTES = VIT_A_PACKS * 16
VIT_XM_PACKS = VIT_SEQ_LEN * (VIT_CM // CP)
VIT_XM_BYTES = VIT_XM_PACKS * 16

# Native GQA: K/V cache stores only KVH heads; QK/RV reuse each KV head for H/KVH query heads.
# K/V cache 都回到量化 q/s 形式：128-bit logical entry，两个 entry 合并到一个
# 256-bit M_AXI word。V 的 future lane 在 DEMUX 里按 padding 语义清零后再量化。
KV_CACHE_HEADS = KVH
K_CACHE_PACK_BITS = 128
V_CACHE_PACK_BITS = 128
K_CACHE_PACKS = LLAMA_L * KV_CACHE_HEADS * S * (HC // CP)
V_CACHE_PACKS = LLAMA_L * KV_CACHE_HEADS * HC * (S // CP)
K_CACHE_BYTES = K_CACHE_PACKS * K_CACHE_PACK_BITS // 8
V_CACHE_BYTES = V_CACHE_PACKS * V_CACHE_PACK_BITS // 8
KV_CACHE_BYTES = K_CACHE_BYTES + V_CACHE_BYTES


# ======================== 权重/参数文件 ========================

PL_BIN_FILES = {
    "decoder_w": "all_decoder_w.bin",
    "cls_w": "all_cls_w.bin",
    "vit_w": "all_vit_w.bin",
    "llm_lnw": "all_llm_lnw_i32.bin",
    "cls_lnw": "cls_lnw_i32.bin",
    "vit_bias": "all_vit_bias_i32.bin",
    "vit_lnw": "all_vit_lnw_i32.bin",
    "vit_lnb": "all_vit_lnb_i64.bin",
}

PS_FLOAT_FILES = {
    "manifest": "ps_float_assets_manifest.json",
    "text_embedding": "text_embedding_fp16.bin",
    "text_embedding_i16": "text_embedding_i16.bin",
    "patch_w": "patch_embed_w_fp32.bin",
    "patch_b": "patch_embed_b_fp32.bin",
    "position_embedding": "position_embedding_fp32.bin",
    "vision_post_ln_w": "vision_post_ln_w_fp32.bin",
    "vision_post_ln_b": "vision_post_ln_b_fp32.bin",
    "connector_w": "connector_w_fp32.bin",
}


# ======================== AXI-Lite 寄存器 ========================

REG_L_BEGIN = 0x0000
REG_L_CLOSE = 0x0010
REG_MEMORY_DECODER_X = 0x0020
REG_MEMORY_DECODER_Y = 0x0030
REG_MEMORY_CLS_Y = 0x0040
REG_MEMORY_DECODER_W_LO = 0x0050
REG_MEMORY_DECODER_W_HI = 0x0060
REG_MEMORY_CLS_W_LO = 0x0070
REG_MEMORY_CLS_W_HI = 0x0080
REG_MEMORY_K_CACHE = 0x0090
REG_POS = 0x00A0
REG_T = 0x00B0
REG_IDLE = 0x00C0
REG_MODE = 0x00D0
REG_PARAM_OP = 0x00E0
REG_STATE_OP = 0x00F0
REG_MEMORY_VIT_BIAS = 0x0100
REG_MEMORY_LLM_LNW = 0x0110
REG_MEMORY_CLS_LNW = 0x0120
REG_MEMORY_VIT_LNW = 0x0130
REG_MEMORY_VIT_LNB = 0x0140
REG_MEMORY_DECODER_STATE = 0x0150
REG_MEMORY_VIT_STATE = 0x0160
REG_MEMORY_VIT_W_LO = 0x0170
REG_MEMORY_VIT_W_HI = 0x0180
REG_RUN_MASK = 0x0190
REG_MEMORY_VIT_A = 0x01A0
REG_MEMORY_VIT_XM = 0x01B0
REG_CACHE_UPDATE_MODE = 0x01C0
REG_VIT_PROBE_FLAGS = 0x0200
REG_VIT_PROBE_LAYER_CYCLES = 0x0210
REG_VIT_PROBE_A_FIRST_W = 0x0220
REG_VIT_PROBE_A_LAST_B = 0x0230
REG_VIT_PROBE_A_FIRST_AR = 0x0240
REG_VIT_PROBE_A_LAST_R = 0x0250
REG_VIT_PROBE_XM_FIRST_W = 0x0260
REG_VIT_PROBE_XM_LAST_B = 0x0270
REG_VIT_PROBE_XM_FIRST_AR = 0x0280
REG_VIT_PROBE_XM_LAST_R = 0x0290
REG_VIT_PROBE_PAIR_BASE = 0x0300
REG_VIT_PROBE_PAIR_STRIDE = 0x0010

VIT_STAGE_PROBE_PAIRS = []

REGISTER_MAP = [
    ("L_BEGIN", REG_L_BEGIN),
    ("L_CLOSE", REG_L_CLOSE),
    ("MEMORY_DECODER_X", REG_MEMORY_DECODER_X),
    ("MEMORY_DECODER_Y", REG_MEMORY_DECODER_Y),
    ("MEMORY_CLS_Y", REG_MEMORY_CLS_Y),
    ("MEMORY_DECODER_W_LO", REG_MEMORY_DECODER_W_LO),
    ("MEMORY_DECODER_W_HI", REG_MEMORY_DECODER_W_HI),
    ("MEMORY_CLS_W_LO", REG_MEMORY_CLS_W_LO),
    ("MEMORY_CLS_W_HI", REG_MEMORY_CLS_W_HI),
    ("MEMORY_K_CACHE", REG_MEMORY_K_CACHE),
    ("POS", REG_POS),
    ("T", REG_T),
    ("IDLE", REG_IDLE),
    ("MODE", REG_MODE),
    ("PARAM_OP", REG_PARAM_OP),
    ("STATE_OP", REG_STATE_OP),
    ("MEMORY_VIT_BIAS", REG_MEMORY_VIT_BIAS),
    ("MEMORY_LLM_LNW", REG_MEMORY_LLM_LNW),
    ("MEMORY_CLS_LNW", REG_MEMORY_CLS_LNW),
    ("MEMORY_VIT_LNW", REG_MEMORY_VIT_LNW),
    ("MEMORY_VIT_LNB", REG_MEMORY_VIT_LNB),
    ("MEMORY_DECODER_STATE", REG_MEMORY_DECODER_STATE),
    ("MEMORY_VIT_STATE", REG_MEMORY_VIT_STATE),
    ("MEMORY_VIT_W_LO", REG_MEMORY_VIT_W_LO),
    ("MEMORY_VIT_W_HI", REG_MEMORY_VIT_W_HI),
    ("RUN_MASK", REG_RUN_MASK),
    ("MEMORY_VIT_A", REG_MEMORY_VIT_A),
    ("MEMORY_VIT_XM", REG_MEMORY_VIT_XM),
    ("CACHE_UPDATE_MODE", REG_CACHE_UPDATE_MODE),
    ("VIT_PROBE_FLAGS", REG_VIT_PROBE_FLAGS),
    ("VIT_PROBE_LAYER_CYCLES", REG_VIT_PROBE_LAYER_CYCLES),
    ("VIT_PROBE_A_FIRST_W", REG_VIT_PROBE_A_FIRST_W),
    ("VIT_PROBE_A_LAST_B", REG_VIT_PROBE_A_LAST_B),
    ("VIT_PROBE_A_FIRST_AR", REG_VIT_PROBE_A_FIRST_AR),
    ("VIT_PROBE_A_LAST_R", REG_VIT_PROBE_A_LAST_R),
    ("VIT_PROBE_XM_FIRST_W", REG_VIT_PROBE_XM_FIRST_W),
    ("VIT_PROBE_XM_LAST_B", REG_VIT_PROBE_XM_LAST_B),
    ("VIT_PROBE_XM_FIRST_AR", REG_VIT_PROBE_XM_FIRST_AR),
    ("VIT_PROBE_XM_LAST_R", REG_VIT_PROBE_XM_LAST_R),
    *(
        (reg_name, REG_VIT_PROBE_PAIR_BASE + idx * REG_VIT_PROBE_PAIR_STRIDE)
        for idx, (_, _, reg_name) in enumerate(VIT_STAGE_PROBE_PAIRS)
    ),
]


# ======================== mode/op/RUN_MASK ========================

MODE_LLM = 0
MODE_VIT = 1

# 旧 raw-V 实验曾使用 decode current-only cache 更新。当前 HLS top 已删除该端口；
# 保留常量仅为兼容旧调试脚本参数，PYNQ driver 不再写该寄存器。
CACHE_UPDATE_FULL_TILE = 0
CACHE_UPDATE_CURRENT_ONLY = 1

AXI_PARAM_BIAS = 0
AXI_PARAM_NORM = 1

AXI_STATE_REPLAY_TOKEN = 0
AXI_STATE_REPLAY_DELTA = 1
AXI_STATE_WRITEBACK = 2
AXI_STATE_WRITE_CLS = 3
AXI_STATE_LAYER = 4

RUN_M_AXI = 1 << 0
RUN_WEIGHT_AXI = 1 << 1
RUN_KV_CACHE = 1 << 2
RUN_STATE_AXI = 1 << 3
RUN_MUX = 1 << 4
RUN_PERMUTE = 1 << 5
RUN_DEMUX = 1 << 6
RUN_ROPE_QK = 1 << 7
RUN_QK_GEMM = 1 << 8
RUN_SOFTMAX = 1 << 9
RUN_RV_GEMM = 1 << 10
RUN_SILU_GELU = 1 << 11
RUN_RESIDUAL = 1 << 12
RUN_RMS_LAYERNORM = 1 << 13
RUN_A_REORDER = 1 << 14
RUN_XM_REORDER = 1 << 15

LLM_LAYER_RUN_MASK = (
    RUN_M_AXI
    | RUN_WEIGHT_AXI
    | RUN_KV_CACHE
    | RUN_STATE_AXI
    | RUN_MUX
    | RUN_PERMUTE
    | RUN_DEMUX
    | RUN_ROPE_QK
    | RUN_QK_GEMM
    | RUN_SOFTMAX
    | RUN_RV_GEMM
    | RUN_SILU_GELU
    | RUN_RESIDUAL
    | RUN_RMS_LAYERNORM
)

LLM_CLS_RUN_MASK = (
    RUN_M_AXI
    | RUN_WEIGHT_AXI
    | RUN_STATE_AXI
    | RUN_MUX
    | RUN_PERMUTE
    | RUN_DEMUX
    | RUN_RESIDUAL
    | RUN_RMS_LAYERNORM
)
LLM_CLS_WRITE_RUN_MASK = RUN_STATE_AXI

VIT_PARAM_NORM_RUN_MASK = RUN_M_AXI
VIT_LAYER_RUN_MASK = (
    RUN_M_AXI
    | RUN_WEIGHT_AXI
    | RUN_STATE_AXI
    | RUN_MUX
    | RUN_PERMUTE
    | RUN_DEMUX
    | RUN_ROPE_QK
    | RUN_QK_GEMM
    | RUN_SOFTMAX
    | RUN_RV_GEMM
    | RUN_A_REORDER
    | RUN_SILU_GELU
    | RUN_XM_REORDER
    | RUN_RESIDUAL
    | RUN_RMS_LAYERNORM
)


# ======================== PS 浮点 baseline 定点边界 ========================

# PL/PS 边界沿用软件整数模型的 residual 主干尺度 O_res=12。
# int32 state 表示 real ~= int * 2^-12；ViT final post-LN 浮点 baseline
# 进入 LayerNorm 前按该 scale 还原，LLM/Vision 写入 PL state 时按同一尺度量化。
FIXED_STATE_FRAC_BITS = 12
FIXED_STATE_SCALE = 2.0 ** (-FIXED_STATE_FRAC_BITS)
DEFAULT_DECODER_STATE_SCALE = FIXED_STATE_SCALE
DEFAULT_VIT_INPUT_SCALE = FIXED_STATE_SCALE
DEFAULT_VIT_STATE_SCALE = FIXED_STATE_SCALE
