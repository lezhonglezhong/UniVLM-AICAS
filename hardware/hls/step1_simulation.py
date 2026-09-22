from pst_syn_process import *
from pre_syn_process import *

import os
# use os to add include path for C++ compiler
os.environ["CPLUS_INCLUDE_PATH"] = os.getcwd()
INSTANCE_DIR = os.path.join(ROOT_DIR, "instances_tp8_sim")

# compile all
case_names = [
      # 2026-06-03：DEC-02 burst/outstanding tuning，验证 WEIGHT_AXI 权重流数值不变。
      "WEIGHT_AXI",
    # LLM 模块（参考顺序，与 ViT 顺序一一对应）
    # "LLM_PERMUTE",
    # "LLM_DEMUX",
    # "LLM_ROPE_QK",   # ViT 无对应（无 RoPE）
    # "LLM_QK_GEMM",
    # "LLM_SOFTMAX",
    # "LLM_RV_GEMM",
    # "LLM_SILU_EM",   # ViT 对应 ViT_GELU
    # "LLM_RESIDUAL",
    # "LLM_RMSNORM",   # ViT 对应 ViT_LAYERNORM
    # "LLM_MUX",
    # ViT 模块（按 LLM 顺序对应；链式依赖：每个模块 condense 输出供后续读取）
    # 注：运行前需先执行 txt_to_bin.py ATTENTION 生成 MHA_R.bin / MHA_V_SPLIT_HEADS.bin
    # "ViT_PERMUTE",     # 1. 投影 GEMM → CONDENSED_GEMM_Y.bin（从 binaries 直接读取激活）
    # "ViT_DEMUX",       # 2. 路由 GEMM 输出 → CONDENSED_DEMUX_QK/V/FC1/OFC2.bin
    # "ViT_QK_GEMM",     # 3. QK^T 得分 → CONDENSED_QK_GEMM_R.bin （已通过）
    # "ViT_SOFTMAX",     # 4. Softmax → CONDENSED_SOFTMAX_R_Q/S.bin （已通过）
    # "ViT_RV_GEMM",     # 5. RV 乘法 → CONDENSED_RV_GEMM_A_Q/S.bin （已通过）
    # "ViT_GELU",        # 6. GELU 激活 → CONDENSED_GELU_XM_Q/S.bin （已通过）
    # "ViT_RESIDUAL",    # 7. 残差更新 → CONDENSED_RESIDUAL_O.bin （已通过）
    # "ViT_LAYERNORM",   # 8. LayerNorm + Quant → CONDENSED_XLN_Q/S.bin
    # "MUX",         # 9. 汇聚 XLN/A/XM → CONDENSED_GEMM_X_Q/S.bin（链式闭环）
]

create_subprojects  (INSTANCE_DIR, case_names=case_names, overwrite=True)
create_tcls         (INSTANCE_DIR, case_names=case_names, do_csim=True, do_csynth=False, do_cosim=False, do_impl=False, pipeline_styles="flp")

# launch the tcl files
# run_instances(INSTANCE_DIR, case_names=case_names, versions="2024.2", parallel=False)
run_instances(INSTANCE_DIR, case_names=case_names, versions="2023.2", parallel=False)
