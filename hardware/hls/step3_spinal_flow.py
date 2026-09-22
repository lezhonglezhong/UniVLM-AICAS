from pre_syn_process import *
from pst_syn_process import *
import os

INSTANCE_DIR = os.path.join(ROOT_DIR, "instances_tp8_syn_kv260/")

case_names = [
    # 2026-06-04: LLM-02 native-GQA touches the full LLM QKV/KV stream chain.
    "WEIGHT_AXI",
    "MUX",
    "PERMUTE",
    "DEMUX",
    "ROPE_QK",
    "QK_GEMM",
    "RV_GEMM",
    "KV_CACHE",
]

instances_list = ["proj_" + case_name for case_name in case_names]

backup_verilog  (INSTANCE_DIR, instances_list=instances_list)
backup_log      (INSTANCE_DIR, instances_list=instances_list)

to_spinal(INSTANCE_DIR, case_names=case_names)
