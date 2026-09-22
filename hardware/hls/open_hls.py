from pst_syn_process import *
from pre_syn_process import *

import os
os.environ["CPLUS_INCLUDE_PATH"] = os.getcwd()

case_names = [
    # "RMSNORM"
    # "CORDIC"
    # "QUANTIZER"
    # "GEMM"
    # "ROPE_QK_QUANT"
    # "DEMUX",
    # "MUX"
    # "BUFFER_TESTBENCH"
    # "RV_GEMM"
    # "QK_GEMM"
    # "QUANTIZER"
     "GEMM_PERMUTE"
    # "MUX"
    # "M_AXI"
]

INSTANCE_DIR = os.path.join(ROOT_DIR, "instances_tp8_syn_kv260")


for case_name in case_names:
    workspace_path = os.path.join(INSTANCE_DIR, f"proj_{case_name}", "work")

    # version = "2022.1"
    version = "2024.2"
    vitis_home = os.path.join("/data/home/songqiangxu/Xilinx/", version, "Vitis_HLS", version, "bin")
    # if version=="2020.1":
    #     vitis_home = os.path.join("C:/programs/xilinx", version.replace('.', '_'), "Vitis", version, "bin")


    # use os.system to open HLS gui
    vitis_hls_cmd = os.path.join(vitis_home, "vitis_hls")

    if version == "2024.2":
        os.system(f'{vitis_hls_cmd} -classic -p {workspace_path} ')
    else:
        os.system(f'{vitis_hls_cmd} -p {workspace_path} ')