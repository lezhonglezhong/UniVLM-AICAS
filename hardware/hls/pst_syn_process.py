"""
This module is for post HLS-synthesis process, including:
1. collect resource report, and print the table. (print_resource_table)
2. collect IP files. (collect_ip)
3. backup verilog files. (backup_verilog)
4. backup log files. (backup_log)
5. inplace replace the fifo depth. This is used in previous convolutional networks. (inplace_replace)
6. to spinal. This is used in spinal simulation. (to_spinal_one_block, to_spinal_all)
7. launch spinal simulation. (launch_one_spinal_sim, launch_all_spinal_sim)
8. get spinal simulation latencies. (get_latencies)
9. calculate bram cost. (cal_block_bram_cost, cal_total_bram_cost)
"""
import glob
import os
import shutil
import re
from math import ceil
import socket
import threading
from constants import *

# K26/KV260 最终器件 xck26-sfvc784-2LV-c 的可用资源。
# 注意：step4 读取的是 Vitis HLS top_export/vitis_hls.log 口径，BRAM 表示 RAMB18 数；
# Vivado implementation 的 Block RAM Tile 是 36K tile，物理可用量为 144 tile = 288 RAMB18。
KV260_RESOURCE_AVAILABLE = {
    "SLICE": 14640,
    "LUT": 117120,
    "FF": 234240,
    "DSP": 1248,
    "BRAM": 288,
    "URAM": 64,
    "LATCH": 234240,
    "SRL": 57600,
    "CP": 2.5,
}

# -------------------------------------------------------------------------------------
def get_resource_table(instances_root: str, instances_list=[]):
    if not instances_list:
        instances_list = os.listdir(instances_root)

    resource_types = ["SLICE", "LUT", "FF", "DSP", "BRAM", "URAM", "LATCH", "SRL"]
    result_dict = {}
    for instance in instances_list:
        log_path = os.path.join(instances_root, instance, "vitis_hls.log")
        try:
            log_content = open(log_path).read()
        except FileNotFoundError:
            print("FileNotFoundError: " + log_path)
            continue
        instance_dict = {}
        for resource_type in resource_types:
            instance_dict[resource_type] = re.findall(resource_type + r":\s+(\d+)", log_content)

        # a special attribute: CP
        cp_achieved = re.findall(r"CP achieved post-synthesis:\s+(\d+\.\d+)", log_content)
        instance_dict["CP"] = cp_achieved

        # if any not filled, don't add to result_dict
        if any([len(instance_dict[key]) == 0 for key in instance_dict]):
            continue
        result_dict[instance] = instance_dict
    return result_dict


# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
def print_resource_table(instances_root: str):
    resource_types = ["SLICE", "LUT", "FF", "DSP", "BRAM", "URAM", "LATCH", "SRL", "CP"]
    result_dict = get_resource_table(instances_root)
    # create a new entry to store the sum of all instances; for CP, use maxvalue
    result_dict["SUM"] = {resource_type: [0] for resource_type in resource_types}
    for instance in result_dict:
        if instance == "SUM":
            continue
        for resource_type in resource_types:
            if resource_type == "CP":
                result_dict["SUM"][resource_type][0] = max(result_dict["SUM"][resource_type][0], float(result_dict[instance][resource_type][0]))
            else:
                result_dict["SUM"][resource_type][0] += int(result_dict[instance][resource_type][0])
    # convert to str
    for instance in result_dict:
        for resource_type in resource_types:
            result_dict[instance][resource_type][0] = str(result_dict[instance][resource_type][0])
    available_row = {rt: str(KV260_RESOURCE_AVAILABLE[rt]) for rt in resource_types}
    print()
    print(f"{'instance':40s}{''.join([f'{resource_type:10s}' for resource_type in resource_types])}\n")
    for instance in result_dict:
        if instance == "SUM":
            continue
        print(f"{instance:40s}", end="")
        for resource_type in resource_types:
            print(f"{result_dict[instance][resource_type][0]:10s}", end="")
        print()
    print(f"{'SUM':40s}", end="")
    for resource_type in resource_types:
        print(f"{result_dict['SUM'][resource_type][0]:10s}", end="")
    print()
    print(f"{'Available':40s}", end="")
    for resource_type in resource_types:
        print(f"{available_row[resource_type]:10s}", end="")
    print()
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
def collect_ip(instances_root: str, target_dir: str):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for zip_file in glob.glob(os.path.join(instances_root, "**/export_ip/*.zip"), recursive=True):
        print(f"Copying {zip_file} to {target_dir}...")
        shutil.copy(zip_file, target_dir)
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
def backup_verilog(instances_root: str, instances_list=[]):
    if not instances_list:
        instances_list = os.listdir(instances_root)

    for instance in instances_list:
        vlog_dir = os.path.join(instances_root, instance, "work/solution/syn/verilog")
        vlog_files = [f for f in os.listdir(vlog_dir) if f.endswith(".v")]
        data_files = [f for f in os.listdir(vlog_dir) if f.endswith(".dat")]  # ROM

        print(f"Backing up {instance}...")
        print(f"vlog_dir: {vlog_dir}")
        print(f"vlog_files: {vlog_files}")
        print(f"data_files: {data_files}")

        backup_dir = os.path.join(instances_root, instance, "work/solution/syn/verilog_backup")

        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        else:
            print(f"backup_dir {backup_dir} already exists, overwrite")

        for file in vlog_files + data_files:
            shutil.copyfile(os.path.join(vlog_dir, file), os.path.join(backup_dir, file))
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
def backup_log(instances_root: str, instances_list=[], overwrite=False):
    if not instances_list:
        instances_list = os.listdir(instances_root)

    for instance in instances_list:
        log_dir = os.path.join(instances_root, instance)
        log_file = "vitis_hls.log"
        backup_file = "golden.log"

        if not os.path.exists(os.path.join(log_dir, backup_file)) or overwrite:
            shutil.copyfile(os.path.join(log_dir, log_file), os.path.join(log_dir, backup_file))
        else:
            print(f"backup_file {backup_file} already exists, skip backup")
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
def to_spinal(instances_root: str, case_names):
    """copy the verilog file of linear"""
    for case_name in case_names:
        # case_name = "FC_SAXILITE"
        target_file = SPINAL_DIR + f"/src/main/verilog/{case_name}/all.v"
        src_dir = instances_root + f"proj_{case_name}/work/solution/syn/verilog/"
        assert os.path.exists(src_dir), f"src_dir {src_dir} not exists"
        print(f"Copying {src_dir} to {target_file}...")

        # use verilog files to generate spinal files
        vlog_files = [f for f in os.listdir(src_dir) if f.endswith(".v")]
        data_files = [f for f in os.listdir(src_dir) if f.endswith(".dat")]  # ROM

        content = []  # one file
        content.append("/* verilator lint_off PINMISSING */\n")
        content.append("/* verilator lint_off CASEINCOMPLETE */\n")
        content.append("/* verilator lint_off COMBDLY */\n")
        content.append("/* verilator lint_off CASEX */\n")
        content.append("/* verilator lint_off CASEOVERLAP */\n")
        for vlog_file in vlog_files:
            with open(src_dir + vlog_file) as f:
                input_lines = f.readlines()
            for input_line in input_lines:
                if input_line.strip().startswith("#0"):
                    input_line = "//" + input_line
                if "readmemh" in input_line:
                    # input_line = input_line.replace('readmemh("./', 'readmemh("' + os.path.join(SPINAL_DIR, f"src/main/verilog/{case_name}/"))
                    # should replace \ with /, otherwise it will be wrong in windows
                    input_line = input_line.replace('readmemh("./', 'readmemh("' + os.path.join(SPINAL_DIR, f"src/main/verilog/{case_name}/").replace("\\", "/"))
                    print(input_line)
                if "top" in input_line:
                    input_line = input_line.replace("top", case_name)
                if "plusargs" in input_line:
                    input_line = "//" + input_line + f'$display("This is {case_name}.\\n");\n'
                # parameter    C_M_AXI_GMEM_ADDR_WIDTH = 64;
                # use re to find "C_M_AXI_\w+_WIDTH\s*=\s*\d+"
                # replace width with 48

                if re.search(r"C_M_AXI_\w+_WIDTH\s*=\s*\d+", input_line):
                    port_name = re.search(r"C_M_AXI_(\w+)_WIDTH", input_line).group(1)
                    input_line = re.sub(r"C_M_AXI_\w+_ADDR_WIDTH\s*=\s*\d+", f"C_M_AXI_{port_name}_WIDTH = 63", input_line)

                content.append(input_line)
        if not os.path.exists(SPINAL_DIR + f"/src/main/verilog/{case_name}"):
            os.makedirs(SPINAL_DIR + f"/src/main/verilog/{case_name}")
        with open(target_file, "w") as f:
            f.writelines(content)

        # copy all .dat files
        for data_file in data_files:
            shutil.copyfile(src_dir + data_file, SPINAL_DIR + f"/src/main/verilog/{case_name}/" + data_file.replace("top", case_name))


# -------------------------------------------------------------------------------------
