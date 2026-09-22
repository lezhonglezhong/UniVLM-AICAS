from pst_syn_process import *
from pre_syn_process import *

import os
# use os to add include path for C++ compiler
os.environ["CPLUS_INCLUDE_PATH"] = os.getcwd()
INSTANCE_DIR = os.path.join(ROOT_DIR, "instances_tp8_syn_kv260")

# compile all
configs = [
    # 2026-06-03：DEC-02 burst/outstanding tuning，验证 WEIGHT_AXI II/资源。
    ("WEIGHT_AXI",       "flp",      "2023.2"),
]
case_names, pipeline_styles, versions = zip(*configs)
create_subprojects(INSTANCE_DIR, case_names=case_names, overwrite=True)
#create_tcls       (INSTANCE_DIR, case_names=case_names, do_csim=False, do_csynth=True, do_cosim=True, do_syn=True, do_impl=False, pipeline_styles=pipeline_styles)
create_tcls       (INSTANCE_DIR, case_names=case_names, do_csim=False, do_csynth=True, do_cosim=False, do_syn=True, do_impl=False, pipeline_styles=pipeline_styles)
# launch the tcl files
run_instances(INSTANCE_DIR, case_names=case_names, versions=versions, parallel=True)
