#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

vivado_version="${VIVADO_VERSION:-2024.2}"
vivado_root="${VIVADO_ROOT:-/home/cjn/xilinx_vitis2024/Vivado/$vivado_version}"
vivado_bin="${VIVADO_BIN:-$vivado_root/bin/vivado}"

if [[ ! -x "$vivado_bin" ]]; then
  echo "Vivado executable was not found: $vivado_bin" >&2
  echo "Set VIVADO_ROOT=/path/to/Vivado/$vivado_version or VIVADO_BIN=/path/to/vivado." >&2
  exit 127
fi

export XILINX_VIVADO="$vivado_root"
export PATH="$vivado_root/bin:$PATH"

run_stamp="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
run_spinal="${RUN_SPINAL:-1}"
python_bin="${PYTHON:-python}"
pl_clk_period="${PL_CLK_PERIOD_NS:-4.687}"
pl_clk_freq="${PL_CLK_FREQ_MHZ:-}"

mkdir -p vivado/logs vivado/reports vivado/bitstream vivado/work

package_log="vivado/logs/package_accelerator_ip_${run_stamp}.log"
package_jou="vivado/logs/package_accelerator_ip_${run_stamp}.jou"
build_log="vivado/logs/build_kv260_bitstream_${run_stamp}.log"
build_jou="vivado/logs/build_kv260_bitstream_${run_stamp}.jou"

echo "VLM Vivado implementation run"
echo "  repo_root: $repo_root"
echo "  vivado:    $vivado_bin"
echo "  stamp:     $run_stamp"
echo "  pl period: $pl_clk_period ns"
if [[ -n "$pl_clk_freq" ]]; then
  echo "  pl freq:   $pl_clk_freq MHz"
fi

if [[ "$run_spinal" == "1" ]]; then
  echo "Refreshing Spinal ACCELERATOR RTL before IP packaging"
  (
    cd SPINAL
    ./tools/sbt "Test/runMain generate_accelerator"
    "$python_bin" to_vivado.py
  )
else
  echo "Skipping Spinal refresh because RUN_SPINAL=$run_spinal"
fi

echo "Packaging smolvlm2_accelerator IP"
"$vivado_bin" \
  -mode batch \
  -source vivado/scripts/package_accelerator_ip.tcl \
  -journal "$package_jou" \
  -log "$package_log"

echo "Running KV260 implementation and bitstream flow"
build_status=0
"$vivado_bin" \
  -mode batch \
  -source vivado/scripts/build_kv260_bitstream.tcl \
  -journal "$build_jou" \
  -log "$build_log" || build_status=$?

echo "Vivado logs:"
echo "  package: $package_log"
echo "  build:   $build_log"
echo "Vivado reports:"
echo "  vivado/reports/kv260_synth_util.rpt"
echo "  vivado/reports/kv260_impl_util.rpt"
echo "  vivado/reports/kv260_impl_timing.rpt"
echo "  vivado/reports/kv260_route_status.rpt"
echo "  vivado/reports/kv260_clocks.rpt"
echo "  vivado/reports/qk01_design_analysis.rpt"
echo "  vivado/reports/qk01_top200_paths.rpt"
echo "  vivado/reports/qk01_timing_summary_unique.rpt"

if [[ -f "$build_log" ]]; then
  echo "Key build log lines:"
  grep -E "VLM_PL_CLK|write_bitstream completed|route_design failed|Design is not legally routed|signals failed to route|node overlaps|WNS=|TNS=|ERROR:|CRITICAL WARNING:" "$build_log" || true
fi

exit "$build_status"
