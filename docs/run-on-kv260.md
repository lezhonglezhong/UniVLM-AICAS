# KV260 deployment reference

This page is a deployment reference for authorized holders of the original AICAS assets. It cannot turn this public archive into a standalone FPGA release: required W5A8 weights, packed binaries, overlay files, PYNQ runtime dependencies, and board-specific paths are intentionally absent.

## Required controlled assets

- KV260 board with the validated PYNQ/XRT environment.
- Overlay rebuilt from design source `3c897b1`, with the accepted bit/HWH hashes.
- Exact W5A8 weight and packed-binary set verified against its controlled manifest.
- A compatible `local_inference` runtime and its PS-side processor/model dependencies.
- Licensed model and benchmark data obtained from their respective sources.

## Historical command shape

On the board, the recovery record used a PYNQ virtual-environment Python interpreter and exported `XILINX_XRT=/usr`. The public `code/demo.py` is the historical demo entry shape:

```bash
export XILINX_XRT=/usr
python core/demo.py --image <image> --max-new-tokens 8 --fclk0-mhz 250
```

The throughput, energy, and TTFT commands are intentionally documented as methodology in this archive rather than shipped as a deployment bundle. Before any board execution, verify the accepted overlay/weight manifest pair; do not substitute a binary produced by another quantization or memory-layout experiment.
