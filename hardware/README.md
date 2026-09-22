# Hardware Source Code

| Directory | Contents |
| --- | --- |
| `hls/case/` | HLS compute and memory modules for ViT and LLM inference. |
| `hls/src/` | Shared types, quantization, stream, and hardware helpers. |
| `spinal/` | SpinalHDL sources, tests, SBT metadata, and Vivado export script. |
| `vivado/scripts/` | KV260 build, packaging, implementation, and report scripts. |
| `pynq/core/` | PYNQ overlay, DDR-buffer, AXI-Lite, and inference orchestration code. |

The hardware flow uses the matching KV260 deployment environment, overlay, model files, and W5A8 packed weights.
