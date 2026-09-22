# W5A8 hardware source snapshot

This directory is a source-only extraction from the historical tag
`w5a8-submission-recovered-20260618`. It is provided for architecture review
and for authorized reproduction work; it is not a turnkey build.

| Directory | Contents |
| --- | --- |
| `hls/case/` | HLS compute/memory modules for LLM and ViT execution. |
| `hls/src/` | Shared types, quantization, stream, and hardware helpers. |
| `spinal/` | SpinalHDL Scala sources, tests, SBT metadata, and Vivado export script. |
| `vivado/scripts/` | KV260 build, packaging, implementation, and report scripts. |
| `pynq/core/` | Overlay, DDR-buffer, AXI-Lite, and inference orchestration driver. |

Excluded by design: generated Verilog/ROM files, Vitis/Vivado generated projects,
bitstreams, HWH files, packed weights/binaries, model files, datasets, board images,
and historical logs. A successful hardware build requires the exact controlled W5A8
asset manifest; see `../../docs/run-on-kv260.md` and `../../docs/provenance.md`.

The snapshot normalizes historical CRLF line endings and trailing whitespace for
readability. It otherwise retains the frozen source logic; use the cited tag
for byte-for-byte historical recovery.
