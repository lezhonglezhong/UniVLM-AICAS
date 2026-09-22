# Provenance: archived KV260-W5A8 record

## What this repository represents

This repository is a new public-facing presentation archive, created from the historical UniVLM source tag `w5a8-submission-recovered-20260618`. It is not a fork or a branch of the active research repository. The archived W5A8 record must never be presented as an outcome of later W4A5 or DAC work.

## Evidence chain

| Item | Verified historical source |
| --- | --- |
| Design source | commit `3c897b1`, tag `backup-20260606-dynamic-fclk-runtime` |
| Recovery anchor | annotated tag `w5a8-submission-recovered-20260618` (recovery package commit `eac38c8`) |
| Overlay | rebuilt on the implementation host with Vivado 2024.2, `RUN_SPINAL=1`, `PL_CLK_PERIOD_NS=4.000`; bit MD5 prefix `d96ccd77`, HWH MD5 prefix `89b24f21` |
| Weights | original best-data W5A8 set; the exact file set was recorded in `MANIFEST_md5.txt` inside the controlled recovery package |
| Board proof | recovery report dated 2026-06-18 and its archived result JSON / evaluation commands |

The recovery was required because an incompatible `weight_fifo_merged` overlay (MD5 prefix `8a0bf06c`) had overwritten the accepted overlay. Its weight layout did not match the retained W5A8 files and produced garbage output. Rebuilding the accepted design and pairing it with the original W5A8 weights restored the results below.

## Public metric record

The values in `results/w5a8_verified_metrics.json` are the only values used for the repository highlights.

| Metric | Value | Conditions |
| --- | ---: | --- |
| OCRBench | 15/30 | sample-30; 250 MHz; `max_image_splits=5`; `max_new_tokens=100` |
| Prefill | 21.23 tok/s | dynamic prepare/prefill 300 MHz and decode 250 MHz; 376 prompt tokens; 5 subimages; 8 completion tokens |
| Decode | 14.28 tok/s | same throughput run, decode phase at 250 MHz |
| Energy efficiency | 1.4954 tok/J | 250 MHz; 643 non-EOS completion tokens; 100 Hz PMBus sampling; 5.08 W average; 429.97 J total |
| TTFT fit | 1.647 ms/char + 14952 ms | 250 MHz; two images and three prompt lengths; warmup + two repeats; median per case |
| Post-route timing | WNS +0.096 ns | 250 MHz target |

These are historical reproduction results, not claims of performance on another board, input mix, model revision, W4A5 configuration, or current source tree.

## Excluded assets

No model weights, quantization scales, packed binaries, overlay/bitstream, board image, test dataset, full result logs, or original recovery tarball is copied here. They are intentionally excluded both because of size and because deployment correctness requires an exact overlay-to-weight manifest pairing.
