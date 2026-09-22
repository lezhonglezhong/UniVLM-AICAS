# Public code boundary

`demo.py` is a compact PYNQ demo entry copied from the archived W5A8 source tag. It illustrates the public host-side command-line contract: choose text-only or image mode, set the PL frequency/image-split cap, invoke `LocalPynqRunner`, and write a structured JSON result.

It is intentionally not runnable in isolation. `LocalPynqRunner` and its dependencies require the controlled deployment environment, exact overlay/weight pairing, and board assets excluded from this repository. The code is included for architecture review and for authorized users maintaining the original deployment environment; it is not a claim that a fresh clone can execute the model.
