#!/usr/bin/env python3
"""Validate the small, public KV260-W5A8 metric record without FPGA hardware."""

from __future__ import annotations

import json
from pathlib import Path


EXPECTED = {
    ("ocrbench_sample30", "correct"): 15,
    ("ocrbench_sample30", "total"): 30,
    ("throughput", "prefill_tok_s"): 21.23,
    ("throughput", "decode_tok_s"): 14.28,
    ("energy", "tokens_per_joule"): 1.4954,
    ("ttft", "slope_ms_per_char"): 1.647,
    ("ttft", "intercept_ms"): 14952,
    ("timing", "wns_ns"): 0.096,
}


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "results" / "w5a8_verified_metrics.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["historical_source_tag"] != "w5a8-submission-recovered-20260618":
        raise SystemExit("FAIL: unexpected provenance tag")

    for (section, key), expected in EXPECTED.items():
        actual = record["results"][section][key]
        if actual != expected:
            raise SystemExit(f"FAIL: {section}.{key}={actual!r}, expected {expected!r}")

    print("PASS: archived W5A8 metrics match the public record.")


if __name__ == "__main__":
    main()
