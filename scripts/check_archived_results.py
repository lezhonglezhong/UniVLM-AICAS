#!/usr/bin/env python3
"""Validate the public KV260-W5A8 metric record without FPGA hardware."""

from __future__ import annotations

import json
from pathlib import Path


EXPECTED = {
    ("ocrbench_sample30", "correct"): 15,
    ("ocrbench_sample30", "total"): 30,
    ("throughput", "prefill_tok_s"): 21.19127881198675,
    ("throughput", "decode_tok_s"): 14.30557238641584,
    ("energy", "tokens_per_joule"): 1.5715118531127308,
    ("ttft", "slope_ms_per_char"): 1.6610218302815856,
    ("ttft", "intercept_ms"): 13282.58568057369,
    ("timing", "wns_ns"): 0.096,
}


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "results" / "w5a8_verified_metrics.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    for (section, key), expected in EXPECTED.items():
        actual = record["results"][section][key]
        if actual != expected:
            raise SystemExit(f"FAIL: {section}.{key}={actual!r}, expected {expected!r}")

    print("PASS: W5A8 metrics match the public record.")


if __name__ == "__main__":
    main()
