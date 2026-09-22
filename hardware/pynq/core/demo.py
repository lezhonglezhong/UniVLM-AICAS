#!/usr/bin/env python3
"""Current-overlay demo entry for KV260 bring-up."""

from __future__ import annotations

import argparse
import os
import traceback

from local_inference import LocalPynqRunner, default_output, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run current SmolVLM2 PYNQ hardware demo.")
    parser.add_argument("--image", default=None, help="Optional image path for multimodal smoke.")
    parser.add_argument("--prompt", default=None, help="Prompt text.")
    parser.add_argument("--text-only", action="store_true", help="Force text-only mode.")
    parser.add_argument("--throughput", action="store_true", help="Run the same local PYNQ generation path and report timing.")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Maximum number of autoregressive tokens to generate.")
    parser.add_argument("--overlay-reload-interval", type=int, default=0, help="Reload PL overlay every N LLM layers; 0 disables.")
    parser.add_argument("--fclk0-mhz", type=float, default=150.0, help="PL fclk0 frequency in MHz.")
    parser.add_argument(
        "--max-image-splits",
        type=int,
        default=5,
        help="Maximum processor subimages per source image, including global image. Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--ps-vision-backend",
        choices=["numpy", "torch"],
        default="numpy",
        help="PS vision backend. torch is a slow BF16 debug path aligned with the software reference.",
    )
    parser.add_argument("--output", default=None, help="Result JSON path. Defaults under results/.")
    parser.add_argument("--dry-run", action="store_true", help="Do not load PYNQ overlay; write a local dry-run JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = args.prompt or ("Describe this image." if args.image and not args.text_only else "What is the capital of France?")
    output = args.output or default_output("demo")
    image_path = None if args.text_only else args.image
    enable_vit = image_path is not None

    try:
        runner = LocalPynqRunner(
            dry_run=args.dry_run,
            enable_vit=enable_vit,
            load_ps_vision=enable_vit,
            load_text_fp16=enable_vit,
            overlay_reload_interval=args.overlay_reload_interval or None,
            ps_vision_backend=args.ps_vision_backend,
            fclk0_mhz=args.fclk0_mhz,
            max_image_splits=args.max_image_splits or None,
            preload_processor=enable_vit,
        )
        result = runner.generate_once(prompt=prompt, image_path=image_path, max_new_tokens=args.max_new_tokens)
        payload = result.to_dict()
        payload["script"] = "core/demo.py"
        payload["throughput_mode"] = bool(args.throughput)
        payload["output_path"] = os.path.abspath(output)
        write_json(output, payload)
    except Exception as exc:
        payload = {
            "status": "error",
            "mode": "text" if args.text_only or not args.image else "image",
            "implementation": "pynq_text_prefill_decode",
            "prompt": prompt,
            "image_path": None if args.text_only else args.image,
            "script": "core/demo.py",
            "throughput_mode": bool(args.throughput),
            "output_path": os.path.abspath(output),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output, payload)
        print("status: error")
        print(f"result_json: {output}")
        print(f"error: {exc}")
        raise SystemExit(1) from exc

    print(f"status: {result.status}")
    print(f"mode: {result.mode}")
    print(f"output_text: {result.output_text}")
    print(f"result_json: {output}")
    if result.error:
        print(f"error: {result.error}")


if __name__ == "__main__":
    main()
