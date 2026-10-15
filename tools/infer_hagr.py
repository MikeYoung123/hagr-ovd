#!/usr/bin/env python3
"""Run one HAGR role with a custom open vocabulary."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLOWorld
from ultralytics.nn.modules.granularity_router import set_granularity_inference_role


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--role", required=True, choices=("category", "attribute", "affordance"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompts", nargs="+")
    group.add_argument("--prompts-file", help="One prompt per line.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--text-batch", type=int, default=64)
    parser.add_argument("--project", default="runs/hagr_predict")
    parser.add_argument("--name", default="predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = args.prompts
    if args.prompts_file:
        prompts = [line.strip() for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError("No non-empty prompts were provided")

    detector = YOLOWorld(args.model)
    world_model = detector.model
    if not set_granularity_inference_role(world_model, args.role):
        raise RuntimeError("The checkpoint does not contain a HAGR router")
    text_model = str(getattr(world_model, "text_model", ""))
    if text_model.split(":", 1)[0].lower() not in {"siglip", "siglip2"}:
        raise RuntimeError(f"Expected a SigLIP/SigLIP2 checkpoint, got {text_model!r}")

    world_model.clip_model = None
    world_model.set_classes(prompts, batch=args.text_batch)
    world_model.names = prompts
    detector.predict(
        source=args.source,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        project=args.project,
        name=args.name,
        save=True,
    )


if __name__ == "__main__":
    main()
