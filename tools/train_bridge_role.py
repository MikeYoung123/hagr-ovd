#!/usr/bin/env python3
"""Fine-tune one released Bridge query role from a CLIP or SigLIP2 checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing aligned detector checkpoint")
    parser.add_argument("--role", choices=("category", "attribute", "affordance"), required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.003)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default=str(ROOT / "runs/bridge_roles"))
    parser.add_argument("--name", default="")
    args = parser.parse_args()
    checkpoint = Path(args.model).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    base = Path(args.data_root).expanduser().resolve() / "bridge_grounding"
    stem = {"category": "category", "attribute": "attribute_phrase", "affordance": "affordance_phrase"}[args.role]
    train = base / f"annotations/bridge_qwen_{stem}_train.json"
    val = base / f"bridge_qwen_{stem}_det.yaml"
    from release_data import verify_files
    verify_files([train, val])
    images = base / "images"
    if not images.is_dir():
        raise FileNotFoundError(images)

    from ultralytics import YOLOWorld
    from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
    from ultralytics.utils import YAML

    model = YOLOWorld(str(checkpoint))
    encoder = getattr(model.model, "text_model", "clip:ViT-B/32")
    name = args.name or f"{encoder.split(':')[0]}_bridge_{args.role}"
    data = {"train": {"grounding_data": [{"img_path": str(images), "json_file": str(train)}]},
            "val": {"yolo_data": [str(val)]}}
    config = Path(args.project).expanduser().resolve() / "_configs" / f"{name}.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(str(config), data)
    print(f"role={args.role}; text_model={encoder}; initialization={checkpoint}")
    model.train(
        data=str(config), trainer=WorldTrainerFromScratch, pretrained=False,
        epochs=args.epochs, batch=args.batch, device=args.device, workers=args.workers,
        imgsz=args.imgsz, optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf,
        seed=args.seed, project=str(Path(args.project).expanduser().resolve()), name=name,
        amp=True, compile=False, val=True,
    )


if __name__ == "__main__":
    main()
