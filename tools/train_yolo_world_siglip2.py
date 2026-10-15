#!/usr/bin/env python3
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Train YOLO-Worldv2 with the SigLIP2 text encoder on mixed OVD data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = (
    str(ROOT)
    if not os.environ.get("PYTHONPATH")
    else f"{ROOT}{os.pathsep}{os.environ['PYTHONPATH']}"
)

from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default="ultralytics/cfg/models/v8/yolov8s-worldv2.yaml",
        help=(
            "YOLO-World model yaml. Scaled names such as yolov8s-worldv2.yaml are virtual Ultralytics names: "
            "they load yolov8-worldv2.yaml and set scale=s."
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ultralytics/cfg/datasets/yolo_world_siglip2.yaml",
        help="Mixed YOLO-World data yaml.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="Optional YOLO-World pretrained .pt to partially load into the SigLIP2 model.",
    )
    parser.add_argument("--device", type=str, default="0,1,2,3", help="GPU ids for DDP training.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=64, help="Global batch size. 64 is 16/GPU on 4 GPUs.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers per DDP process.")
    parser.add_argument("--project", type=str, default="runs/world", help="Output project directory.")
    parser.add_argument("--name", type=str, default="yolov8s_worldv2_siglip2", help="Run name.")
    parser.add_argument("--resume", action="store_true", help="Resume the latest run if supported by Ultralytics.")
    parser.add_argument("--cache", type=str, default="", choices=["", "ram", "disk"], help="Image cache mode.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction for smoke tests.")
    parser.add_argument("--no-val", action="store_true", help="Disable validation during training.")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer, e.g. SGD or AdamW. Avoid auto here.")
    parser.add_argument("--lr0", type=float, default=0.003, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR fraction, final_lr=lr0*lrf.")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine LR schedule.")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay.")
    return parser.parse_args()


def main() -> None:
    """Launch training."""
    args = parse_args()
    model = YOLOWorld(args.model)
    model.train(
        data=args.data,
        trainer=WorldTrainerFromScratch,
        pretrained=args.pretrained or False,
        device=args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(Path(args.project).expanduser().resolve()),
        name=args.name,
        resume=args.resume,
        cache=args.cache or False,
        fraction=args.fraction,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        amp=True,
        compile=False,
        val=not args.no_val,
        save_period=5,
        close_mosaic=10,
        plots=True,
    )


if __name__ == "__main__":
    main()
