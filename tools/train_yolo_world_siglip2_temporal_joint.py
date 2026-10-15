#!/usr/bin/env python3
"""Jointly fine-tune HAGR routing and its shared box branch on one temporal neighbor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = (
    str(ROOT) if not os.environ.get("PYTHONPATH") else f"{ROOT}{os.pathsep}{os.environ['PYTHONPATH']}"
)

DEFAULT_BASE = "data/bridge_grounding"
DEFAULT_TRI = "data/bridge_grounding_tri_granular"
DEFAULT_TEMPORAL_DATASET = "data/bridge_grounding_temporal_joint"
DEFAULT_PROPAGATED = "data/bridge_temporal_propagated"
DEFAULT_MODEL = "checkpoints/hagr_ovd_s.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Trained category-rooted HAGR checkpoint.")
    parser.add_argument("--original-train-json", default=f"{DEFAULT_TRI}/bridge_qwen_tri_granular_train.json")
    parser.add_argument("--train-vocab", default=f"{DEFAULT_TRI}/bridge_qwen_tri_granular_train_vocab.json")
    parser.add_argument("--original-summary", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--temporal-train-json",
        default=f"{DEFAULT_TEMPORAL_DATASET}/bridge_qwen_tri_granular_temporal_train.json",
    )
    parser.add_argument("--temporal-summary", default="", help=argparse.SUPPRESS)
    parser.add_argument("--original-image-dir", default=f"{DEFAULT_BASE}/images")
    parser.add_argument("--temporal-image-dir", default=f"{DEFAULT_PROPAGATED}/images/train")
    parser.add_argument("--category-data", default=f"{DEFAULT_BASE}/bridge_qwen_category_det.yaml")
    parser.add_argument("--attribute-data", default=f"{DEFAULT_BASE}/bridge_qwen_attribute_phrase_det.yaml")
    parser.add_argument("--affordance-data", default=f"{DEFAULT_BASE}/bridge_qwen_affordance_phrase_det.yaml")
    parser.add_argument("--selection-role", choices=("category", "attribute", "affordance"), default="attribute")
    parser.add_argument("--device", default="0,1,2,3")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(ROOT / "runs/world_temporal_joint"))
    parser.add_argument("--name", default="siglip2_bridge_hagr_temporal_joint")
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--shared-lr-mult", type=float, default=0.1)
    parser.add_argument(
        "--box-lr-mult",
        type=float,
        default=0.25,
        help="LR ratio for the shared cv2 box/DFL branch; backbone, neck and visual/text projections remain frozen.",
    )
    parser.add_argument("--category-prior", type=float, default=1.0)
    parser.add_argument("--inheritance-max", type=float, default=0.5)
    parser.add_argument("--category-gain", type=float, default=1.0)
    parser.add_argument("--attribute-gain", type=float, default=1.0)
    parser.add_argument("--affordance-gain", type=float, default=1.0)
    parser.add_argument("--attribute-preserve-gain", type=float, default=1.0)
    parser.add_argument("--max-supervision", type=int, default=48)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--lrf", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--close-mosaic", type=int, default=5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--cache", choices=("", "ram", "disk"), default="")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--cos-lr", action="store_true")
    return parser.parse_args()


def require_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def require_dir(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_artifacts(args: argparse.Namespace) -> dict[str, Path]:
    """Validate the published files and episode split without construction logs."""
    from release_data import validate_training_release

    paths = {
        "model": require_file(args.model, "HAGR initialization checkpoint"),
        "original_train_json": require_file(args.original_train_json, "original train JSON"),
        "train_vocab": require_file(args.train_vocab, "train vocabulary"),
        "temporal_train_json": require_file(args.temporal_train_json, "temporal train JSON"),
        "original_image_dir": require_dir(args.original_image_dir, "original image directory"),
        "temporal_image_dir": require_dir(args.temporal_image_dir, "temporal image directory"),
        "selection_data": require_file(getattr(args, f"{args.selection_role}_data"), "selection validation YAML"),
    }
    validation_yamls = [require_file(getattr(args, f"{role}_data"), f"{role} validation YAML")
                        for role in ("category", "attribute", "affordance")]
    validate_training_release(
        paths["original_train_json"], paths["train_vocab"], validation_yamls,
        paths["temporal_train_json"], paths["original_image_dir"], paths["temporal_image_dir"],
    )
    return paths


def write_training_yaml(args: argparse.Namespace, paths: dict[str, Path]) -> Path:
    from ultralytics.utils import YAML

    project = Path(args.project).expanduser().resolve()
    path = project / "_configs" / f"{args.name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(
        str(path),
        {
            "train": {
                "grounding_data": [
                    {
                        "img_path": str(paths["original_image_dir"]),
                        "json_file": str(paths["original_train_json"]),
                    },
                    {
                        "img_path": str(paths["temporal_image_dir"]),
                        "json_file": str(paths["temporal_train_json"]),
                    },
                ]
            },
            "val": {"yolo_data": [str(paths["selection_data"])]},
        },
    )
    return path


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.hidden < 1 or args.max_supervision < 3:
        raise ValueError("epochs/hidden must be positive and max-supervision must be at least 3")
    if not 0.0 < args.box_lr_mult <= 1.0 or not 0.0 < args.shared_lr_mult <= 1.0:
        raise ValueError("box-lr-mult and shared-lr-mult must be in (0, 1]")
    if not 0.0 < args.category_prior <= 1.0 or not 0.0 < args.inheritance_max <= 1.0:
        raise ValueError("category-prior and inheritance-max must be in (0, 1]")
    if min(args.category_gain, args.attribute_gain, args.affordance_gain) <= 0:
        raise ValueError("All three role-gain ratios must be positive")
    if args.attribute_preserve_gain < 0:
        raise ValueError("attribute-preserve-gain must be non-negative")

    paths = validate_artifacts(args)
    data_yaml = write_training_yaml(args, paths)
    project = Path(args.project).expanduser().resolve()
    print(
        "Bridge temporal_joint fine-tuning:\n"
        f"  initialization={paths['model']}\n"
        f"  original_grounding={paths['original_train_json']}\n"
        f"  temporal_grounding={paths['temporal_train_json']}\n"
        "  sampling_ratio=one original frame + one selected temporal frame per qualified video (approximately 1:1)\n"
        "  trainable=complete HAGR router + shared cv2 box/DFL branch\n"
        "  frozen=backbone, neck, SigLIP2, visual/text projections and classification contrastive head\n"
        f"  learning_rates=router {args.lr0:g}; box/DFL {args.lr0 * args.box_lr_mult:g}\n"
        "  loss=three routed partial-label BCE + attribute retention + box + DFL\n"
        f"  best.pt selection={args.selection_role} mAP50-95 on the unchanged validation split\n"
        f"  output={project / args.name}",
        flush=True,
    )

    from ultralytics import YOLOWorld
    from ultralytics.models.yolo.world.train_granularity_router import PrimaryPreservingGranularityTrainer

    model = YOLOWorld(str(paths["model"]))
    model.train(
        data=str(data_yaml),
        trainer=PrimaryPreservingGranularityTrainer,
        pretrained=False,
        device=args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(project),
        name=args.name,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        close_mosaic=args.close_mosaic,
        patience=args.patience,
        save_period=args.save_period,
        fraction=args.fraction,
        cache=args.cache or False,
        val=True,
        amp=True,
        compile=False,
        plots=True,
        exist_ok=args.exist_ok,
        partial_mp_max_supervision=args.max_supervision,
        partial_mp_rank_gain=0.0,
        granularity_router_vocab=str(paths["train_vocab"]),
        granularity_router_hidden=args.hidden,
        granularity_shared_lr_mult=args.shared_lr_mult,
        granularity_box_lr_mult=args.box_lr_mult,
        granularity_train_stage="temporal_joint",
        granularity_selection_role=args.selection_role,
        granularity_category_prior=args.category_prior,
        granularity_inheritance_max=args.inheritance_max,
        granularity_category_gain=args.category_gain,
        granularity_attribute_gain=args.attribute_gain,
        granularity_affordance_gain=args.affordance_gain,
        granularity_attribute_preserve_gain=args.attribute_preserve_gain,
    )


if __name__ == "__main__":
    main()
