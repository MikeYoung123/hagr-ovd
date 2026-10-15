#!/usr/bin/env python3
"""Train primary-preserving hierarchical granularity routing on audited Bridge labels."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = (
    str(ROOT) if not os.environ.get("PYTHONPATH") else f"{ROOT}{os.pathsep}{os.environ['PYTHONPATH']}"
)

from release_data import validate_training_release


DEFAULT_MODEL = "runs/bridge_roles/siglip_bridge_attribute/weights/best.pt"
DEFAULT_TRI_ROOT = "data/bridge_grounding_tri_granular"
DEFAULT_BASE_ROOT = "data/bridge_grounding"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-json", default=f"{DEFAULT_TRI_ROOT}/bridge_qwen_tri_granular_train.json")
    parser.add_argument("--train-vocab", default=f"{DEFAULT_TRI_ROOT}/bridge_qwen_tri_granular_train_vocab.json")
    parser.add_argument("--summary", default="", help=argparse.SUPPRESS)
    parser.add_argument("--image-dir", default=f"{DEFAULT_BASE_ROOT}/images")
    parser.add_argument("--category-data", default=f"{DEFAULT_BASE_ROOT}/bridge_qwen_category_det.yaml")
    parser.add_argument("--attribute-data", default=f"{DEFAULT_BASE_ROOT}/bridge_qwen_attribute_phrase_det.yaml")
    parser.add_argument("--affordance-data", default=f"{DEFAULT_BASE_ROOT}/bridge_qwen_affordance_phrase_det.yaml")
    parser.add_argument("--device", default="0,1,2,3")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(ROOT / "runs/world_granularity_router"))
    parser.add_argument("--name", default="siglip2_bridge_hagr_stable")
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--shared-lr-mult", type=float, default=0.1)
    parser.add_argument(
        "--train-stage",
        choices=("gate", "gate_child", "full"),
        default="full",
        help="gate: direct inheritance only; gate_child: gates plus child adapters; full: all router parameters.",
    )
    parser.add_argument(
        "--selection-role",
        choices=("auto", "category", "attribute", "affordance"),
        default="auto",
        help="Validation role used for best.pt; auto uses affordance for child-only stages and category for full.",
    )
    parser.add_argument(
        "--category-prior",
        type=float,
        default=1.0,
        help="Scale of detached category context and direct inheritance; set 0 for the exact flat ablation.",
    )
    parser.add_argument(
        "--inheritance-max",
        type=float,
        default=0.5,
        help="Maximum absolute per-channel coefficient for direct category residual inheritance.",
    )
    parser.add_argument(
        "--stop-gradient",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Block child-role gradients from updating the category root (use --no-stop-gradient for ablation).",
    )
    parser.add_argument("--category-gain", type=float, default=1.0)
    parser.add_argument("--attribute-gain", type=float, default=1.0)
    parser.add_argument("--affordance-gain", type=float, default=1.0)
    parser.add_argument("--attribute-preserve-gain", type=float, default=1.0)
    parser.add_argument("--max-supervision", type=int, default=48)
    parser.add_argument("--lr0", type=float, default=0.0003)
    parser.add_argument("--lrf", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--cache", choices=("", "ram", "disk"), default="")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--cos-lr", action="store_true")
    return parser.parse_args()


def require_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def validate_artifacts(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path, str]:
    """Validate the public manifest, fixed split, and reconstructed training images."""
    model = require_file(args.model, "initialization checkpoint")
    train_json = require_file(args.train_json, "tri-granular train JSON")
    train_vocab = require_file(args.train_vocab, "exact train vocabulary")
    # Public release identity is certified by its SHA-256 manifest.
    selection_role = (
        "affordance" if args.train_stage in {"gate", "gate_child"} else "category"
    ) if args.selection_role == "auto" else args.selection_role
    selection_data = require_file(getattr(args, f"{selection_role}_data"), f"{selection_role} validation YAML")
    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing training image directory: {image_dir}")
    validation_yamls = [require_file(getattr(args, f"{role}_data"), f"{role} validation YAML")
                        for role in ("category", "attribute", "affordance")]
    summary_path = validate_training_release(train_json, train_vocab, validation_yamls,
                                             original_images=image_dir)
    return model, train_json, train_vocab, summary_path, image_dir, selection_data, selection_role


def write_training_yaml(
    project: Path,
    name: str,
    image_dir: Path,
    train_json: Path,
    selection_data: Path,
    selection_role: str,
) -> Path:
    """Write a deterministic stage-aware validation YAML outside the run directory."""
    from ultralytics.utils import YAML

    path = project / "_configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(
        str(path),
        {
            "train": {"grounding_data": [{"img_path": str(image_dir), "json_file": str(train_json)}]},
            # Gate-only stages cannot change category scores, so affordance
            # selects child checkpoints. Full training uses category by default.
            # Every final checkpoint is still evaluated on all three roles.
            "val": {"yolo_data": [str(selection_data)]},
        },
    )
    return path


def main() -> None:
    args = parse_args()
    if args.hidden < 1 or args.max_supervision < 3:
        raise ValueError("hidden must be positive and max-supervision must be at least 3")
    if args.category_gain <= 0 or args.attribute_gain <= 0 or args.affordance_gain <= 0:
        raise ValueError("category-gain, attribute-gain and affordance-gain must all be positive")
    if args.attribute_preserve_gain < 0:
        raise ValueError("attribute-preserve-gain must be non-negative")
    if not 0.0 < args.shared_lr_mult <= 1.0:
        raise ValueError("shared-lr-mult must be in (0, 1]")
    if not 0.0 <= args.category_prior <= 1.0:
        raise ValueError("category-prior must be in [0, 1]")
    if not 0.0 < args.inheritance_max <= 1.0:
        raise ValueError("inheritance-max must be in (0, 1]")
    if args.train_stage in {"gate", "gate_child"} and args.category_prior <= 0.0:
        raise ValueError(f"train-stage={args.train_stage!r} requires category-prior > 0")
    model_path, train_json, train_vocab, summary, image_dir, selection_data, selection_role = validate_artifacts(args)
    project = Path(args.project).expanduser().resolve()
    data_yaml = write_training_yaml(
        project, args.name, image_dir, train_json, selection_data, selection_role
    )
    print(
        "primary-preserving hierarchical granularity routing:\n"
        f"  model={model_path}\n"
        f"  release_manifest={summary}\n"
        f"  train_json={train_json}\n"
        f"  train_vocab={train_vocab}\n"
        f"  checkpoint_selection={selection_role} AP on {selection_data}\n"
        f"  initialization=zero-residual identity of the frozen base checkpoint\n"
        f"  train_stage={args.train_stage}\n"
        f"  hierarchy=category -> (attribute, affordance), prior={args.category_prior}, "
        f"direct_inheritance_max={args.inheritance_max}, stop_gradient={args.stop_gradient}\n"
        f"  shared_semantic_residual=True, shared_lr_mult={args.shared_lr_mult}\n"
        f"  role_gain_ratios={args.category_gain},{args.attribute_gain},{args.affordance_gain}; normalized=True\n"
        f"  attribute_preserve_gain={args.attribute_preserve_gain}\n"
        f"  output={project / args.name}",
        flush=True,
    )

    from ultralytics import YOLOWorld
    from ultralytics.models.yolo.world.train_granularity_router import PrimaryPreservingGranularityTrainer

    model = YOLOWorld(str(model_path))
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
        granularity_router_vocab=str(train_vocab),
        granularity_router_hidden=args.hidden,
        granularity_shared_lr_mult=args.shared_lr_mult,
        granularity_train_stage=args.train_stage,
        granularity_selection_role=selection_role,
        granularity_category_prior=args.category_prior,
        granularity_inheritance_max=args.inheritance_max,
        granularity_stop_gradient=args.stop_gradient,
        granularity_category_gain=args.category_gain,
        granularity_attribute_gain=args.attribute_gain,
        granularity_affordance_gain=args.affordance_gain,
        granularity_attribute_preserve_gain=args.attribute_preserve_gain,
    )


if __name__ == "__main__":
    main()
