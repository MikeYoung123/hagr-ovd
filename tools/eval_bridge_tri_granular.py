#!/usr/bin/env python3
"""Evaluate one unified YOLO-World checkpoint on all three immutable Bridge vocabularies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) if not os.environ.get("PYTHONPATH") else f"{ROOT}{os.pathsep}{os.environ['PYTHONPATH']}"

import torch  # noqa: E402
from ultralytics import YOLOWorld  # noqa: E402
from ultralytics.nn.text_model import SigLIP  # noqa: E402
from ultralytics.nn.modules.granularity_router import set_granularity_inference_role  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402


DEFAULT_SUMMARY = "data/bridge_grounding_tri_granular/summary_tri_granular.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--train-vocab", default="", help="Exact train vocab JSON; defaults to summary files.train_vocab.")
    parser.add_argument("--base-dataset", default="data/bridge_grounding", help="Fallback root containing three val YAMLs.")
    parser.add_argument(
        "--tri-dataset", default="data/bridge_grounding_tri_granular", help="Fallback root containing train vocabulary."
    )
    parser.add_argument("--category-data", default="")
    parser.add_argument("--attribute-data", default="")
    parser.add_argument("--affordance-data", default="")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--text-batch", type=int, default=64)
    parser.add_argument("--output", default="", help="Defaults beside the model run directory.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_fingerprint(features: torch.Tensor) -> str:
    payload = features.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def load_names(path: Path) -> list[str]:
    payload = YAML.load(str(path))
    names = payload.get("names")
    if isinstance(names, dict):
        return [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
    if isinstance(names, list):
        return [str(value) for value in names]
    raise ValueError(f"No ordered names vocabulary in {path}")


def load_summary(args: argparse.Namespace) -> dict:
    summary_path = Path(args.summary).expanduser().resolve() if args.summary else None
    if not summary_path or not summary_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("kind") != "bridge_tri_granular":
        raise ValueError(f"Unsupported summary: {summary_path}")
    return summary


def resolve_validation_paths(args: argparse.Namespace, summary: dict) -> dict[str, Path]:
    paths = {}
    if summary:
        paths.update({key: Path(value) for key, value in summary.get("validation_yamls", {}).items()})
    for role, value in (
        ("category", args.category_data),
        ("attribute", args.attribute_data),
        ("affordance", args.affordance_data),
    ):
        if value:
            paths[role] = Path(value)
    base_dataset = Path(args.base_dataset).expanduser()
    fallback = {
        "category": base_dataset / "bridge_qwen_category_det.yaml",
        "attribute": base_dataset / "bridge_qwen_attribute_phrase_det.yaml",
        "affordance": base_dataset / "bridge_qwen_affordance_phrase_det.yaml",
    }
    for role, path in fallback.items():
        if role not in paths and path.is_file():
            paths[role] = path
    missing = set(("category", "attribute", "affordance")) - set(paths)
    if missing:
        attempted = {role: str(fallback[role].resolve()) for role in sorted(missing)}
        raise ValueError(
            f"Missing validation YAMLs for roles: {sorted(missing)}. "
            f"Summary={args.summary!r}; fallback paths={attempted}. "
            "Pass --base-dataset or the three --*-data arguments."
        )
    result = {role: path.expanduser().resolve() for role, path in paths.items()}
    for role, path in result.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role} validation YAML: {path}")
    return result


def load_train_vocabulary(args: argparse.Namespace, summary: dict) -> tuple[dict[str, set[str]], Path]:
    raw_path = (
        args.train_vocab
        or summary.get("files", {}).get("train_vocab", "")
        or str(Path(args.tri_dataset).expanduser() / "bridge_qwen_tri_granular_train_vocab.json")
    )
    if not raw_path:
        raise ValueError("Missing train vocabulary. Rebuild the tri-granular dataset or pass --train-vocab.")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing train vocabulary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    roles = {role: {str(value) for value in payload.get(role, [])} for role in ("category", "attribute", "affordance")}
    if any(not values for values in roles.values()):
        raise ValueError(f"Incomplete train vocabulary: {path}")
    return roles, path


def seen_unseen_metrics(metrics, names: list[str], train_names: set[str]) -> dict[str, dict[str, float | int]]:
    """Aggregate class AP over exact train-vocabulary seen/unseen partitions."""
    box = metrics.box
    indices = np.asarray(box.ap_class_index, dtype=int)
    all_ap = np.asarray(box.all_ap, dtype=float)
    precision = np.asarray(box.p, dtype=float)
    recall = np.asarray(box.r, dtype=float)
    if len(indices) != len(all_ap):
        raise RuntimeError("Per-class AP indices and rows differ")
    result = {}
    for group, want_seen in (("seen", True), ("unseen", False)):
        mask = np.asarray([(names[index] in train_names) == want_seen for index in indices], dtype=bool)
        group_ap = all_ap[mask]
        result[group] = {
            "classes_in_val_vocabulary": int(sum((name in train_names) == want_seen for name in names)),
            "classes_with_gt": int(mask.sum()),
            "metrics/precision(B)": float(precision[mask].mean()) if mask.any() else 0.0,
            "metrics/recall(B)": float(recall[mask].mean()) if mask.any() else 0.0,
            "metrics/mAP50(B)": float(group_ap[:, 0].mean()) if mask.any() else 0.0,
            "metrics/mAP50-95(B)": float(group_ap.mean()) if mask.any() else 0.0,
        }
    return result


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    summary = load_summary(args)
    paths = resolve_validation_paths(args, summary)
    train_vocabulary, train_vocab_path = load_train_vocabulary(args, summary)
    print(
        "resolved evaluation inputs:\n"
        f"  summary={Path(args.summary).expanduser().resolve()} ({'found' if summary else 'not used'})\n"
        f"  category={paths['category']}\n"
        f"  attribute={paths['attribute']}\n"
        f"  affordance={paths['affordance']}\n"
        f"  train_vocab={train_vocab_path}",
        flush=True,
    )
    model = YOLOWorld(str(model_path))
    world_model = model.model
    text_model = str(getattr(world_model, "text_model", "clip:ViT-B/32"))
    if text_model.split(":", 1)[0].lower() not in {"siglip", "siglip2"}:
        raise RuntimeError(f"Expected a SigLIP/SigLIP2 checkpoint, got text_model={text_model!r}")
    model_device = f"cuda:{args.device}" if str(args.device).isdigit() and torch.cuda.is_available() else str(args.device)
    val_device: int | str = int(args.device) if str(args.device).isdigit() and torch.cuda.is_available() else args.device
    world_model.to(model_device)

    router = getattr(world_model.model[-1], "granularity_router", None)
    router_version = int(getattr(router, "router_version", 0)) if router is not None else 0
    category_prior = float(getattr(router, "category_prior", 0.0)) if router is not None else 0.0
    direct_inheritance = bool(
        router is not None
        and router_version >= 5
        and hasattr(router, "attribute_inheritance")
        and hasattr(router, "affordance_inheritance")
    )
    if router is None:
        route_kind = "base"
    elif category_prior == 0.0:
        route_kind = "flat_role_residual"
    elif direct_inheritance:
        route_kind = "category_rooted_gated_residual"
    else:
        route_kind = "category_conditioned_residual"
    router_kind = {
        "base": "none",
        "flat_role_residual": "flat_role_router",
        "category_rooted_gated_residual": "category_rooted_gated_two_child",
        "category_conditioned_residual": "category_conditioned_two_child",
    }[route_kind]
    report = {
        "kind": "bridge_tri_granular_eval",
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "train_vocab": str(train_vocab_path),
        "train_vocab_sha256": sha256_file(train_vocab_path),
        "text_model": text_model,
        "settings": {
            "batch": args.batch,
            "imgsz": args.imgsz,
            "workers": args.workers,
            "device": args.device,
            "text_batch": args.text_batch,
        },
        "roles": {},
        "granularity_router": bool(router is not None),
        "granularity_router_settings": {
            "kind": router_kind,
            "router_version": router_version,
            "bounded_residual": bool(router is not None and router_version >= 4),
            "shared_semantic_residual": bool(router is not None and hasattr(router, "shared")),
            "category_prior": category_prior,
            "category_to_children_stop_gradient": (
                bool(getattr(router, "stop_gradient", True)) if router is not None else False
            ),
            "direct_category_inheritance": bool(direct_inheritance and category_prior > 0.0),
            "inheritance_gates_present": direct_inheritance,
            "inheritance_max": float(getattr(router, "inheritance_max", 0.0)) if direct_inheritance else 0.0,
            "train_stage": str(getattr(router, "train_stage", "unknown")) if router is not None else "none",
            "attribute_route": route_kind,
            "affordance_route": route_kind,
            "gate_statistics": router.inheritance_statistics() if direct_inheritance else {},
        },
    }
    for role in ("category", "attribute", "affordance"):
        data_path = paths[role]
        names = load_names(data_path)
        routed = set_granularity_inference_role(world_model, role)
        # Rebuild the exact role vocabulary every time; never carry class
        # features over from another granularity or a serialized encoder object.
        world_model.clip_model = None
        world_model.set_classes(names, batch=args.text_batch)
        if not isinstance(world_model.clip_model, SigLIP):
            raise RuntimeError(f"{role}: set_classes used {type(world_model.clip_model).__name__}, expected SigLIP")
        world_model.names = names
        if model.predictor:
            model.predictor.model.names = names
        before = feature_fingerprint(world_model.txt_feats)
        print(
            f"[{role}] classes={len(names)} text_sha256={before} routed={routed} data={data_path}", flush=True
        )
        metrics = model.val(
            data=str(data_path),
            split="val",
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            device=val_device,
        )
        after = feature_fingerprint(world_model.txt_feats)
        if after != before:
            raise RuntimeError(f"{role}: validation changed text features ({before} -> {after})")
        values = {key: float(value) for key, value in metrics.results_dict.items()}
        report["roles"][role] = {
            "data": str(data_path),
            "data_sha256": sha256_file(data_path),
            "classes": len(names),
            "text_feature_sha256": before,
            "metrics": values,
            "seen_unseen": seen_unseen_metrics(metrics, names, train_vocabulary[role]),
        }
        print(f"[{role}] {values}", flush=True)

    metric_keys = ("metrics/mAP50(B)", "metrics/mAP50-95(B)")
    report["macro"] = {
        key: sum(report["roles"][role]["metrics"][key] for role in report["roles"]) / 3.0
        for key in metric_keys
    }
    report["worst"] = {
        key: min(report["roles"][role]["metrics"][key] for role in report["roles"])
        for key in metric_keys
    }
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else model_path.parents[1] / "eval_tri_granular.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
