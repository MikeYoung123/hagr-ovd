#!/usr/bin/env python3
"""Evaluate Bridge object localization, cross-granularity consistency and long-tail strata."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = (
    str(ROOT) if not os.environ.get("PYTHONPATH") else f"{ROOT}{os.pathsep}{os.environ['PYTHONPATH']}"
)

import torch  # noqa: E402
from ultralytics import YOLOWorld  # noqa: E402
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402
from ultralytics.nn.modules.granularity_router import set_granularity_inference_role  # noqa: E402
from ultralytics.nn.text_model import SigLIP  # noqa: E402
from ultralytics.utils import ops  # noqa: E402

from bridge_granularity_metrics import (  # noqa: E402
    ROLES,
    evaluate_cross_granularity,
    object_average_recall,
    stratified_metrics,
)
from eval_bridge_tri_granular import (  # noqa: E402
    feature_fingerprint,
    load_names,
    load_summary,
    resolve_validation_paths,
    sha256_file,
)


class CaptureDetectionValidator(DetectionValidator):
    """Detection validator that retains compact native-coordinate predictions and GT."""

    last_records: dict[str, dict[str, np.ndarray]] = {}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.capture_records: dict[str, dict[str, np.ndarray]] = {}

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        super().update_metrics(preds, batch)
        for index, pred in enumerate(preds):
            prepared = self._prepare_batch(index, batch)
            pred_scaled = self.scale_preds(pred, prepared)
            gt_boxes = ops.scale_boxes(
                prepared["imgsz"],
                prepared["bboxes"].clone(),
                prepared["ori_shape"],
                ratio_pad=prepared["ratio_pad"],
            )
            image_name = Path(prepared["im_file"]).name
            if image_name in self.capture_records:
                raise RuntimeError(f"Duplicate validation image basename: {image_name}")
            self.capture_records[image_name] = {
                "gt_boxes": gt_boxes.detach().float().cpu().numpy().astype(np.float32, copy=False),
                "gt_cls": prepared["cls"].detach().long().cpu().numpy().astype(np.int32, copy=False),
                "pred_boxes": pred_scaled["bboxes"].detach().float().cpu().numpy().astype(np.float32, copy=False),
                "pred_conf": pred_scaled["conf"].detach().float().cpu().numpy().astype(np.float32, copy=False),
                "pred_cls": pred_scaled["cls"].detach().long().cpu().numpy().astype(np.int32, copy=False),
            }

    def finalize_metrics(self) -> None:
        super().finalize_metrics()
        type(self).last_records = self.capture_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--summary", default="data/bridge_grounding_tri_granular/summary_tri_granular.json")
    parser.add_argument("--strata", required=True, help="Output of build_bridge_eval_strata.py")
    parser.add_argument("--base-dataset", default="data/bridge_grounding")
    parser.add_argument("--category-data", default="")
    parser.add_argument("--attribute-data", default="")
    parser.add_argument("--affordance-data", default="")
    parser.add_argument("--device", default="0", help="Use one validation device; diagnostics do not use DDP.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--text-batch", type=int, default=64)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--cross-iou", type=float, default=0.5)
    parser.add_argument("--gt-align-iou", type=float, default=0.99)
    parser.add_argument("--cache-dir", default="", help="Defaults beside output; cache avoids repeated inference.")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def metrics_payload(metrics) -> dict[str, Any]:
    return {
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "metrics/mAP75(B)": float(metrics.box.map75),
        "per_class": {
            "ap_class_index": np.asarray(metrics.box.ap_class_index, dtype=np.int64).tolist(),
            "all_ap": np.asarray(metrics.box.all_ap, dtype=np.float64).tolist(),
            "precision": np.asarray(metrics.box.p, dtype=np.float64).tolist(),
            "recall": np.asarray(metrics.box.r, dtype=np.float64).tolist(),
            "nt_per_class": np.asarray(metrics.nt_per_class, dtype=np.int64).tolist(),
        },
    }


def _concat(
    records: dict[str, dict[str, np.ndarray]], key: str, width: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    names = sorted(records)
    arrays = [np.asarray(records[name][key]) for name in names]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(array) for array in arrays])
    if arrays and offsets[-1]:
        values = np.concatenate(arrays, axis=0)
    elif width is None:
        values = np.zeros(0, dtype=np.float32)
    else:
        values = np.zeros((0, width), dtype=np.float32)
    return offsets, values


def save_capture(path: Path, records: dict[str, dict[str, np.ndarray]]) -> None:
    names = sorted(records)
    gt_offsets, gt_boxes = _concat(records, "gt_boxes", width=4)
    _, gt_cls = _concat(records, "gt_cls")
    pred_offsets, pred_boxes = _concat(records, "pred_boxes", width=4)
    _, pred_conf = _concat(records, "pred_conf")
    _, pred_cls = _concat(records, "pred_cls")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image_names=np.asarray(names),
        gt_offsets=gt_offsets,
        gt_boxes=gt_boxes.astype(np.float32, copy=False),
        gt_cls=gt_cls.astype(np.int32, copy=False),
        pred_offsets=pred_offsets,
        pred_boxes=pred_boxes.astype(np.float32, copy=False),
        pred_conf=pred_conf.astype(np.float32, copy=False),
        pred_cls=pred_cls.astype(np.int32, copy=False),
    )


def load_capture(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        names = payload["image_names"].tolist()
        gt_offsets = payload["gt_offsets"]
        pred_offsets = payload["pred_offsets"]
        records = {}
        for index, raw_name in enumerate(names):
            name = str(raw_name)
            g0, g1 = int(gt_offsets[index]), int(gt_offsets[index + 1])
            p0, p1 = int(pred_offsets[index]), int(pred_offsets[index + 1])
            records[name] = {
                "gt_boxes": payload["gt_boxes"][g0:g1].copy(),
                "gt_cls": payload["gt_cls"][g0:g1].copy(),
                "pred_boxes": payload["pred_boxes"][p0:p1].copy(),
                "pred_conf": payload["pred_conf"][p0:p1].copy(),
                "pred_cls": payload["pred_cls"][p0:p1].copy(),
            }
    return records


def router_metadata(world_model) -> dict[str, Any]:
    router = getattr(world_model.model[-1], "granularity_router", None)
    version = int(getattr(router, "router_version", 0)) if router is not None else 0
    prior = float(getattr(router, "category_prior", 0.0)) if router is not None else 0.0
    direct = bool(
        router is not None
        and version >= 5
        and hasattr(router, "attribute_inheritance")
        and hasattr(router, "affordance_inheritance")
    )
    if router is None:
        kind = "none"
    elif prior == 0.0:
        kind = "flat_role_router"
    elif direct:
        kind = "category_rooted_gated_two_child"
    else:
        kind = "category_conditioned_two_child"
    return {
        "kind": kind,
        "router_version": version,
        "category_prior": prior,
        "direct_category_inheritance": bool(direct and prior > 0.0),
        "gate_statistics": router.inheritance_statistics() if direct else {},
    }


def set_role_vocabulary(model: YOLOWorld, role: str, names: list[str], text_batch: int) -> tuple[str, bool]:
    world_model = model.model
    routed = set_granularity_inference_role(world_model, role)
    world_model.clip_model = None
    world_model.set_classes(names, batch=text_batch)
    if not isinstance(world_model.clip_model, SigLIP):
        raise RuntimeError(f"{role}: set_classes used {type(world_model.clip_model).__name__}, expected SigLIP")
    world_model.names = names
    if model.predictor:
        model.predictor.model.names = names
    return feature_fingerprint(world_model.txt_feats), routed


def run_or_load_validation(
    model: YOLOWorld,
    model_sha: str,
    role: str,
    names: list[str],
    data_path: Path,
    args: argparse.Namespace,
    cache_dir: Path,
    object_mode: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    suffix = "object" if object_mode else role
    npz_path = cache_dir / f"{suffix}.npz"
    meta_path = cache_dir / f"{suffix}.json"
    signature = {
        "version": 1,
        "model_sha256": model_sha,
        "role": role,
        "object_mode": object_mode,
        "data_sha256": sha256_file(data_path),
        "classes": len(names),
        "settings": {
            "batch": args.batch,
            "imgsz": args.imgsz,
            "text_batch": args.text_batch,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
        },
    }
    if not args.rebuild_cache and npz_path.is_file() and meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            print(f"[{suffix}] reusing verified cache: {npz_path}", flush=True)
            return metadata["evaluation"], load_capture(npz_path)

    text_sha, routed = set_role_vocabulary(model, role, names, args.text_batch)
    CaptureDetectionValidator.last_records = {}
    print(
        f"[{suffix}] inference classes={len(names)} routed={routed} single_cls={object_mode} data={data_path}",
        flush=True,
    )
    metrics = model.val(
        validator=CaptureDetectionValidator,
        data=str(data_path),
        split="val",
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=int(args.device) if args.device.isdigit() and torch.cuda.is_available() else args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        single_cls=object_mode,
        agnostic_nms=object_mode,
        plots=False,
        save=False,
        verbose=False,
    )
    records = CaptureDetectionValidator.last_records
    if not records:
        raise RuntimeError(f"{suffix}: validator captured no records")
    evaluation = {**metrics_payload(metrics), "text_feature_sha256": text_sha, "routed": bool(routed)}
    save_capture(npz_path, records)
    meta_path.write_text(
        json.dumps({"signature": signature, "evaluation": evaluation}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return evaluation, records


def load_strata(path: Path, validation_paths: dict[str, Path]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "bridge_tri_granular_frequency_strata":
        raise ValueError(f"Unsupported strata file: {path}")
    for role in ROLES:
        expected = payload.get("validation_yaml_sha256", {}).get(role)
        actual = sha256_file(validation_paths[role])
        if expected != actual:
            raise RuntimeError(
                f"{role} validation YAML differs from immutable strata vocabulary: {expected} != {actual}"
            )
    return payload


def main() -> None:
    args = parse_args()
    if "," in args.device:
        raise ValueError("Diagnostics require one device so prediction caches remain deterministic")
    if args.max_det < 300:
        raise ValueError("Use --max-det >= 300 so AR@300 is defined")
    for value, name in ((args.cross_iou, "cross-iou"), (args.gt_align_iou, "gt-align-iou")):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")

    model_path = Path(args.model).expanduser().resolve()
    strata_path = Path(args.strata).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for path in (model_path, strata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = load_summary(args)
    validation_paths = resolve_validation_paths(args, summary)
    strata = load_strata(strata_path, validation_paths)
    names_by_role = {role: load_names(validation_paths[role]) for role in ROLES}
    model_sha = sha256_file(model_path)

    model = YOLOWorld(str(model_path))
    world_model = model.model
    text_model = str(getattr(world_model, "text_model", "clip:ViT-B/32"))
    if text_model.split(":", 1)[0].lower() not in {"siglip", "siglip2"}:
        raise RuntimeError(f"Expected SigLIP/SigLIP2 checkpoint, got {text_model!r}")
    model_device = f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else args.device
    world_model.to(model_device)

    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else output.parent / f"{output.stem}_cache"
    )
    role_records = {}
    role_reports = {}
    for role in ROLES:
        evaluation, records = run_or_load_validation(
            model,
            model_sha,
            role,
            names_by_role[role],
            validation_paths[role],
            args,
            cache_dir,
        )
        role_records[role] = records
        role_reports[role] = {
            "data": str(validation_paths[role]),
            "data_sha256": sha256_file(validation_paths[role]),
            "classes": len(names_by_role[role]),
            "metrics": evaluation["metrics"],
            "metrics/mAP75(B)": evaluation["metrics/mAP75(B)"],
            "stratified": stratified_metrics(
                evaluation["per_class"], names_by_role[role], strata["roles"][role]
            ),
        }

    object_evaluation, object_records = run_or_load_validation(
        model,
        model_sha,
        "category",
        names_by_role["category"],
        validation_paths["category"],
        args,
        cache_dir,
        object_mode=True,
    )
    object_metrics = {
        **object_evaluation["metrics"],
        "metrics/mAP75(B)": object_evaluation["metrics/mAP75(B)"],
        **object_average_recall(object_records),
        "definition": {
            "vocabulary": "immutable category validation vocabulary",
            "ground_truth": "all category GT classes collapsed to object",
            "predictions": "all predicted classes collapsed to object with class-agnostic NMS",
            "note": "No new images, labels or validation split are generated.",
        },
    }
    cross = evaluate_cross_granularity(role_records, args.cross_iou, args.gt_align_iou)

    report = {
        "kind": "bridge_granularity_diagnostics",
        "version": 2,
        "model": str(model_path),
        "model_sha256": model_sha,
        "text_model": text_model,
        "router": router_metadata(world_model),
        "settings": {
            "device": args.device,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "workers": args.workers,
            "text_batch": args.text_batch,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "cross_iou": args.cross_iou,
            "gt_align_iou": args.gt_align_iou,
        },
        "strata": {
            "path": str(strata_path),
            "sha256": sha256_file(strata_path),
            "definition": strata["definition"],
        },
        "roles": role_reports,
        "class_agnostic_object": object_metrics,
        "cross_granularity": cross,
        "cache_dir": str(cache_dir),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    concise = {
        "model": report["model"],
        "router": report["router"],
        "class_agnostic_object": report["class_agnostic_object"],
        "cross_granularity": report["cross_granularity"],
        "stratified_mAP50-95": {
            role: {
                group: values["metrics/mAP50-95(B)"]
                for group, values in role_reports[role]["stratified"].items()
            }
            for role in ROLES
        },
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
