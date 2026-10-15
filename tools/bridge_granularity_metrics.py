#!/usr/bin/env python3
"""Pure NumPy metrics for Bridge tri-granular diagnostic evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


ROLES = ("category", "attribute", "affordance")


def box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Return pairwise IoU for xyxy boxes."""
    boxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)
    if not len(boxes1) or not len(boxes2):
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    intersection = wh[..., 0] * wh[..., 1]
    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0.0, None) * np.clip(
        boxes1[:, 3] - boxes1[:, 1], 0.0, None
    )
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0.0, None) * np.clip(
        boxes2[:, 3] - boxes2[:, 1], 0.0, None
    )
    return intersection / np.clip(area1[:, None] + area2[None, :] - intersection, 1e-9, None)


def align_boxes(reference: np.ndarray, candidate: np.ndarray, min_iou: float = 0.99) -> np.ndarray:
    """One-to-one align candidate boxes to reference boxes, refusing incomplete alignments."""
    reference = np.asarray(reference, dtype=np.float32).reshape(-1, 4)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1, 4)
    if len(reference) != len(candidate):
        raise ValueError(f"GT counts differ: reference={len(reference)} candidate={len(candidate)}")
    if not len(reference):
        return np.zeros(0, dtype=np.int64)
    ious = box_iou_numpy(reference, candidate)
    order = np.argsort(ious, axis=None)[::-1]
    assigned_ref: set[int] = set()
    assigned_candidate: set[int] = set()
    result = np.full(len(reference), -1, dtype=np.int64)
    for flat_index in order:
        ref_index, candidate_index = np.unravel_index(int(flat_index), ious.shape)
        if float(ious[ref_index, candidate_index]) < min_iou:
            break
        if ref_index in assigned_ref or candidate_index in assigned_candidate:
            continue
        result[ref_index] = candidate_index
        assigned_ref.add(ref_index)
        assigned_candidate.add(candidate_index)
    if (result < 0).any():
        worst = float(max(ious[index].max(initial=0.0) for index in np.where(result < 0)[0]))
        raise ValueError(
            f"Could not align {int((result < 0).sum())}/{len(reference)} GT boxes at IoU>={min_iou}; "
            f"best unresolved IoU={worst:.6f}"
        )
    return result


def match_predictions(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_conf: np.ndarray,
    iou_threshold: float,
    gt_cls: np.ndarray | None = None,
    gt_label_sets: list[set[int]] | None = None,
    pred_cls: np.ndarray | None = None,
    class_aware: bool = False,
    max_predictions: int | None = None,
) -> np.ndarray:
    """Greedily assign confidence-sorted predictions to GT and return pred index per GT."""
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_conf = np.asarray(pred_conf, dtype=np.float32).reshape(-1)
    result = np.full(len(gt_boxes), -1, dtype=np.int64)
    if not len(gt_boxes) or not len(pred_boxes):
        return result
    if class_aware:
        if pred_cls is None or (gt_cls is None and gt_label_sets is None):
            raise ValueError("Class-aware matching requires pred_cls and GT labels")
        if gt_label_sets is not None and len(gt_label_sets) != len(gt_boxes):
            raise ValueError("GT label-set count differs from GT box count")
        if gt_cls is not None:
            gt_cls = np.asarray(gt_cls, dtype=np.int64).reshape(-1)
        pred_cls = np.asarray(pred_cls, dtype=np.int64).reshape(-1)
    order = np.argsort(-pred_conf, kind="stable")
    if max_predictions is not None:
        order = order[:max_predictions]
    ious = box_iou_numpy(pred_boxes[order], gt_boxes)
    unmatched = np.ones(len(gt_boxes), dtype=bool)
    for ordered_index, pred_index in enumerate(order):
        candidates = unmatched.copy()
        if class_aware:
            if gt_label_sets is None:
                candidates &= gt_cls == pred_cls[pred_index]
            else:
                candidates &= np.asarray(
                    [int(pred_cls[pred_index]) in labels for labels in gt_label_sets], dtype=bool
                )
        if not candidates.any():
            continue
        candidate_indices = np.flatnonzero(candidates)
        best_local = int(np.argmax(ious[ordered_index, candidate_indices]))
        gt_index = int(candidate_indices[best_local])
        if float(ious[ordered_index, gt_index]) >= iou_threshold:
            result[gt_index] = int(pred_index)
            unmatched[gt_index] = False
    return result


def object_average_recall(
    records: dict[str, dict[str, np.ndarray]],
    max_detections: tuple[int, ...] = (100, 300),
    iou_thresholds: tuple[float, ...] = tuple(float(value) for value in np.arange(0.5, 0.96, 0.05)),
) -> dict[str, float | int | list[float]]:
    """Compute class-agnostic proposal recall averaged over images and IoU thresholds."""
    total_gt = int(sum(len(record["gt_boxes"]) for record in records.values()))
    output: dict[str, float | int | list[float]] = {
        "images": len(records),
        "gt_instances": total_gt,
        "iou_thresholds": list(iou_thresholds),
    }
    for limit in max_detections:
        recalls = []
        for threshold in iou_thresholds:
            matched = 0
            for record in records.values():
                assignments = match_predictions(
                    record["gt_boxes"],
                    record["pred_boxes"],
                    record["pred_conf"],
                    threshold,
                    max_predictions=limit,
                )
                matched += int((assignments >= 0).sum())
            recalls.append(matched / total_gt if total_gt else 0.0)
        output[f"AR@{limit}"] = float(np.mean(recalls)) if recalls else 0.0
        output[f"R50@{limit}"] = float(recalls[0]) if recalls else 0.0
        output[f"R75@{limit}"] = float(recalls[5]) if len(recalls) > 5 else 0.0
    return output


def stratified_metrics(
    per_class: dict[str, Any], names: list[str], label_strata: dict[str, dict[str, Any]]
) -> dict[str, dict[str, float | int]]:
    """Aggregate standard per-class AP over immutable frequency groups."""
    indices = np.asarray(per_class["ap_class_index"], dtype=np.int64)
    all_ap = np.asarray(per_class["all_ap"], dtype=np.float64)
    precision = np.asarray(per_class["precision"], dtype=np.float64)
    recall = np.asarray(per_class["recall"], dtype=np.float64)
    nt_per_class = np.asarray(per_class["nt_per_class"], dtype=np.int64)
    if len(indices) != len(all_ap) or len(indices) != len(precision) or len(indices) != len(recall):
        raise ValueError("Per-class AP, precision and recall lengths differ")
    missing = [name for name in names if name not in label_strata]
    if missing:
        raise ValueError(f"Strata miss {len(missing)} validation labels, for example {missing[:3]}")

    class_groups = [str(label_strata[name]["group"]) for name in names]
    groups = {
        "all": lambda group: True,
        "seen": lambda group: group != "unseen",
        "unseen": lambda group: group == "unseen",
        "rare": lambda group: group == "rare",
        "common": lambda group: group == "common",
        "frequent": lambda group: group == "frequent",
    }
    output = {}
    total_gt = int(nt_per_class.sum())
    for group_name, predicate in groups.items():
        vocabulary_indices = np.asarray([i for i, group in enumerate(class_groups) if predicate(group)], dtype=np.int64)
        row_mask = np.asarray([predicate(class_groups[int(index)]) for index in indices], dtype=bool)
        group_ap = all_ap[row_mask]
        gt_instances = int(nt_per_class[vocabulary_indices].sum()) if len(vocabulary_indices) else 0
        output[group_name] = {
            "classes_in_val_vocabulary": int(len(vocabulary_indices)),
            "classes_with_gt": int(row_mask.sum()),
            "gt_instances": gt_instances,
            "gt_instance_fraction": gt_instances / total_gt if total_gt else 0.0,
            "metrics/precision(B)": float(precision[row_mask].mean()) if row_mask.any() else 0.0,
            "metrics/recall(B)": float(recall[row_mask].mean()) if row_mask.any() else 0.0,
            "metrics/mAP50(B)": float(group_ap[:, 0].mean()) if row_mask.any() else 0.0,
            "metrics/mAP50-95(B)": float(group_ap.mean()) if row_mask.any() else 0.0,
        }
    return output


def evaluate_cross_granularity(
    role_records: dict[str, dict[str, dict[str, np.ndarray]]],
    iou_threshold: float = 0.5,
    gt_align_iou: float = 0.99,
) -> dict[str, Any]:
    """Measure joint semantic recall, spatial agreement and root-child compatibility."""
    if set(role_records) != set(ROLES):
        raise ValueError(f"Expected role caches {ROLES}, got {sorted(role_records)}")
    image_sets = {role: set(role_records[role]) for role in ROLES}
    if len({frozenset(values) for values in image_sets.values()}) != 1:
        sizes = {role: len(values) for role, values in image_sets.items()}
        raise ValueError(f"Validation image sets differ across roles: {sizes}")

    child_to_categories: dict[str, dict[int, set[int]]] = {
        "attribute": defaultdict(set),
        "affordance": defaultdict(set),
    }
    aligned: dict[str, dict[str, Any]] = {}
    raw_gt_counts = {role: 0 for role in ROLES}
    merged_gt_counts = {role: 0 for role in ROLES}
    for image_name in sorted(image_sets["category"]):
        groups: list[dict[str, Any]] = []
        records = {role: role_records[role][image_name] for role in ROLES}
        for role in ROLES:
            record = records[role]
            raw_gt_counts[role] += len(record["gt_boxes"])
            for box, class_id in zip(record["gt_boxes"], record["gt_cls"]):
                if groups:
                    representatives = np.stack([group["box"] for group in groups])
                    ious = box_iou_numpy(np.asarray(box).reshape(1, 4), representatives)[0]
                    group_index = int(np.argmax(ious))
                    if float(ious[group_index]) < gt_align_iou:
                        group_index = -1
                else:
                    group_index = -1
                if group_index < 0:
                    groups.append(
                        {
                            "box": np.asarray(box, dtype=np.float32),
                            "labels": {candidate_role: set() for candidate_role in ROLES},
                        }
                    )
                    group_index = len(groups) - 1
                groups[group_index]["labels"][role].add(int(class_id))

        incomplete = [
            index
            for index, group in enumerate(groups)
            if any(not group["labels"][role] for role in ROLES)
        ]
        if incomplete:
            examples = [
                {role: sorted(groups[index]["labels"][role]) for role in ROLES}
                for index in incomplete[:3]
            ]
            raise ValueError(
                f"{image_name}: {len(incomplete)}/{len(groups)} geometry groups miss at least one role; "
                f"examples={examples}. The three validation datasets are not object-aligned."
            )
        gt_boxes = np.stack([group["box"] for group in groups]).astype(np.float32, copy=False)
        gt_label_sets = {
            role: [group["labels"][role] for group in groups]
            for role in ROLES
        }
        for role in ROLES:
            merged_gt_counts[role] += sum(max(len(labels) - 1, 0) for labels in gt_label_sets[role])
        item = {"gt_boxes": gt_boxes, "records": records, "gt_label_sets": gt_label_sets}
        for object_index in range(len(groups)):
            for role in ("attribute", "affordance"):
                for child_cls in gt_label_sets[role][object_index]:
                    child_to_categories[role][int(child_cls)].update(
                        int(value) for value in gt_label_sets["category"][object_index]
                    )
        aligned[image_name] = item

    total_gt = int(sum(len(item["gt_boxes"]) for item in aligned.values()))
    semantic_count = {role: 0 for role in ROLES}
    spatial_count = {role: 0 for role in ROLES}
    joint_semantic = 0
    joint_spatial = 0
    category_semantic = 0
    conditional = {"attribute": 0, "affordance": 0}
    compatibility = {
        "attribute": {"compatible": 0, "evaluated": 0, "unmapped": 0},
        "affordance": {"compatible": 0, "evaluated": 0, "unmapped": 0},
    }
    pair_ious: dict[str, list[float]] = {
        "category_attribute": [],
        "category_affordance": [],
        "attribute_affordance": [],
    }
    triplet_box_ious: list[float] = []

    for item in aligned.values():
        gt_boxes = item["gt_boxes"]
        semantic_matches = {}
        spatial_matches = {}
        for role in ROLES:
            record = item["records"][role]
            gt_label_sets = item["gt_label_sets"][role]
            semantic_matches[role] = match_predictions(
                gt_boxes,
                record["pred_boxes"],
                record["pred_conf"],
                iou_threshold,
                gt_label_sets=gt_label_sets,
                pred_cls=record["pred_cls"],
                class_aware=True,
            )
            spatial_matches[role] = match_predictions(
                gt_boxes, record["pred_boxes"], record["pred_conf"], iou_threshold
            )
            semantic_count[role] += int((semantic_matches[role] >= 0).sum())
            spatial_count[role] += int((spatial_matches[role] >= 0).sum())

        semantic_ok = {role: semantic_matches[role] >= 0 for role in ROLES}
        spatial_ok = {role: spatial_matches[role] >= 0 for role in ROLES}
        joint_semantic += int(np.logical_and.reduce([semantic_ok[role] for role in ROLES]).sum())
        joint_spatial += int(np.logical_and.reduce([spatial_ok[role] for role in ROLES]).sum())
        category_semantic += int(semantic_ok["category"].sum())
        for role in ("attribute", "affordance"):
            conditional[role] += int((semantic_ok["category"] & semantic_ok[role]).sum())

            both_spatial = spatial_ok["category"] & spatial_ok[role]
            for gt_index in np.flatnonzero(both_spatial):
                category_pred_index = spatial_matches["category"][gt_index]
                child_pred_index = spatial_matches[role][gt_index]
                category_cls = int(item["records"]["category"]["pred_cls"][category_pred_index])
                child_cls = int(item["records"][role]["pred_cls"][child_pred_index])
                allowed = child_to_categories[role].get(child_cls)
                if not allowed:
                    compatibility[role]["unmapped"] += 1
                    continue
                compatibility[role]["evaluated"] += 1
                compatibility[role]["compatible"] += int(category_cls in allowed)

        pair_definitions = (
            ("category_attribute", "category", "attribute"),
            ("category_affordance", "category", "affordance"),
            ("attribute_affordance", "attribute", "affordance"),
        )
        per_gt_triplet: dict[int, list[float]] = defaultdict(list)
        for pair_name, left, right in pair_definitions:
            both_correct = semantic_ok[left] & semantic_ok[right]
            for gt_index in np.flatnonzero(both_correct):
                left_index = semantic_matches[left][gt_index]
                right_index = semantic_matches[right][gt_index]
                left_box = item["records"][left]["pred_boxes"][left_index : left_index + 1]
                right_box = item["records"][right]["pred_boxes"][right_index : right_index + 1]
                value = float(box_iou_numpy(left_box, right_box)[0, 0])
                pair_ious[pair_name].append(value)
                per_gt_triplet[int(gt_index)].append(value)
        for values in per_gt_triplet.values():
            if len(values) == 3:
                triplet_box_ious.append(float(np.mean(values)))

    roles_report = {}
    for role in ROLES:
        roles_report[role] = {
            "spatial_recall": spatial_count[role] / total_gt if total_gt else 0.0,
            "semantic_recall": semantic_count[role] / total_gt if total_gt else 0.0,
            "semantic_given_spatial": semantic_count[role] / spatial_count[role] if spatial_count[role] else 0.0,
            "spatial_matches": spatial_count[role],
            "semantic_matches": semantic_count[role],
        }
    compatibility_report = {}
    for role, counts in compatibility.items():
        evaluated = counts["evaluated"]
        compatible = counts["compatible"]
        compatibility_report[role] = {
            **counts,
            "compatibility_rate": compatible / evaluated if evaluated else 0.0,
            "conflict_rate": 1.0 - compatible / evaluated if evaluated else 0.0,
            "mapping_classes": len(child_to_categories[role]),
        }
    return {
        "definition": {
            "iou_threshold": iou_threshold,
            "gt_alignment_iou": gt_align_iou,
            "semantic_match": "one-to-one confidence-ordered match with exact role class and IoU threshold",
            "spatial_match": "one-to-one confidence-ordered class-agnostic match at IoU threshold",
            "gt_object_unit": (
                "role labels whose boxes have IoU at or above gt_alignment_iou are merged into one object; "
                "all labels in that role-specific set are valid"
            ),
            "compatibility": "predicted child label must map to the predicted category in immutable validation triples",
        },
        "images": len(aligned),
        "gt_instances": total_gt,
        "ground_truth_consolidation": {
            "raw_role_labels": raw_gt_counts,
            "unique_objects": total_gt,
            "extra_labels_merged_into_objects": merged_gt_counts,
        },
        "roles": roles_report,
        "joint_spatial_recall": joint_spatial / total_gt if total_gt else 0.0,
        "joint_semantic_recall": joint_semantic / total_gt if total_gt else 0.0,
        "joint_semantic_given_spatial": joint_semantic / joint_spatial if joint_spatial else 0.0,
        "conditional_child_recall": {
            role: conditional[role] / category_semantic if category_semantic else 0.0
            for role in ("attribute", "affordance")
        },
        "root_child_compatibility": compatibility_report,
        "correct_box_agreement": {
            **{key: float(np.mean(values)) if values else 0.0 for key, values in pair_ious.items()},
            "triplet_mean_iou": float(np.mean(triplet_box_ious)) if triplet_box_ious else 0.0,
            "triplets": len(triplet_box_ious),
        },
    }
