#!/usr/bin/env python3
"""CPU-only unit tests for Bridge diagnostic metric definitions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from bridge_granularity_metrics import (  # noqa: E402
    align_boxes,
    evaluate_cross_granularity,
    object_average_recall,
    stratified_metrics,
)
from build_bridge_eval_strata import build_strata, frequency_group  # noqa: E402


def record(gt_boxes, gt_cls, pred_boxes, pred_conf, pred_cls):
    return {
        "gt_boxes": np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4),
        "gt_cls": np.asarray(gt_cls, dtype=np.int32),
        "pred_boxes": np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4),
        "pred_conf": np.asarray(pred_conf, dtype=np.float32),
        "pred_cls": np.asarray(pred_cls, dtype=np.int32),
    }


class TestBridgeGranularityDiagnostics(unittest.TestCase):
    def setUp(self) -> None:
        boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
        self.records = {
            "category": {
                "image.jpg": record(boxes, [0, 1], boxes, [0.9, 0.8], [0, 1]),
            },
            "attribute": {
                # Reverse GT order to prove GT alignment is geometry-based.
                "image.jpg": record(
                    boxes[::-1],
                    [1, 0],
                    boxes,
                    [0.9, 0.8],
                    [0, 0],
                ),
            },
            "affordance": {
                "image.jpg": record(boxes, [0, 1], boxes, [0.9, 0.8], [0, 1]),
            },
        }

    def test_box_alignment_handles_reordered_ground_truth(self):
        order = align_boxes(
            np.asarray([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
            np.asarray([[20, 20, 30, 30], [0, 0, 10, 10]], dtype=np.float32),
        )
        self.assertEqual(order.tolist(), [1, 0])

    def test_cross_granularity_metrics(self):
        result = evaluate_cross_granularity(self.records)
        self.assertEqual(result["gt_instances"], 2)
        self.assertAlmostEqual(result["roles"]["category"]["semantic_recall"], 1.0)
        self.assertAlmostEqual(result["roles"]["attribute"]["semantic_recall"], 0.5)
        self.assertAlmostEqual(result["joint_spatial_recall"], 1.0)
        self.assertAlmostEqual(result["joint_semantic_recall"], 0.5)
        self.assertAlmostEqual(result["conditional_child_recall"]["attribute"], 0.5)
        self.assertAlmostEqual(result["conditional_child_recall"]["affordance"], 1.0)
        self.assertAlmostEqual(result["root_child_compatibility"]["attribute"]["conflict_rate"], 0.5)
        self.assertAlmostEqual(result["root_child_compatibility"]["affordance"]["conflict_rate"], 0.0)
        self.assertAlmostEqual(result["correct_box_agreement"]["triplet_mean_iou"], 1.0)

    def test_same_box_child_labels_are_one_multilabel_object(self):
        box = [[0, 0, 10, 10]]
        records = {
            "category": {"image.jpg": record(box, [0], box, [0.9], [0])},
            "attribute": {
                "image.jpg": record(box + box, [0, 1], box, [0.9], [1]),
            },
            "affordance": {"image.jpg": record(box, [0], box, [0.9], [0])},
        }
        result = evaluate_cross_granularity(records)
        self.assertEqual(result["gt_instances"], 1)
        self.assertEqual(
            result["ground_truth_consolidation"]["extra_labels_merged_into_objects"]["attribute"], 1
        )
        self.assertAlmostEqual(result["roles"]["attribute"]["semantic_recall"], 1.0)
        self.assertAlmostEqual(result["joint_semantic_recall"], 1.0)

    def test_object_average_recall(self):
        result = object_average_recall(self.records["category"])
        self.assertAlmostEqual(result["AR@100"], 1.0)
        self.assertAlmostEqual(result["AR@300"], 1.0)
        self.assertAlmostEqual(result["R75@300"], 1.0)

    def test_stratified_metrics(self):
        per_class = {
            "ap_class_index": [0, 1, 2, 3],
            "all_ap": [[0.1] * 10, [0.2] * 10, [0.3] * 10, [0.4] * 10],
            "precision": [0.1, 0.2, 0.3, 0.4],
            "recall": [0.2, 0.3, 0.4, 0.5],
            "nt_per_class": [1, 2, 3, 4],
        }
        names = ["unseen", "rare", "common", "frequent"]
        labels = {
            name: {"group": name, "train_frequency": index}
            for index, name in enumerate(names)
        }
        result = stratified_metrics(per_class, names, labels)
        self.assertAlmostEqual(result["all"]["metrics/mAP50-95(B)"], 0.25)
        self.assertAlmostEqual(result["seen"]["metrics/mAP50-95(B)"], 0.3)
        self.assertAlmostEqual(result["unseen"]["metrics/mAP50-95(B)"], 0.1)
        self.assertEqual(result["frequent"]["gt_instances"], 4)

    def test_training_only_frequency_groups(self):
        self.assertEqual(frequency_group(0, 10, 100), "unseen")
        self.assertEqual(frequency_group(10, 10, 100), "rare")
        self.assertEqual(frequency_group(11, 10, 100), "common")
        self.assertEqual(frequency_group(101, 10, 100), "frequent")
        frequencies = {
            role: {
                "seen rare": {"instance": 20, "image": 12, "video": 2},
                "seen frequent": {"instance": 200, "image": 150, "video": 120},
            }
            for role in ("category", "attribute", "affordance")
        }
        names = {role: ["unseen", "seen rare", "seen frequent"] for role in frequencies}
        roles, summary = build_strata(frequencies, names, "video", 10, 100)
        self.assertEqual(roles["category"]["unseen"]["group"], "unseen")
        self.assertEqual(roles["category"]["seen rare"]["group"], "rare")
        self.assertEqual(roles["category"]["seen frequent"]["group"], "frequent")
        self.assertEqual(summary["attribute"]["groups"]["unseen"], 1)


if __name__ == "__main__":
    unittest.main()
