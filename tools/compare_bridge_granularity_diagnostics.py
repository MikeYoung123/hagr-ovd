#!/usr/bin/env python3
"""Compare flat and category-rooted Bridge diagnostic reports with strict protocol checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROLES = ("category", "attribute", "affordance")
GROUPS = ("all", "seen", "unseen", "rare", "common", "frequent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Flat diagnostic JSON")
    parser.add_argument("--candidate", required=True, help="Category-rooted diagnostic JSON")
    parser.add_argument("--baseline-label", default="flat")
    parser.add_argument("--candidate-label", default="category-rooted")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "bridge_granularity_diagnostics":
        raise ValueError(f"Unsupported diagnostic report: {path}")
    return payload


def check_protocol(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    if baseline["strata"]["sha256"] != candidate["strata"]["sha256"]:
        raise RuntimeError("Strata metadata differ; refusing an incomparable result")
    for key in ("imgsz", "conf", "iou", "max_det", "cross_iou", "gt_align_iou"):
        if baseline["settings"].get(key) != candidate["settings"].get(key):
            raise RuntimeError(
                f"Evaluation setting {key} differs: {baseline['settings'].get(key)} != {candidate['settings'].get(key)}"
            )
    for role in ROLES:
        if baseline["roles"][role].get("data_sha256") != candidate["roles"][role].get("data_sha256"):
            raise RuntimeError(f"{role} validation data differ")


def cell(baseline: float, candidate: float) -> dict[str, float]:
    return {"baseline": float(baseline), "candidate": float(candidate), "delta": float(candidate - baseline)}


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    standard = {}
    stratified = {}
    for role in ROLES:
        standard[role] = {
            key: cell(baseline["roles"][role]["metrics"][key], candidate["roles"][role]["metrics"][key])
            for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)")
        }
        stratified[role] = {
            group: cell(
                baseline["roles"][role]["stratified"][group]["metrics/mAP50-95(B)"],
                candidate["roles"][role]["stratified"][group]["metrics/mAP50-95(B)"],
            )
            for group in GROUPS
        }

    object_keys = ("metrics/mAP50(B)", "metrics/mAP75(B)", "metrics/mAP50-95(B)", "AR@100", "AR@300")
    object_comparison = {
        key: cell(baseline["class_agnostic_object"][key], candidate["class_agnostic_object"][key])
        for key in object_keys
    }
    base_cross, candidate_cross = baseline["cross_granularity"], candidate["cross_granularity"]
    cross = {
        "joint_spatial_recall": cell(base_cross["joint_spatial_recall"], candidate_cross["joint_spatial_recall"]),
        "joint_semantic_recall": cell(base_cross["joint_semantic_recall"], candidate_cross["joint_semantic_recall"]),
        "joint_semantic_given_spatial": cell(
            base_cross["joint_semantic_given_spatial"], candidate_cross["joint_semantic_given_spatial"]
        ),
        "attribute_given_category": cell(
            base_cross["conditional_child_recall"]["attribute"],
            candidate_cross["conditional_child_recall"]["attribute"],
        ),
        "affordance_given_category": cell(
            base_cross["conditional_child_recall"]["affordance"],
            candidate_cross["conditional_child_recall"]["affordance"],
        ),
        "attribute_conflict_rate": cell(
            base_cross["root_child_compatibility"]["attribute"]["conflict_rate"],
            candidate_cross["root_child_compatibility"]["attribute"]["conflict_rate"],
        ),
        "affordance_conflict_rate": cell(
            base_cross["root_child_compatibility"]["affordance"]["conflict_rate"],
            candidate_cross["root_child_compatibility"]["affordance"]["conflict_rate"],
        ),
        "triplet_box_iou": cell(
            base_cross["correct_box_agreement"]["triplet_mean_iou"],
            candidate_cross["correct_box_agreement"]["triplet_mean_iou"],
        ),
    }
    return {
        "standard": standard,
        "class_agnostic_object": object_comparison,
        "cross_granularity": cross,
        "stratified": stratified,
    }


def percentage(value: float) -> str:
    return f"{100.0 * value:.3f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def render_markdown(result: dict[str, Any], baseline_label: str, candidate_label: str) -> str:
    lines = ["# Bridge Granularity Diagnostic Comparison", ""]
    standard_rows = []
    for role in ROLES:
        value = result["standard"][role]["metrics/mAP50-95(B)"]
        standard_rows.append(
            [role, percentage(value["baseline"]), percentage(value["candidate"]), percentage(value["delta"])]
        )
    lines += [
        "## Standard role AP50-95",
        "",
        markdown_table(["role", baseline_label, candidate_label, "delta (AP points)"], standard_rows),
        "",
        "## Class-agnostic object",
        "",
    ]
    object_rows = []
    for key, value in result["class_agnostic_object"].items():
        object_rows.append(
            [key, percentage(value["baseline"]), percentage(value["candidate"]), percentage(value["delta"])]
        )
    lines += [markdown_table(["metric", baseline_label, candidate_label, "delta (points)"], object_rows), ""]

    cross_rows = []
    lower_is_better = {"attribute_conflict_rate", "affordance_conflict_rate"}
    for key, value in result["cross_granularity"].items():
        suffix = " (lower is better)" if key in lower_is_better else ""
        cross_rows.append(
            [key + suffix, percentage(value["baseline"]), percentage(value["candidate"]), percentage(value["delta"])]
        )
    lines += [
        "## Cross-granularity consistency",
        "",
        markdown_table(["metric", baseline_label, candidate_label, "delta (points)"], cross_rows),
        "",
        "## Frequency-stratified AP50-95",
        "",
    ]
    frequency_rows = []
    for role in ROLES:
        for group in GROUPS:
            value = result["stratified"][role][group]
            frequency_rows.append(
                [role, group, percentage(value["baseline"]), percentage(value["candidate"]), percentage(value["delta"])]
            )
    lines.append(
        markdown_table(["role", "stratum", baseline_label, candidate_label, "delta (AP points)"], frequency_rows)
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    baseline, candidate = load_report(baseline_path), load_report(candidate_path)
    check_protocol(baseline, candidate)
    values = compare(baseline, candidate)
    report = {
        "kind": "bridge_granularity_diagnostics_comparison",
        "baseline": {"label": args.baseline_label, "path": str(baseline_path), "model": baseline["model"]},
        "candidate": {"label": args.candidate_label, "path": str(candidate_path), "model": candidate["model"]},
        "comparison": values,
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(values, args.baseline_label, args.candidate_label), encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"))
    print(f"saved: {output_json}")
    print(f"saved: {output_md}")


if __name__ == "__main__":
    main()
