#!/usr/bin/env python3
"""Build immutable train-frequency strata metadata for Bridge validation vocabularies.

This does not create a new detection dataset. It reads the existing tri-granular
training JSON and validation YAML vocabularies, then records unseen/rare/common/
frequent membership using training data only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROLES = ("category", "attribute", "affordance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", required=True, help="bridge_qwen_tri_granular_train.json")
    parser.add_argument("--category-data", required=True)
    parser.add_argument("--attribute-data", required=True)
    parser.add_argument("--affordance-data", required=True)
    parser.add_argument(
        "--frequency-unit",
        choices=("video", "image", "instance"),
        default="video",
        help="Default video frequency avoids counting adjacent Bridge frames as independent evidence.",
    )
    parser.add_argument("--rare-max", type=int, default=10)
    parser.add_argument("--common-max", type=int, default=100)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_names(path: Path) -> list[str]:
    # Bridge detection YAMLs are generated with a JSON-compatible, one-line
    # ``names`` value. Keep this metadata builder independent of torch/PyYAML
    # so strata can also be audited on the synchronization host.
    raw_names = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("names:"):
            raw_names = line.split(":", 1)[1].strip()
            break
    if raw_names is None:
        raise ValueError(f"No ordered names vocabulary in {path}")
    names = json.loads(raw_names)
    if isinstance(names, dict):
        return [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
    if isinstance(names, list):
        return [str(value) for value in names]
    raise ValueError(f"No ordered names vocabulary in {path}")


def count_training_frequencies(train_json: Path) -> dict[str, dict[str, dict[str, int]]]:
    payload = json.loads(train_json.read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in payload.get("images", [])}
    if not images or not payload.get("annotations"):
        raise ValueError(f"Incomplete training JSON: {train_json}")

    instances = {role: Counter() for role in ROLES}
    image_sets: dict[str, dict[str, set[int]]] = {role: defaultdict(set) for role in ROLES}
    video_sets: dict[str, dict[str, set[int]]] = {role: defaultdict(set) for role in ROLES}
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        image = images[image_id]
        video_id = int(image.get("video_idx", image.get("bridge_row_idx", image_id)))
        tri_roles = annotation.get("tri_roles") or {}
        for role in ROLES:
            label = str(tri_roles.get(role, "")).strip()
            if not label:
                raise ValueError(f"Annotation {annotation.get('id')} has no tri_roles.{role}")
            instances[role][label] += 1
            image_sets[role][label].add(image_id)
            video_sets[role][label].add(video_id)

    return {
        role: {
            label: {
                "instance": int(instances[role][label]),
                "image": len(image_sets[role][label]),
                "video": len(video_sets[role][label]),
            }
            for label in instances[role]
        }
        for role in ROLES
    }


def frequency_group(frequency: int, rare_max: int, common_max: int) -> str:
    if frequency <= 0:
        return "unseen"
    if frequency <= rare_max:
        return "rare"
    if frequency <= common_max:
        return "common"
    return "frequent"


def build_strata(
    frequencies: dict[str, dict[str, dict[str, int]]],
    validation_names: dict[str, list[str]],
    frequency_unit: str,
    rare_max: int,
    common_max: int,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    roles = {}
    summary = {}
    for role in ROLES:
        labels = {}
        histogram = Counter()
        for label in validation_names[role]:
            counts = frequencies[role].get(label, {"instance": 0, "image": 0, "video": 0})
            group = frequency_group(int(counts[frequency_unit]), rare_max, common_max)
            labels[label] = {"group": group, "train_frequency": int(counts[frequency_unit]), **counts}
            histogram[group] += 1
        roles[role] = labels
        summary[role] = {
            "validation_classes": len(labels),
            "groups": {group: int(histogram.get(group, 0)) for group in ("unseen", "rare", "common", "frequent")},
            "training_labels_not_in_validation": len(set(frequencies[role]) - set(validation_names[role])),
        }
    return roles, summary


def main() -> None:
    args = parse_args()
    if args.rare_max < 1 or args.common_max <= args.rare_max:
        raise ValueError("Require 1 <= rare-max < common-max")
    paths = {
        "train_json": Path(args.train_json).expanduser().resolve(),
        "category": Path(args.category_data).expanduser().resolve(),
        "attribute": Path(args.attribute_data).expanduser().resolve(),
        "affordance": Path(args.affordance_data).expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    validation_names = {role: load_names(paths[role]) for role in ROLES}
    frequencies = count_training_frequencies(paths["train_json"])
    roles, summary = build_strata(
        frequencies, validation_names, args.frequency_unit, args.rare_max, args.common_max
    )
    output = Path(args.output).expanduser().resolve()
    report = {
        "kind": "bridge_tri_granular_frequency_strata",
        "version": 1,
        "definition": {
            "source": "training annotations only",
            "frequency_unit": args.frequency_unit,
            "unseen": 0,
            "rare": f"1..{args.rare_max}",
            "common": f"{args.rare_max + 1}..{args.common_max}",
            "frequent": f">{args.common_max}",
            "note": "video is the preferred unit because adjacent Bridge frames are correlated",
        },
        "train_json": str(paths["train_json"]),
        "train_json_sha256": sha256_file(paths["train_json"]),
        "validation_yamls": {role: str(paths[role]) for role in ROLES},
        "validation_yaml_sha256": {role: sha256_file(paths[role]) for role in ROLES},
        "summary": summary,
        "roles": roles,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("kind", "definition", "summary")}, indent=2, ensure_ascii=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
