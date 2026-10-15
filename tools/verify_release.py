#!/usr/bin/env python3
"""Check downloaded annotations, split integrity, and optionally reconstructed images."""

import argparse
import json
from pathlib import Path

from release_data import validate_training_release, verify_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    manifest = json.loads((root / "release_manifest.json").read_text())
    verify_files([root / item["path"] for item in manifest["files"]])
    base = root / "bridge_grounding"
    validate_training_release(
        root / "bridge_grounding_tri_granular/bridge_qwen_tri_granular_train.json",
        root / "bridge_grounding_tri_granular/bridge_qwen_tri_granular_train_vocab.json",
        [base / f"bridge_qwen_{role}_det.yaml" for role in ("category", "attribute_phrase", "affordance_phrase")],
        root / "bridge_grounding_temporal_joint/bridge_qwen_tri_granular_temporal_train.json",
        base / "images" if args.check_images else None,
        root / "bridge_temporal_propagated/images/train" if args.check_images else None,
    )
    if args.check_images:
        from release_data import iter_images
        for role in ("category", "attribute_phrase", "affordance_phrase"):
            annotation = base / f"annotations/bridge_qwen_{role}_val.json"
            for image in iter_images(annotation):
                path = base / f"yolo/{role}/images/val" / image["file_name"]
                if not path.is_file():
                    raise FileNotFoundError(path)
    print(f"Verified {len(manifest['files'])} published files and original/temporal split integrity")


if __name__ == "__main__":
    main()
