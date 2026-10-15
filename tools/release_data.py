"""Validate portable published annotations without historical construction summaries."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def verify_files(paths: list[Path]) -> Path:
    """Verify consumed inputs against their common published SHA-256 manifest."""
    resolved = [Path(p).expanduser().resolve() for p in paths]
    if not resolved:
        raise ValueError("No release inputs supplied")
    manifest = next((p / "release_manifest.json" for p in resolved[0].parents
                     if (p / "release_manifest.json").is_file()), None)
    if manifest is None:
        raise FileNotFoundError("Keep release_manifest.json beside the three dataset directories")
    entries = {x["path"]: x for x in json.loads(manifest.read_text())["files"]}
    for path in resolved:
        relative = path.relative_to(manifest.parent).as_posix()
        if relative not in entries:
            raise ValueError(f"Input is not covered by release manifest: {path}")
        entry = entries[relative]
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Release size mismatch: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Release SHA-256 mismatch: {relative}")
    return manifest


def iter_images(path: Path):
    """Stream the images array of the released COCO files with bounded memory.

    These generated files place images immediately after info. Refuse a different
    layout rather than accidentally interpreting another nested images field.
    """
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as handle:
        buffer = handle.read(65536)
        match = re.search(r'"images"\s*:\s*\[', buffer)
        if not match:
            raise ValueError(f"No release-format images array near the start of {path}")
        buffer = buffer[match.end():]
        while True:
            buffer = buffer.lstrip()
            if not buffer:
                buffer = handle.read(65536)
                if not buffer:
                    raise ValueError(f"Truncated images array: {path}")
                continue
            if buffer[0] == "]":
                return
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue
            try:
                image, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = handle.read(65536)
                if not chunk:
                    raise ValueError(f"Truncated image record: {path}") from None
                buffer += chunk
                continue
            if not isinstance(image, dict):
                raise ValueError(f"Invalid image record: {path}")
            yield image
            buffer = buffer[end:]


def validate_training_release(train_json: Path, train_vocab: Path, validation_yamls: list[Path],
                              temporal_json: Path | None = None,
                              original_images: Path | None = None,
                              temporal_images: Path | None = None) -> Path:
    """Check file identity, episode membership and selected-neighbor constraints."""
    root = Path(train_json).resolve().parent.parent
    train_split = root / "bridge_grounding/splits/train_rows.json"
    val_split = root / "bridge_grounding/splits/val_rows.json"
    val_json = root / "bridge_grounding/annotations/bridge_qwen_category_val.json"
    files = [Path(train_json), Path(train_vocab), train_split, val_split, val_json, *validation_yamls]
    if temporal_json is not None:
        files.append(Path(temporal_json))
    manifest = verify_files(files)
    train_rows = set(json.loads(train_split.read_text()))
    val_rows = set(json.loads(val_split.read_text()))
    if not train_rows or not val_rows or train_rows & val_rows:
        raise ValueError("Empty or overlapping train/validation episode split")
    validation_rows, validation_videos = set(), set()
    for image in iter_images(val_json):
        row = int(image["bridge_row_idx"])
        if row not in val_rows or image.get("split") != "val":
            raise ValueError(f"Validation split violation: {image['file_name']}")
        validation_rows.add(row)
        validation_videos.add(int(image["video_idx"]))
    if validation_rows != val_rows:
        raise ValueError("Validation images do not cover the fixed validation split")

    def check_images(annotation: Path, image_dir: Path | None, temporal: bool):
        rows, videos, count = set(), set(), 0
        for image in iter_images(annotation):
            row, video = int(image["bridge_row_idx"]), int(image["video_idx"])
            frame = int(image["frame_idx"])
            if row not in train_rows or row in val_rows or image.get("split") != "train":
                raise ValueError(f"Training split violation: {image['file_name']}")
            if video in validation_videos:
                raise ValueError(f"Video leakage into training: {image['file_name']}")
            if row in rows or video in videos:
                raise ValueError(f"Multiple selected frames per episode: {image['file_name']}")
            if (temporal and frame <= 0) or (not temporal and frame != 0):
                raise ValueError(f"Unexpected frame index: {image['file_name']}")
            if image_dir is not None and not (image_dir / image["file_name"]).is_file():
                raise FileNotFoundError(f"Missing reconstructed image: {image_dir / image['file_name']}")
            rows.add(row)
            videos.add(video)
            count += 1
        if not count:
            raise ValueError(f"Empty training images: {annotation}")
        return rows, videos

    rows, videos = check_images(Path(train_json), original_images, False)
    if rows != train_rows:
        raise ValueError("Original training images do not cover the fixed training split")
    if temporal_json is not None:
        neighbor_rows, neighbor_videos = check_images(Path(temporal_json), temporal_images, True)
        if not neighbor_rows <= rows or not neighbor_videos <= videos:
            raise ValueError("Temporal episode does not belong to original training data")
        print(f"Temporal training episode coverage: {len(neighbor_rows)}/{len(rows)}")
    return manifest
