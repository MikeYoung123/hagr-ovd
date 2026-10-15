"""Exercise release identity and leakage checks with small portable fixtures."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from release_data import validate_training_release


class ReleaseValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.train = self.root / "bridge_grounding_tri_granular/train.json"
        self.vocab = self.train.with_name("vocab.json")
        self.temporal = self.root / "bridge_grounding_temporal_joint/temporal.json"
        self.val = self.root / "bridge_grounding/val.yaml"
        self.write(self.root / "bridge_grounding/splits/train_rows.json", [1])
        self.write(self.root / "bridge_grounding/splits/val_rows.json", [2])
        self.write(self.vocab, {"category": ["carrot"]})
        self.write(self.val, {})
        self.write(self.train, {"images": [self.image(1, 0)], "annotations": []})
        self.write(self.temporal, {"images": [self.image(1, 1)], "annotations": []})
        validation = {**self.image(2, 0), "split": "val"}
        self.write(self.root / "bridge_grounding/annotations/bridge_qwen_category_val.json",
                   {"images": [validation], "annotations": []})
        self.manifest()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    @staticmethod
    def image(row, frame):
        return {"id": 1, "bridge_row_idx": row, "video_idx": row,
                "frame_idx": frame, "split": "train", "file_name": f"row{row}_frame{frame}.jpg"}

    def manifest(self):
        entries = []
        for path in self.root.rglob("*"):
            if path.is_file() and path.name != "release_manifest.json":
                entries.append({"path": path.relative_to(self.root).as_posix(),
                                "bytes": path.stat().st_size,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.write(self.root / "release_manifest.json", {"files": entries})

    def validate(self):
        return validate_training_release(self.train, self.vocab, [self.val], self.temporal)

    def test_valid_release(self):
        self.assertEqual(self.validate(), self.root / "release_manifest.json")

    def test_changed_file_is_rejected(self):
        self.write(self.vocab, {"category": ["modified"]})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.validate()

    def test_validation_episode_as_temporal_training_is_rejected(self):
        self.write(self.temporal, {"images": [self.image(2, 1)], "annotations": []})
        self.manifest()
        with self.assertRaisesRegex(ValueError, "split violation"):
            self.validate()

    def test_frame_zero_neighbor_is_rejected(self):
        self.write(self.temporal, {"images": [self.image(1, 0)], "annotations": []})
        self.manifest()
        with self.assertRaisesRegex(ValueError, "frame index"):
            self.validate()

    def test_same_video_with_different_rows_is_rejected(self):
        image = {**self.image(1, 0), "video_idx": 2}
        self.write(self.train, {"images": [image], "annotations": []})
        self.manifest()
        with self.assertRaisesRegex(ValueError, "Video leakage"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
