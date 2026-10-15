"""Check public shell-to-Python arguments without loading CUDA or model weights."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


class PublicLauncherTests(unittest.TestCase):
    def capture(self, launcher, **variables):
        with tempfile.TemporaryDirectory() as folder:
            executable = Path(folder) / "python3"
            executable.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
            executable.chmod(0o755)
            env = {**os.environ, **variables, "PATH": folder + os.pathsep + os.environ["PATH"]}
            result = subprocess.run(["bash", str(ROOT / "tools" / launcher), "train"],
                                    cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            argv = json.loads(result.stdout)
        target = ROOT / argv[0]
        spec = importlib.util.spec_from_file_location("public_training_entry", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(sys, "argv", [str(target), *argv[1:]]):
            return module.parse_args()

    def test_routing_variants(self):
        for variant in ("full", "flat", "no_stop_gradient", "no_retention"):
            with self.subTest(variant=variant):
                args = self.capture("run_bridge_granularity_router.sh", VARIANT=variant)
                self.assertEqual(args.batch, 64)
                self.assertEqual(args.epochs, 40)
                self.assertEqual(args.category_prior, 0.0 if variant == "flat" else 1.0)
                self.assertEqual(args.stop_gradient, variant != "no_stop_gradient")
                self.assertEqual(args.attribute_preserve_gain, 0.0 if variant == "no_retention" else 1.0)

    def test_temporal_stage(self):
        args = self.capture("run_bridge_temporal_joint.sh")
        self.assertEqual(args.epochs, 50)
        self.assertEqual(args.batch, 64)
        self.assertEqual(args.box_lr_mult, 0.25)
        self.assertEqual(args.selection_role, "attribute")


if __name__ == "__main__":
    unittest.main()
