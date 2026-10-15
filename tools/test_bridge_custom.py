# Ultralytics AGPL-3.0 License
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import hashlib

import torch
from ultralytics import YOLOWorld
from ultralytics.nn.text_model import CLIP, SigLIP
from ultralytics.utils import YAML

DEFAULT_BRIDGE_DATA = "data/bridge_grounding/bridge_qwen_attribute_phrase_det.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLO-World on Bridge/LVIS and audit its text encoder.")
    parser.add_argument("--model", required=True, help="Local CLIP or SigLIP YOLO-World checkpoint.")
    parser.add_argument(
        "--data",
        default=DEFAULT_BRIDGE_DATA,
        help="Detection yaml used for validation. Use the Bridge *_det.yaml to test on Bridge.",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Validation device, e.g. 0. Defaults to cuda:0 if available.")
    parser.add_argument("--text-batch", type=int, default=64, help="Batch size used when encoding class text features.")
    parser.add_argument(
        "--expect-text-encoder",
        choices=("auto", "clip", "siglip"),
        default="auto",
        help="Fail early if the checkpoint is not using the expected text encoder family.",
    )
    return parser.parse_args()


def encoder_family(variant):
    base = variant.split(":", 1)[0].lower()
    if base == "clip":
        return CLIP
    if base in {"siglip", "siglip2"}:
        return SigLIP
    raise ValueError(f"Unsupported text encoder in checkpoint: {variant!r}")


def feature_fingerprint(features):
    data = features.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def load_class_names(data_yaml_path: str) -> list[str]:
    data_yaml = YAML.load(data_yaml_path)
    names = data_yaml["names"]
    if isinstance(names, dict):
        return [str(v) for _, v in sorted(names.items(), key=lambda x: int(x[0]))]
    return [str(x) for x in names]


def resolve_devices(device_arg):
    if device_arg is None:
        return ("cuda:0", 0) if torch.cuda.is_available() else ("cpu", "cpu")
    device_arg = str(device_arg)
    if device_arg.isdigit() and torch.cuda.is_available():
        return f"cuda:{device_arg}", int(device_arg)
    return device_arg, device_arg


def encoder_family_name(variant: str) -> str:
    family = variant.split(":", 1)[0].lower()
    return "siglip" if family in {"siglip", "siglip2"} else family


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Local checkpoint not found: {model_path}. This evaluator never downloads model weights; "
            "pass an existing local file with --model."
        )
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Validation YAML not found: {data_path}")
    args.model = str(model_path)
    args.data = str(data_path)
    model = YOLOWorld(args.model)
    world_model = model.model
    model_device, val_device = resolve_devices(args.device)

    # Old official YOLO-World checkpoints do not have text_model and therefore use CLIP ViT-B/32.
    text_model = getattr(world_model, "text_model", "clip:ViT-B/32")
    expected_encoder = encoder_family(text_model)
    print(f"checkpoint: {args.model}", flush=True)
    print(f"configured text encoder: {text_model}", flush=True)
    if args.expect_text_encoder != "auto" and encoder_family_name(text_model) != args.expect_text_encoder:
        raise RuntimeError(
            f"Expected a {args.expect_text_encoder} checkpoint, but this checkpoint is configured as {text_model!r}."
        )

    class_names = load_class_names(args.data)
    print(f"data yaml: {args.data}", flush=True)
    print(
        f"encoding {len(class_names)} class texts on {model_device} with text_batch={args.text_batch}...",
        flush=True,
    )

    # Never reuse an encoder object that may have been serialized in a checkpoint or left by an earlier set_classes().
    model.model.to(model_device)
    world_model.clip_model = None
    world_model.set_classes(class_names, batch=args.text_batch)
    model.model.names = class_names
    if model.predictor:
        model.predictor.model.names = class_names
    print("text features ready.", flush=True)

    actual_encoder = world_model.clip_model
    if not isinstance(actual_encoder, expected_encoder):
        raise RuntimeError(
            f"Text encoder mismatch: checkpoint requests {text_model!r}, "
            f"but set_classes() used {type(actual_encoder).__name__}."
        )

    features_before_val = world_model.txt_feats.detach().float().cpu().clone()
    fingerprint_before_val = feature_fingerprint(features_before_val)
    print(f"actual encoder class: {type(actual_encoder).__name__}", flush=True)
    print(f"classes: {len(class_names)}; text features: {tuple(features_before_val.shape)}", flush=True)
    print(f"text feature SHA256: {fingerprint_before_val}", flush=True)

    metrics = model.val(data=args.data, split="val", batch=args.batch, imgsz=args.imgsz, device=val_device)

    fingerprint_after_val = feature_fingerprint(world_model.txt_feats)
    if fingerprint_after_val != fingerprint_before_val:
        raise RuntimeError("Text features changed during validation.")
    print(f"validation used unchanged text features: {fingerprint_after_val}", flush=True)
    print(metrics.results_dict)


if __name__ == "__main__":
    main()
