#!/usr/bin/env python3
"""Benchmark offline-vocabulary compute cost for base, flat, and HAGR Bridge detectors."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = (
    str(ROOT) if not os.environ.get("PYTHONPATH") else f"{ROOT}{os.pathsep}{os.environ['PYTHONPATH']}"
)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from ultralytics import YOLOWorld, __version__ as ultralytics_version  # noqa: E402
from ultralytics.nn.modules.granularity_router import set_granularity_inference_role  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402


ROLES = ("category", "attribute", "affordance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Aligned detector without a granularity router.")
    parser.add_argument("--flat-model", required=True, help="Matched flat three-role router checkpoint.")
    parser.add_argument("--hagr-model", required=True, help="Full category-rooted HAGR checkpoint.")
    parser.add_argument("--base-dataset", default="data/bridge_grounding")
    parser.add_argument("--category-data", default="")
    parser.add_argument("--attribute-data", default="")
    parser.add_argument("--affordance-data", default="")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--text-batch", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-flops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="runs/world_granularity_router/hagr_efficiency.json")
    return parser.parse_args()


def require_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_names(path: Path) -> list[str]:
    names = YAML.load(str(path)).get("names")
    if isinstance(names, dict):
        return [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
    if isinstance(names, list):
        return [str(value) for value in names]
    raise ValueError(f"No ordered names vocabulary in {path}")


def resolve_role_yamls(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.base_dataset).expanduser().resolve()
    values = {
        "category": args.category_data or root / "bridge_qwen_category_det.yaml",
        "attribute": args.attribute_data or root / "bridge_qwen_attribute_phrase_det.yaml",
        "affordance": args.affordance_data or root / "bridge_qwen_affordance_phrase_det.yaml",
    }
    return {role: require_file(str(path), f"{role} validation YAML") for role, path in values.items()}


def resolve_device(value: str) -> torch.device:
    if value.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {value} requested but CUDA is unavailable")
        return torch.device(f"cuda:{value}")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value} requested but CUDA is unavailable")
    return device


class OfflineVocabularyForward(nn.Module):
    """Expose the exact tensor forward used after text features have been cached."""

    def __init__(self, detector: nn.Module, text_features: torch.Tensor) -> None:
        super().__init__()
        self.detector = detector
        self.register_buffer("text_features", text_features, persistent=False)

    def forward(self, image: torch.Tensor):
        return self.detector(image, txt_feats=self.text_features)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))


def time_forward(
    wrapper: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float], dict[str, float]]:
    with torch.inference_mode():
        for _ in range(warmup):
            wrapper(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            resident_allocated = torch.cuda.memory_allocated(device) / (1024**2)
            resident_reserved = torch.cuda.memory_reserved(device) / (1024**2)
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            for start, end in zip(starts, ends):
                start.record()
                wrapper(image)
                end.record()
            torch.cuda.synchronize(device)
            elapsed = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
            memory = {
                "resident_allocated_mib": resident_allocated,
                "resident_reserved_mib": resident_reserved,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
            }
        else:
            elapsed = []
            for _ in range(iterations):
                start = time.perf_counter()
                wrapper(image)
                elapsed.append((time.perf_counter() - start) * 1000.0)
            memory = {}
    mean_ms = statistics.fmean(elapsed)
    latency = {
        "mean_ms": mean_ms,
        "std_ms": statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0,
        "p50_ms": percentile(elapsed, 0.50),
        "p95_ms": percentile(elapsed, 0.95),
        "fps": 1000.0 / mean_ms,
    }
    return latency, memory


def profile_flops(wrapper: nn.Module, image: torch.Tensor) -> tuple[float, str, str]:
    """Return FLOPs for one batch-1 forward; prefer THOP and retain failure provenance."""
    try:
        import thop

        with torch.inference_mode():
            operations = thop.profile(wrapper, inputs=(image,), verbose=False)[0]
        return float(operations * 2.0 / 1e9), "thop_2x_macs", ""
    except Exception as error:
        thop_error = f"{type(error).__name__}: {error}"
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if image.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.inference_mode(), torch.profiler.profile(activities=activities, with_flops=True) as profiler:
            wrapper(image)
        operations = sum(float(item.flops) for item in profiler.key_averages())
        return operations / 1e9, "torch_profiler_partial", thop_error
    except Exception as error:
        return 0.0, "unavailable", f"THOP={thop_error}; profiler={type(error).__name__}: {error}"


def install_vocabulary(
    detector: nn.Module,
    names: list[str],
    text_batch: int,
    cache: dict[tuple[str, str], torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, object]]:
    vocabulary_hash = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    text_model = str(getattr(detector, "text_model", "unknown"))
    key = (text_model, vocabulary_hash)
    started = time.perf_counter()
    cached = key in cache
    if not cached:
        detector.clip_model = None
        detector.set_classes(names, batch=text_batch)
        cache[key] = detector.txt_feats.detach().float().cpu().contiguous()
    features = cache[key].to(device=device, dtype=dtype)
    detector.txt_feats = features
    detector.model[-1].nc = len(names)
    detector.names = names
    detector.clip_model = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return features, {
        "text_model": text_model,
        "vocabulary_sha256": vocabulary_hash,
        "text_features_cached": cached,
        "setup_seconds": time.perf_counter() - started,
    }


def model_metadata(detector: nn.Module, checkpoint: Path) -> dict[str, object]:
    router = getattr(detector.model[-1], "granularity_router", None)
    total = sum(parameter.numel() for parameter in detector.parameters())
    router_parameters = sum(parameter.numel() for parameter in router.parameters()) if router is not None else 0
    category_prior = float(getattr(router, "category_prior", 0.0)) if router is not None else 0.0
    direct_inheritance = bool(
        router is not None
        and hasattr(router, "attribute_inheritance")
        and hasattr(router, "affordance_inheritance")
    )
    if router is None:
        kind = "base"
    elif category_prior == 0.0:
        kind = "flat_role_router"
    elif not direct_inheritance:
        kind = "category_conditioned_router"
    else:
        kind = "category_rooted_hagr"
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_mib": checkpoint.stat().st_size / (1024**2),
        "kind": kind,
        "parameters": total,
        "router_parameters": router_parameters,
        "router_parameter_overhead_percent": 100.0 * router_parameters / max(total - router_parameters, 1),
        "routing_stage_trainable_parameters": router_parameters,
        "router_version": int(getattr(router, "router_version", 0)) if router is not None else 0,
        "category_prior": category_prior,
        "direct_category_inheritance": direct_inheritance,
        "stop_gradient": bool(getattr(router, "stop_gradient", True)) if router is not None else False,
    }


def benchmark_model(
    label: str,
    checkpoint: Path,
    vocabularies: dict[str, tuple[Path, list[str]]],
    args: argparse.Namespace,
    device: torch.device,
    text_cache: dict[tuple[str, str], torch.Tensor],
) -> dict[str, object]:
    print(f"\n[{label}] loading {checkpoint}", flush=True)
    api_model = YOLOWorld(str(checkpoint))
    detector = api_model.model.to(device).eval()
    detector.clip_model = None
    dtype = torch.float16 if args.half and device.type == "cuda" else torch.float32
    if dtype == torch.float16:
        detector.half()
    else:
        detector.float()
    result = model_metadata(detector, checkpoint)
    expected_kind = {
        "base": "base",
        "flat": "flat_role_router",
        "hagr": "category_rooted_hagr",
    }[label]
    if result["kind"] != expected_kind:
        raise RuntimeError(
            f"{label} checkpoint resolved as {result['kind']!r}, expected {expected_kind!r}: {checkpoint}"
        )
    result["roles"] = {}
    for role in ROLES:
        yaml_path, names = vocabularies[role]
        routed = set_granularity_inference_role(detector, role)
        features, text_info = install_vocabulary(
            detector, names, args.text_batch, text_cache, device, dtype
        )
        image = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device, dtype=dtype)
        wrapper = OfflineVocabularyForward(detector, features).eval()
        flops, flops_method, flops_warning = (0.0, "disabled", "")
        if args.profile_flops:
            flops, flops_method, flops_warning = profile_flops(wrapper, image)
        latency, memory = time_forward(wrapper, image, device, args.warmup, args.iterations)
        role_result = {
            "validation_yaml": str(yaml_path),
            "classes": len(names),
            "routed": routed,
            "precision": "fp16" if dtype == torch.float16 else "fp32",
            "gflops": flops,
            "flops_method": flops_method,
            "forward_latency": latency,
            "gpu_memory": memory,
            **text_info,
        }
        if flops_warning:
            role_result["flops_warning"] = flops_warning
        result["roles"][role] = role_result
        print(
            f"[{label}/{role}] classes={len(names)} routed={routed} GFLOPs={flops:.3f} "
            f"latency={latency['mean_ms']:.3f} ms FPS={latency['fps']:.2f}",
            flush=True,
        )
        del wrapper, image, features
    del detector, api_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def relative_overhead(candidate: dict, reference: dict) -> dict[str, object]:
    output: dict[str, object] = {
        "parameter_delta": int(candidate["parameters"] - reference["parameters"]),
        "parameter_delta_percent": 100.0
        * (candidate["parameters"] - reference["parameters"])
        / max(reference["parameters"], 1),
        "checkpoint_mib_delta": float(candidate["checkpoint_mib"] - reference["checkpoint_mib"]),
        "roles": {},
    }
    for role in ROLES:
        current = candidate["roles"][role]
        baseline = reference["roles"][role]
        current_ms = current["forward_latency"]["mean_ms"]
        baseline_ms = baseline["forward_latency"]["mean_ms"]
        output["roles"][role] = {
            "gflops_delta": current["gflops"] - baseline["gflops"],
            "gflops_delta_percent": 100.0
            * (current["gflops"] - baseline["gflops"])
            / max(baseline["gflops"], 1e-12),
            "latency_delta_ms": current_ms - baseline_ms,
            "latency_delta_percent": 100.0 * (current_ms - baseline_ms) / max(baseline_ms, 1e-12),
        }
    return output


def main() -> None:
    args = parse_args()
    if args.imgsz < 32 or args.text_batch < 1 or args.warmup < 1 or args.iterations < 2:
        raise ValueError("imgsz>=32, text-batch>=1, warmup>=1 and iterations>=2 are required")
    device = resolve_device(args.device)
    role_yamls = resolve_role_yamls(args)
    vocabularies = {role: (path, load_names(path)) for role, path in role_yamls.items()}
    checkpoints = {
        "base": require_file(args.base_model, "base checkpoint"),
        "flat": require_file(args.flat_model, "flat-router checkpoint"),
        "hagr": require_file(args.hagr_model, "HAGR checkpoint"),
    }
    report = {
        "kind": "bridge_granularity_efficiency",
        "protocol": {
            "scope": (
                "detector forward after offline text-feature caching; "
                "preprocessing, NMS, and text encoding excluded"
            ),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch": torch.__version__,
            "ultralytics": ultralytics_version,
            "imgsz": args.imgsz,
            "batch": 1,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "half": bool(args.half and device.type == "cuda"),
            "profile_flops": args.profile_flops,
            "trainable_parameter_definition": (
                "additional parameters optimized in the frozen-detector routing stage; "
                "the base initialization therefore has zero routing-stage trainable parameters"
            ),
        },
        "models": {},
    }
    text_cache: dict[tuple[str, str], torch.Tensor] = {}
    for label, checkpoint in checkpoints.items():
        report["models"][label] = benchmark_model(
            label, checkpoint, vocabularies, args, device, text_cache
        )
    report["overhead_vs_base"] = {
        label: relative_overhead(report["models"][label], report["models"]["base"])
        for label in ("flat", "hagr")
    }
    report["hagr_vs_flat"] = relative_overhead(report["models"]["hagr"], report["models"]["flat"])
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved efficiency report to {output}", flush=True)


if __name__ == "__main__":
    main()
