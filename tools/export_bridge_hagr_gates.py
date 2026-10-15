#!/usr/bin/env python3
"""Export learned HAGR category-inheritance gates and generate Figure 4 source plots.

This script is intentionally checkpoint-only: it reads the learned parameters
without running validation inference or modifying the model.  The JSON and CSV
files are the auditable source data for the distribution and heatmap panels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = (
    str(ROOT) if not os.environ.get("PYTHONPATH") else f"{ROOT}{os.pathsep}{os.environ['PYTHONPATH']}"
)

import torch  # noqa: E402
from ultralytics import YOLOWorld, __version__ as ultralytics_version  # noqa: E402


ROLES = ("attribute", "affordance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Full HAGR v5 best.pt checkpoint.")
    parser.add_argument(
        "--output-dir",
        default="runs/world_granularity_router/hagr_gate_export",
        help="Directory for JSON, CSV, PNG and PDF outputs.",
    )
    parser.add_argument(
        "--scale-names",
        default="P3,P4,P5",
        help="Comma-separated feature-scale names in model order.",
    )
    parser.add_argument(
        "--near-zero",
        type=float,
        default=1e-6,
        help="Absolute threshold used only for the reported near-zero fraction.",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate publication-source PNG/PDF plots (requires matplotlib).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_router(model_path: Path):
    """Load one YOLOWorld checkpoint and return its detector, head and v5 router."""
    api_model = YOLOWorld(str(model_path))
    detector = getattr(api_model, "model", None)
    if detector is None or not hasattr(detector, "model"):
        raise RuntimeError(
            "YOLOWorld checkpoint did not expose the expected detector at api_model.model.model."
        )
    try:
        head = detector.model[-1]
    except (IndexError, TypeError) as error:
        raise RuntimeError("Could not resolve the final YOLO-World detection head.") from error
    router = getattr(head, "granularity_router", None)
    if router is None:
        raise RuntimeError("Checkpoint has no granularity_router; a Full HAGR checkpoint is required.")

    version = int(getattr(router, "router_version", 0))
    prior = float(getattr(router, "category_prior", 0.0))
    if version < 5:
        raise RuntimeError(f"Checkpoint router_version={version}; direct HAGR v5 gates are required.")
    if prior <= 0.0:
        raise RuntimeError(
            f"Checkpoint category_prior={prior}; this is a flat control, not the Full HAGR model."
        )
    for role in ROLES:
        if not hasattr(router, f"{role}_inheritance"):
            raise RuntimeError(f"Checkpoint is missing {role}_inheritance gates.")
    return api_model, detector, head, router


def tensor_parameter_name(detector, target: torch.Tensor) -> str:
    """Return the actual checkpoint parameter path for provenance."""
    for name, parameter in detector.named_parameters():
        if parameter is target:
            return name
    return "unresolved"


def summarize(values: torch.Tensor, near_zero: float) -> dict[str, Any]:
    values = values.detach().float().cpu().reshape(-1)
    if values.numel() == 0:
        raise RuntimeError("Encountered an empty inheritance-gate tensor.")
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("Inheritance-gate tensor contains NaN or Inf.")
    absolute = values.abs()
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95]))
    positive = int(values.gt(near_zero).sum())
    negative = int(values.lt(-near_zero).sum())
    near = int(values.abs().le(near_zero).sum())
    count = int(values.numel())
    return {
        "channels": count,
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean_abs": float(absolute.mean()),
        "max_abs": float(absolute.max()),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "positive_channels": positive,
        "negative_channels": negative,
        "near_zero_channels": near,
        "positive_fraction": positive / count,
        "negative_fraction": negative / count,
        "near_zero_fraction": near / count,
    }


def collect_gates(detector, router, scale_names: list[str], near_zero: float) -> list[dict[str, Any]]:
    """Collect raw logits and bounded coefficients in an auditable long-form representation."""
    num_scales = int(getattr(router, "num_scales", 0))
    if num_scales < 1:
        raise RuntimeError(f"Invalid router num_scales={num_scales}.")
    if len(scale_names) != num_scales:
        raise ValueError(
            f"--scale-names contains {len(scale_names)} names but checkpoint has {num_scales} scales."
        )

    records: list[dict[str, Any]] = []
    channel_counts: set[int] = set()
    for role in ROLES:
        gates = getattr(router, f"{role}_inheritance")
        if len(gates) != num_scales:
            raise RuntimeError(f"{role} has {len(gates)} gates, expected {num_scales}.")
        for scale_index, (scale_name, gate) in enumerate(zip(scale_names, gates)):
            if not hasattr(gate, "logits") or not callable(getattr(gate, "coefficients", None)):
                raise RuntimeError(f"{role}/{scale_name} is not a CategoryInheritanceGate.")
            logits = gate.logits.detach().float().cpu().reshape(-1)
            coefficients = gate.coefficients().detach().float().cpu().reshape(-1)
            if logits.shape != coefficients.shape:
                raise RuntimeError(f"{role}/{scale_name}: logits and coefficient shapes differ.")
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(coefficients).all()):
                raise RuntimeError(f"{role}/{scale_name}: gate contains NaN or Inf.")
            max_gain = float(getattr(gate, "max_gain", getattr(router, "inheritance_max", math.nan)))
            if not math.isfinite(max_gain) or max_gain <= 0.0:
                raise RuntimeError(f"{role}/{scale_name}: invalid max_gain={max_gain}.")
            if float(coefficients.abs().max()) > max_gain + 1e-6:
                raise RuntimeError(
                    f"{role}/{scale_name}: coefficient exceeds tanh bound {max_gain}."
                )
            channel_counts.add(int(coefficients.numel()))
            records.append(
                {
                    "role": role,
                    "scale_index": scale_index,
                    "scale": scale_name,
                    "parameter_name": tensor_parameter_name(detector, gate.logits),
                    "max_gain": max_gain,
                    "statistics": summarize(coefficients, near_zero),
                    "logits": logits.tolist(),
                    "coefficients": coefficients.tolist(),
                }
            )
    if len(channel_counts) != 1:
        raise RuntimeError(f"Gate channel counts differ across role/scale pairs: {sorted(channel_counts)}")
    return records


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["role", "scale_index", "scale", "channel", "logit", "coefficient", "abs_coefficient"]
        )
        for record in records:
            for channel, (logit, coefficient) in enumerate(
                zip(record["logits"], record["coefficients"])
            ):
                writer.writerow(
                    [
                        record["role"],
                        record["scale_index"],
                        record["scale"],
                        channel,
                        format(logit, ".17g"),
                        format(coefficient, ".17g"),
                        format(abs(coefficient), ".17g"),
                    ]
                )


def plot_figure(path_stem: Path, records: list[dict[str, Any]], dpi: int) -> None:
    """Plot raw signed channel distributions and a channel-aligned heatmap."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "matplotlib is required for --plots. Install it or rerun with --no-plots; JSON/CSV remain sufficient."
        ) from error

    labels = [f"{record['role'].capitalize()}-{record['scale']}" for record in records]
    values = [record["coefficients"] for record in records]
    matrix = values
    limit = max(abs(value) for row in matrix for value in row)
    limit = max(limit, 1e-6)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15.0, 5.2),
        gridspec_kw={"width_ratios": [1.15, 1.85]},
        constrained_layout=True,
    )

    # Panel (a): every violin is formed from all learned channels of one gate.
    colors = ["#3572A5"] * (len(records) // 2) + ["#D97706"] * (len(records) // 2)
    for position, (row, color) in enumerate(zip(values, colors), start=1):
        if max(row) - min(row) <= 1e-12:
            axes[0].plot([position - 0.25, position + 0.25], [row[0], row[0]], color=color, linewidth=2)
            continue
        violin = axes[0].violinplot(
            [row], positions=[position], widths=0.78, showmeans=True, showmedians=True, showextrema=True
        )
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.55)
        for key in ("cmeans", "cmedians", "cmins", "cmaxes", "cbars"):
            violin[key].set_color("#202020")
            violin[key].set_linewidth(0.8)
    axes[0].axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(range(1, len(labels) + 1), labels, rotation=35, ha="right")
    axes[0].set_ylabel(r"Signed gate coefficient $g_r^{(l)}$")
    axes[0].set_title("(a) Learned channel-wise distributions")
    axes[0].grid(axis="y", alpha=0.2)

    # Panel (b): all rows retain the same original checkpoint channel order.
    image = axes[1].imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].set_xlabel("Embedding channel (original checkpoint order)")
    axes[1].set_title("(b) Signed channel-aligned gate coefficients")
    colorbar = figure.colorbar(image, ax=axes[1], fraction=0.035, pad=0.02)
    colorbar.set_label(r"$g_r^{(l)}$")

    for suffix in ("png", "pdf"):
        target = path_stem.with_suffix(f".{suffix}")
        figure.savefig(target, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.near_zero < 0.0:
        raise ValueError("--near-zero must be non-negative.")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72.")
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing HAGR checkpoint: {model_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scale_names = [item.strip() for item in args.scale_names.split(",") if item.strip()]
    if len(scale_names) != len(set(scale_names)):
        raise ValueError(f"Scale names must be unique: {scale_names}")

    api_model, detector, head, router = resolve_router(model_path)
    records = collect_gates(detector, router, scale_names, args.near_zero)
    report = {
        "kind": "bridge_hagr_gate_export",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "ultralytics_version": ultralytics_version,
        "torch_version": torch.__version__,
        "router": {
            "class": type(router).__name__,
            "router_version": int(getattr(router, "router_version", 0)),
            "category_prior": float(getattr(router, "category_prior", 0.0)),
            "inheritance_max": float(getattr(router, "inheritance_max", math.nan)),
            "stop_gradient": bool(getattr(router, "stop_gradient", True)),
            "num_scales": int(getattr(router, "num_scales", 0)),
            "head_class": type(head).__name__,
        },
        "definition": {
            "coefficient": "g = max_gain * tanh(logit)",
            "heatmap_channel_order": "original checkpoint order, shared unchanged across all rows",
            "near_zero_absolute_threshold": args.near_zero,
            "interpretation_limit": (
                "Channels are learned embedding dimensions, not named semantic concepts. Static gate coefficients "
                "show selective routing but not input-dependent inherited activation magnitude."
            ),
        },
        "gates": records,
    }

    json_path = output_dir / "hagr_gate_coefficients.json"
    csv_path = output_dir / "hagr_gate_coefficients.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, records)
    if args.plots:
        plot_figure(output_dir / "figure4_hagr_gates", records, args.dpi)

    concise = {
        f"{record['role']}/{record['scale']}": record["statistics"] for record in records
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    print(f"JSON source data: {json_path}")
    print(f"CSV source data:  {csv_path}")
    if args.plots:
        print(f"Figure PNG/PDF: {output_dir / 'figure4_hagr_gates.[png|pdf]'}")

    # Keep the loaded model alive until every tensor has been materialized and written.
    del router, head, detector, api_model


if __name__ == "__main__":
    main()
