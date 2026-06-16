#!/usr/bin/env python3
"""
plot_results.py
===============
Render all benchmark figures for the conversational-commerce LLM study from the
per-run ``metrics_*.json`` files under ``Results/``.

It produces, in ``figures/`` (both vector PDF for LaTeX and 300-dpi PNG):

  1. f1_by_regime          - Operation-F1 grouped bars (model x prompting regime)
  2. quality_cost_tradeoff - energy/command (log x) vs Operation-F1 Pareto plot
  3. latency_f1_tradeoff   - mean latency (log x) vs Operation-F1 Pareto plot
  4. quality_metrics_grid  - grouped bars for every quality metric
  5. cost_metrics_grid     - grouped bars for every cost metric
  6. tokens_by_regime      - prompt vs output token counts (the decode-cost driver)

Only matplotlib + numpy are required:

    python3 -m venv .venv && source .venv/bin/activate
    pip install matplotlib numpy
    python plot_results.py                       # writes ./figures
    python plot_results.py --results Results --outdir figures --dpi 300

Energy per command is a derived quantity, E = mean_power(W) x mean_latency(s) [J].
VRAM is reported in GB (decimal, MB/1000) and reflects the serving runtime's
reservation, not the minimal model footprint.
"""

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render straight to files
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Preferred left-to-right model order (roughly descending by parameter count).
# Any model directory not listed here is appended in alphabetical order.
PREFERRED_ORDER = [
    "gemma4_31b", "gpt_oss_20b", "llama3_1_8b",
    "mistral_7b", "granite4_3b", "llama3_2_3b",
]

# Models excluded from the paper. Phi3 3.8B is omitted: under the few-shot regime
# it emits non-JSON prose (JSON validity ~0.002), so its operation scores are not
# a faithful measure of extraction quality. The raw run is retained under
# Results/phi3_3_8b/ but is neither plotted nor tabulated.
EXCLUDE_MODELS = {"phi3_3_8b"}

DISPLAY = {
    "gemma4_31b": "Gemma4 31B", "gpt_oss_20b": "GPT-OSS 20B",
    "llama3_1_8b": "Llama3.1 8B", "mistral_7b": "Mistral 7B",
    "granite4_3b": "Granite4 3B", "llama3_2_3b": "Llama3.2 3B",
}

REGIMES = ["minimal", "extended", "few_shot"]
REGIME_LABELS = {"minimal": "Minimal", "extended": "Extended", "few_shot": "Few-shot"}
REGIME_COLORS = {"minimal": "#4C72B0", "extended": "#DD8452", "few_shot": "#1F9E89"}
REGIME_MARKERS = {"minimal": "o", "extended": "s", "few_shot": "*"}

# --------------------------------------------------------------------------- #
# Metric accessors  (each takes one loaded metrics dict -> float)
# --------------------------------------------------------------------------- #

def energy_j(rec):
    """Estimated energy per command (J) = mean GPU power (W) x mean latency (s)."""
    perf = rec["performance"]
    return perf["resources"]["gpu_power_watts"]["mean"] * perf["latency_seconds"]["mean"]

QUALITY_METRICS = [
    ("JSON validity",   lambda r: r["quality"]["json_validity_rate"]),
    ("Schema validity", lambda r: r["quality"]["schema_validity_rate"]),
    ("Operation F1",    lambda r: r["quality"]["operation_f1"]),
    ("Exact match",     lambda r: r["quality"]["full_sample_exact_match_ordered_rate"]),
    ("Action acc.",     lambda r: r["quality"]["action_accuracy"]),
    ("Product acc.",    lambda r: r["quality"]["product_accuracy"]),
    ("Quantity acc.",   lambda r: r["quality"]["quantity_accuracy"]),
]

COST_METRICS = [
    ("Mean latency (s)",    lambda r: r["performance"]["latency_seconds"]["mean"]),
    ("P95 latency (s)",     lambda r: r["performance"]["latency_seconds"]["p95"]),
    ("Throughput (cmd/s)",  lambda r: r["performance"]["throughput_samples_per_second"]),
    ("GPU power (W)",       lambda r: r["performance"]["resources"]["gpu_power_watts"]["mean"]),
    ("Energy/command (J)",  energy_j),
    ("VRAM (GB)",           lambda r: r["performance"]["resources"]["vram_used_mb"]["mean"] / 1000.0),
    ("Prompt tokens",       lambda r: r["performance"]["tokens"]["prompt_eval_count_avg"]),
    ("Output tokens",       lambda r: r["performance"]["tokens"]["eval_count_avg"]),
]

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_data(results_dir):
    """Return (models, data) where data[model][regime] = parsed metrics dict."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir.resolve()}")

    discovered = []
    data = {}
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or child.name in EXCLUDE_MODELS:
            continue
        regime_files = {r: child / f"metrics_{r}.json" for r in REGIMES}
        if not any(p.exists() for p in regime_files.values()):
            continue
        discovered.append(child.name)
        data[child.name] = {}
        for regime, path in regime_files.items():
            if path.exists():
                with open(path) as fh:
                    data[child.name][regime] = json.load(fh)

    if not discovered:
        raise SystemExit(f"No metrics_*.json files found under {results_dir.resolve()}")

    models = ([m for m in PREFERRED_ORDER if m in discovered]
              + [m for m in discovered if m not in PREFERRED_ORDER])
    return models, data


def label_for(model, data):
    if model in DISPLAY:
        return DISPLAY[model]
    # fall back to the model string recorded in the JSON
    for regime in REGIMES:
        if regime in data[model]:
            return data[model][regime].get("model", model)
    return model

# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #

def setup_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": 110,
    })


def grouped_bar(ax, models, data, value_fn, ylabel, annotate_zero=True):
    """Draw one grouped bar chart (one group per model, one bar per regime)."""
    x = np.arange(len(models))
    width = 0.27
    ymax = 0.0
    for i, regime in enumerate(REGIMES):
        vals = [value_fn(data[m][regime]) if regime in data[m] else np.nan
                for m in models]
        ymax = max(ymax, max(v for v in vals if not math.isnan(v)))
        offs = x + (i - 1) * width
        ax.bar(offs, vals, width, label=REGIME_LABELS[regime],
               color=REGIME_COLORS[regime], edgecolor="white", linewidth=0.4)
        if annotate_zero:
            for xi, v in zip(offs, vals):
                if not math.isnan(v) and abs(v) < 1e-9:
                    ax.text(xi, 0.0, "0.00", ha="center", va="bottom",
                            rotation=90, fontsize=6, color=REGIME_COLORS[regime])
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(m, data) for m in models], rotation=30, ha="right")
    ax.set_axisbelow(True)
    ax.margins(x=0.02)
    # headroom so labels/annotations are not clipped
    ax.set_ylim(top=ymax * 1.12 if ymax > 0 else 1.0)


def save(fig, outdir, name, dpi):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outdir / name}.pdf / .png")

# --------------------------------------------------------------------------- #
# Individual figures
# --------------------------------------------------------------------------- #

def fig_f1_by_regime(models, data, outdir, dpi):
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    grouped_bar(ax, models, data,
                lambda r: r["quality"]["operation_f1"], "Operation F1")
    ax.set_ylim(0, 1.08)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    save(fig, outdir, "f1_by_regime", dpi)


def _tradeoff(models, data, x_fn, xlabel, name, outdir, dpi, logx=True):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    f1 = lambda r: r["quality"]["operation_f1"]

    for m in models:
        regs = [r for r in REGIMES if r in data[m]]
        xs = [x_fn(data[m][r]) for r in regs]
        ys = [f1(data[m][r]) for r in regs]
        # trajectory line (min -> ext -> few-shot); dashed red for a collapsing model
        collapse = any(f1(data[m][r]) < 0.05 for r in regs)
        ax.plot(xs, ys, lw=1.0, zorder=1,
                color="#C44E52" if collapse else "0.6",
                ls="--" if collapse else "-", alpha=0.8)

    # regime-coloured marker layers (so the legend is by prompt)
    for regime in REGIMES:
        xs = [x_fn(data[m][regime]) for m in models if regime in data[m]]
        ys = [f1(data[m][regime]) for m in models if regime in data[m]]
        ax.scatter(xs, ys, s=70 if regime == "few_shot" else 45,
                   marker=REGIME_MARKERS[regime], color=REGIME_COLORS[regime],
                   edgecolor="white", linewidth=0.5, zorder=3,
                   label=REGIME_LABELS[regime])

    # model labels near the few-shot point (tweak offsets to taste)
    label_off = {
        "gemma4_31b": (0, 10), "gpt_oss_20b": (8, 6), "llama3_1_8b": (8, 8),
        "mistral_7b": (8, -14), "granite4_3b": (-8, 12), "llama3_2_3b": (0, -16),
    }
    for m in models:
        regs = [r for r in REGIMES if r in data[m]]
        anchor = "few_shot" if "few_shot" in regs else regs[-1]
        dx, dy = label_off.get(m, (8, 6))
        ax.annotate(label_for(m, data),
                    (x_fn(data[m][anchor]), f1(data[m][anchor])),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, ha="center")

    if logx:
        ax.set_xscale("log")
        all_x = [x_fn(data[m][r]) for m in models for r in REGIMES if r in data[m]]
        ax.set_xlim(min(all_x) * 0.6, max(all_x) * 1.7)  # padding so edge labels don't clip
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Operation F1")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", title="Prompt")
    save(fig, outdir, name, dpi)


def fig_quality_cost(models, data, outdir, dpi):
    _tradeoff(models, data, energy_j,
              "Estimated energy per command (J, log scale)",
              "quality_cost_tradeoff", outdir, dpi)


def fig_latency_f1(models, data, outdir, dpi):
    _tradeoff(models, data, lambda r: r["performance"]["latency_seconds"]["mean"],
              "Mean latency per command (s, log scale)",
              "latency_f1_tradeoff", outdir, dpi)


def _metric_grid(models, data, metrics, name, outdir, dpi, ncols=4):
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.1))
    axes = np.atleast_1d(axes).flatten()
    for ax, (label, fn) in zip(axes, metrics):
        grouped_bar(ax, models, data, fn, label)
        ax.set_title(label)
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, outdir, name, dpi)


def fig_quality_grid(models, data, outdir, dpi):
    _metric_grid(models, data, QUALITY_METRICS, "quality_metrics_grid", outdir, dpi)


def fig_cost_grid(models, data, outdir, dpi):
    _metric_grid(models, data, COST_METRICS, "cost_metrics_grid", outdir, dpi)


def fig_tokens(models, data, outdir, dpi):
    """Prompt vs output tokens per regime - the driver of the decode-bound cost."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    grouped_bar(ax1, models, data,
                lambda r: r["performance"]["tokens"]["prompt_eval_count_avg"],
                "Mean prompt tokens")
    ax1.set_title("Prompt (input) tokens")
    grouped_bar(ax2, models, data,
                lambda r: r["performance"]["tokens"]["eval_count_avg"],
                "Mean output tokens")
    ax2.set_title("Output (decoded) tokens")
    ax1.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    save(fig, outdir, "tokens_by_regime", dpi)

# --------------------------------------------------------------------------- #
# Optional: dump the tidy table that backs the figures (handy for the paper)
# --------------------------------------------------------------------------- #

def dump_table(models, data, outdir):
    rows = []
    header = (["model", "regime"]
              + [lbl for lbl, _ in QUALITY_METRICS]
              + [lbl for lbl, _ in COST_METRICS])
    rows.append(",".join(header))
    for m in models:
        for regime in REGIMES:
            if regime not in data[m]:
                continue
            rec = data[m][regime]
            vals = ([label_for(m, data), regime]
                    + [f"{fn(rec):.4f}" for _, fn in QUALITY_METRICS]
                    + [f"{fn(rec):.4f}" for _, fn in COST_METRICS])
            rows.append(",".join(vals))
    out = Path(outdir) / "summary_table.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n")
    print(f"  wrote {out}")

# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="Results", help="directory of per-model results")
    ap.add_argument("--outdir", default="figures", help="where to write the figures")
    ap.add_argument("--dpi", type=int, default=300, help="raster (PNG) resolution")
    ap.add_argument("--no-table", action="store_true", help="skip summary_table.csv")
    args = ap.parse_args()

    setup_style()
    models, data = load_data(args.results)
    print(f"Loaded {len(models)} models: {', '.join(label_for(m, data) for m in models)}")
    print(f"Writing figures to ./{args.outdir}/")

    fig_f1_by_regime(models, data, args.outdir, args.dpi)
    fig_quality_cost(models, data, args.outdir, args.dpi)
    fig_latency_f1(models, data, args.outdir, args.dpi)
    fig_quality_grid(models, data, args.outdir, args.dpi)
    fig_cost_grid(models, data, args.outdir, args.dpi)
    fig_tokens(models, data, args.outdir, args.dpi)
    if not args.no_table:
        dump_table(models, data, args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()
