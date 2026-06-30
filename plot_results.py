#!/usr/bin/env python3
"""
plot_results.py
===============
Render all benchmark figures for the conversational-commerce LLM study from the
per-run ``metrics_*.json`` files under ``responses/``.

It produces, in ``figures/`` (both vector PDF for LaTeX and 300-dpi PNG):

    1. f1_by_regime              - Operation-F1 grouped bars (model x prompting regime)
    2. quality_cost_tradeoff     - energy/command (log x) vs Operation-F1 Pareto plot
    3. latency_f1_tradeoff       - mean latency (log x) vs Operation-F1 Pareto plot
    4. quality_*_by_regime       - one grouped bar chart per quality metric
    5. cost_*_by_regime          - one grouped bar chart per cost metric
    6. prompt_tokens_by_regime   - prompt token counts
    7. output_tokens_by_regime   - output token counts

Only matplotlib + numpy are required:

    python3 -m venv .venv && source .venv/bin/activate
    pip install matplotlib numpy
    python plot_results.py                       # writes ./figures
    python plot_results.py --results responses --outdir figures --dpi 300

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
# responses/phi3_3_8b/ but is neither plotted nor tabulated.
EXCLUDE_MODELS = {"phi3_3_8b"}

DISPLAY = {
    "gemma4_31b": "Gemma4 31B", "gpt_oss_20b": "GPT-OSS 20B",
    "llama3_1_8b": "Llama3.1 8B", "mistral_7b": "Mistral 7B",
    "granite4_3b": "Granite4 3B", "llama3_2_3b": "Llama3.2 3B",
}

REGIMES = ["minimal", "extended", "few_shot"]
REGIME_LABELS = {"minimal": "Minimal", "extended": "Extended", "few_shot": "Few-shot"}
REGIME_COLORS = {"minimal": "#4AA8EF", "extended": "#FFA53C", "few_shot": "#2FD3B4"}
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

# Metrics promoted to large standalone charts in the paper (rendered like
# f1_by_regime: big figure with on-bar black value labels) instead of compact
# grid panels. The token charts are handled separately in fig_tokens().
HERO_METRICS = {"JSON validity", "GPU power (W)"}

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
        model_data = {}
        for regime, path in regime_files.items():
            if path.exists():
                with open(path) as fh:
                    model_data[regime] = json.load(fh)

        # Skip partial runs so grouped bars stay consistent across models.
        if not all(r in model_data for r in REGIMES):
            print(f"  skipping {child.name}: missing one or more metrics_<regime>.json files")
            continue

        discovered.append(child.name)
        data[child.name] = model_data

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
            json_label = data[model][regime].get("model")
            return json_label if json_label else model
    return model


def metric_slug(label):
    slug = []
    for ch in label.lower():
        if ch.isalnum():
            slug.append(ch)
        else:
            slug.append("_")
    return "".join(slug).strip("_")

# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #

def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 26,
        "axes.titlesize": 42,
        "axes.labelsize": 38,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 28,
        "legend.title_fontsize": 28,
        "figure.dpi": 110,
    })


def format_bar_value(v):
    """Compact numeric format for bar labels across mixed-scale metrics."""
    av = abs(v)
    if av >= 100:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def grouped_bar(ax, models, data, value_fn, ylabel, annotate_values=True, y_min=0.0,
                value_fontsize=32, label_above=False):
    """Draw one grouped bar chart (one group per model, one bar per regime)."""
    x = np.arange(len(models))
    width = 0.30
    ymax = 0.0
    bar_groups = []
    for i, regime in enumerate(REGIMES):
        vals = [value_fn(data[m][regime]) if regime in data[m] else np.nan
                for m in models]
        ymax = max(ymax, max(v for v in vals if not math.isnan(v)))
        offs = x + (i - 1) * width
        bars = ax.bar(
            offs,
            vals,
            width,
            label=REGIME_LABELS[regime],
            color=REGIME_COLORS[regime],
            edgecolor="white",
            linewidth=0.4,
        )
        bar_groups.append((bars, vals))
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(m, data) for m in models], rotation=0, ha="center")
    ax.set_axisbelow(True)
    ax.margins(x=0.02)
    # headroom so labels/annotations are not clipped
    ax.set_ylim(bottom=y_min, top=ymax * 1.08 if ymax > 0 else 1.0)

    if annotate_values:
        yr = (ymax - y_min) if ymax > y_min else 1.0
        common_label_y = y_min + yr * 0.48
        for bars, vals in bar_groups:
            for bar, v in zip(bars, vals):
                if math.isnan(v):
                    continue
                if label_above:
                    # Sit just above each bar on the white background: the most
                    # legible placement, and the bars' own heights stagger the
                    # labels so near-equal values do not collide.
                    label_y, va = v + yr * 0.012, "bottom"
                elif v > common_label_y + 0.01:
                    # one common height when the bar is tall enough,
                    label_y, va = common_label_y, "center"
                else:
                    # otherwise centered in the (short) bar.
                    label_y = y_min + (v - y_min) / 2.0 if v >= y_min else v / 2.0
                    va = "center"
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    label_y,
                    format_bar_value(v),
                    ha="center",
                    va=va,
                    fontsize=value_fontsize,
                    color="black",
                    fontweight="bold",
                )


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
    fig, ax = plt.subplots(figsize=(20, 12))
    grouped_bar(ax, models, data,
                lambda r: r["quality"]["operation_f1"], "Operation F1", y_min=0.4,
                value_fontsize=26, label_above=True)
    ax.set_ylim(0.4, 1.10)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, outdir, "f1_by_regime", dpi)


def fig_metric_hero(models, data, label, value_fn, name, outdir, dpi, y_min=0.0,
                    value_fontsize=22, y_max=None):
    """Large single-metric chart in the prominent f1_by_regime style: big
    figure with on-bar black value labels (via the global style). Used for the
    standalone charts kept in the paper (GPU power, JSON validity, tokens).

    value_fontsize defaults to 22 (smaller than f1_by_regime's 32): these metrics
    often have near-equal bars within a group (e.g. JSON validity ~1.0), so larger
    labels at the common height would abut. The token charts override it upward
    since each is shown on its own at full text width."""
    fig, ax = plt.subplots(figsize=(20, 12))
    grouped_bar(ax, models, data, value_fn, label, annotate_values=True, y_min=y_min,
                value_fontsize=value_fontsize)
    if y_max is not None:
        ax.set_ylim(y_min, y_max)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, outdir, name, dpi)


def _tradeoff(models, data, x_fn, xlabel, name, outdir, dpi, logx=True, y_min=-0.03):
    fig, ax = plt.subplots(figsize=(16, 11))
    f1 = lambda r: r["quality"]["operation_f1"]

    for m in models:
        regs = [r for r in REGIMES if r in data[m]]
        xs = [x_fn(data[m][r]) for r in regs]
        ys = [f1(data[m][r]) for r in regs]
        # trajectory line (min -> ext -> few-shot); dashed red for a collapsing model
        collapse = any(f1(data[m][r]) < 0.05 for r in regs)
        ax.plot(xs, ys, lw=1.8, zorder=1,
                color="#C44E52" if collapse else "0.6",
                ls="--" if collapse else "-", alpha=0.8)

    # regime-coloured marker layers (so the legend is by prompt)
    for regime in REGIMES:
        xs = [x_fn(data[m][regime]) for m in models if regime in data[m]]
        ys = [f1(data[m][regime]) for m in models if regime in data[m]]
        ax.scatter(xs, ys, s=320 if regime == "few_shot" else 200,
                   marker=REGIME_MARKERS[regime], color=REGIME_COLORS[regime],
                   edgecolor="white", linewidth=0.8, zorder=3,
                   label=REGIME_LABELS[regime])

    # model labels near the few-shot point (tweak offsets to taste)
    # Offsets (in points) placing each model name in clear space above/beside its
    # few-shot point, away from the grey trajectory lines that descend from it.
    label_off = {
        "gemma4_31b": (0, 16), "gpt_oss_20b": (0, 16), "llama3_1_8b": (90, 6),
        "mistral_7b": (0, 26), "granite4_3b": (-40, 12), "llama3_2_3b": (-26, -14),
    }
    for m in models:
        regs = [r for r in REGIMES if r in data[m]]
        anchor = "few_shot" if "few_shot" in regs else regs[-1]
        dx, dy = label_off.get(m, (8, 6))
        ax.annotate(label_for(m, data),
                    (x_fn(data[m][anchor]), f1(data[m][anchor])),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=22, ha="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))

    if logx:
        ax.set_xscale("log")
        all_x = [x_fn(data[m][r]) for m in models for r in REGIMES if r in data[m]]
        ax.set_xlim(min(all_x) * 0.6, max(all_x) * 1.7)  # padding so edge labels don't clip
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Operation F1")
    ax.set_ylim(y_min, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="Prompt")
    save(fig, outdir, name, dpi)


def fig_quality_cost(models, data, outdir, dpi):
    # Zoom the F1 axis to [0.4, 1.05]: with Phi3 excluded every configuration
    # scores >= 0.5, so starting near 0 wastes the lower half of the panel.
    _tradeoff(models, data, energy_j,
              "Estimated energy per command (J, log scale)",
              "quality_cost_tradeoff", outdir, dpi, y_min=0.4)


def fig_latency_f1(models, data, outdir, dpi):
    _tradeoff(models, data, lambda r: r["performance"]["latency_seconds"]["mean"],
              "Mean latency per command (s, log scale)",
              "latency_f1_tradeoff", outdir, dpi, y_min=0.4)


def fig_metric_by_regime(models, data, label, value_fn, name, outdir, dpi):
    # These per-metric charts sit at half text-width inside the paper's figure
    # grids, so on-bar value labels are dropped (the exact numbers live in
    # Tables 1-2) and the tick fonts are reduced so the six model names fit
    # without colliding. The large single charts (e.g. f1_by_regime) keep the
    # big black value labels via the global style.
    with plt.rc_context({
        "axes.labelsize": 32,
        "xtick.labelsize": 20,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
    }):
        fig, ax = plt.subplots(figsize=(18, 8))
        grouped_bar(ax, models, data, value_fn, label, annotate_values=False)
        if label.strip().lower() != ax.get_ylabel().strip().lower():
            ax.set_title(label)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        save(fig, outdir, name, dpi)


def fig_quality_metrics(models, data, outdir, dpi):
    for label, fn in QUALITY_METRICS:
        name = f"quality_{metric_slug(label)}_by_regime"
        if label in HERO_METRICS:
            # JSON validity sits at 0.89-1.0, so zoom its axis to [0.4, 1.0]
            # rather than starting at 0 where the bars look near-identical.
            kw = {"y_min": 0.4, "y_max": 1.0} if label == "JSON validity" else {}
            fig_metric_hero(models, data, label, fn, name, outdir, dpi, **kw)
        else:
            fig_metric_by_regime(models, data, label, fn, name, outdir, dpi)


def fig_cost_metrics(models, data, outdir, dpi):
    for label, fn in COST_METRICS:
        name = f"cost_{metric_slug(label)}_by_regime"
        if label in HERO_METRICS:
            # GPU power only spans ~115-160 W, so start its axis at 100 to make
            # the (small) cross-model differences legible rather than near-flat.
            y_min = 100.0 if label == "GPU power (W)" else 0.0
            fig_metric_hero(models, data, label, fn, name, outdir, dpi, y_min=y_min)
        else:
            fig_metric_by_regime(models, data, label, fn, name, outdir, dpi)


def fig_tokens(models, data, outdir, dpi):
    """Prompt/output token plots written as separate files, in the large style
    with on-bar value labels. Each is shown on its own at full text width in the
    paper, so the labels use a slightly larger font than the other hero charts."""
    fig_metric_hero(
        models,
        data,
        "Mean prompt tokens",
        lambda r: r["performance"]["tokens"]["prompt_eval_count_avg"],
        "prompt_tokens_by_regime",
        outdir,
        dpi,
        value_fontsize=26,
    )
    fig_metric_hero(
        models,
        data,
        "Mean output tokens",
        lambda r: r["performance"]["tokens"]["eval_count_avg"],
        "output_tokens_by_regime",
        outdir,
        dpi,
        value_fontsize=26,
    )

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
    ap.add_argument("--results", default="responses", help="directory of per-model results")
    ap.add_argument("--outdir", default="figures", help="where to write the figures")
    ap.add_argument("--dpi", type=int, default=800, help="raster (PNG) resolution")
    ap.add_argument("--no-table", action="store_true", help="skip summary_table.csv")
    args = ap.parse_args()

    setup_style()
    models, data = load_data(args.results)
    print(f"Loaded {len(models)} models: {', '.join(label_for(m, data) for m in models)}")
    print(f"Writing figures to ./{args.outdir}/")

    fig_f1_by_regime(models, data, args.outdir, args.dpi)
    fig_quality_cost(models, data, args.outdir, args.dpi)
    fig_latency_f1(models, data, args.outdir, args.dpi)
    fig_quality_metrics(models, data, args.outdir, args.dpi)
    fig_cost_metrics(models, data, args.outdir, args.dpi)
    fig_tokens(models, data, args.outdir, args.dpi)
    if not args.no_table:
        dump_table(models, data, args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()
