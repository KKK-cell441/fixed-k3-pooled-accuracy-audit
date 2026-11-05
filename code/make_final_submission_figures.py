from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "manuscript_draft" / "figures"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

BLUE = "#1f5a8a"
ORANGE = "#c65f1a"
GREEN = "#2f6b4f"
RED = "#a23b3b"
GRAY = "#6f6f6f"
LIGHT_GRAY = "#d8d8d8"


def load_json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6, zorder=0)
    ax.tick_params(length=2.5, width=0.7)
    ax.axhline(0, color="#444444", linewidth=0.8, zorder=1)


def same_source_figure() -> None:
    fixed = load_json("hust_dual_validation_results.json")
    multi = load_json("hust_noise_realization_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    panels = [
        ("speed", "Held-out motor speed (Hz)", "Speed-wise"),
        ("bearing", "Held-out physical bearing", "Bearing-wise"),
    ]
    for ax, (scheme, xlabel, title) in zip(axes, panels):
        fixed_rows = fixed[scheme]["units"]
        multi_rows = multi["summaries"][scheme]["unit_effects"]
        x = np.arange(len(fixed_rows))
        fixed_vals = np.asarray([row["delta"] for row in fixed_rows]) * 100
        multi_vals = np.asarray([row["delta"] for row in multi_rows]) * 100
        labels = [row["unit"] for row in fixed_rows]

        ax.scatter(
            x - 0.12,
            fixed_vals,
            s=18,
            color=BLUE,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
            label="Fixed realization",
        )
        ax.scatter(
            x + 0.12,
            multi_vals,
            s=16,
            color=ORANGE,
            marker="s",
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
            label="15-aggregate",
        )
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("K3 minus raw (pp)")
        ax.set_title(title, loc="left", fontweight="bold")
        style_axis(ax)

    axes[0].set_ylim(-3, 23)
    axes[1].set_ylim(-22, 17)
    axes[0].legend(frameon=False, loc="upper left", ncols=2)
    save(fig, "final_same_source_split")


def matched_augmentation_figure() -> None:
    split = load_json("hust_augmented_split_results.json")
    signal = load_json("hust_signal_control_results.json")
    mismatch = load_json("hust_noise_mismatch_results.json")

    labels = [
        "Raw\nclean-clean",
        "K3\nclean-clean",
        "Raw\nclean-0 dB",
        "K3\nclean-0 dB",
        "Raw\nmatched aug.",
        "Band-pass",
        "Wavelet",
    ]
    keys = [
        "raw_clean_clean",
        "k3_clean_clean",
        "raw_clean_noisy",
        "raw_clean_noisy_k3",
        "raw_aug_noisy",
        "bandpass_clean_noisy",
        "wavelet_clean_noisy",
    ]
    values = {
        row["representation"]: row["pooled_balanced_accuracy"] * 100
        for row in signal["summary"]
    }
    val = np.asarray([values[key] for key in keys])
    colors = [GRAY, BLUE, GRAY, BLUE, GREEN, ORANGE, ORANGE]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    ax = axes[0]
    ax.bar(np.arange(len(val)), val, color=colors, width=0.72, zorder=2)
    ax.set_xticks(np.arange(len(val)), labels, rotation=35, ha="right")
    ax.set_ylabel("Pooled balanced accuracy (%)")
    ax.set_ylim(0, 85)
    ax.set_title("(A) Signal controls", loc="left", fontweight="bold")
    style_axis(ax)

    ax = axes[1]
    specs = [
        (
            "Speed, matched",
            split["summaries"]["speed"]["mean_unit_delta"] * 100,
            np.asarray(split["summaries"]["speed"]["unit_interval"]) * 100,
            BLUE,
            "o",
        ),
        (
            "Bearing, matched",
            split["summaries"]["bearing"]["mean_unit_delta"] * 100,
            np.asarray(split["summaries"]["bearing"]["unit_interval"]) * 100,
            GREEN,
            "s",
        ),
        (
            "Speed, -3 dB",
            mismatch["summaries"]["speed"]["-3.0"]["mean_unit_delta"] * 100,
            np.asarray(mismatch["summaries"]["speed"]["-3.0"]["unit_interval"]) * 100,
            ORANGE,
            "D",
        ),
        (
            "Speed, +4 dB",
            mismatch["summaries"]["speed"]["4.0"]["mean_unit_delta"] * 100,
            np.asarray(mismatch["summaries"]["speed"]["4.0"]["unit_interval"]) * 100,
            RED,
            "^",
        ),
    ]
    for y, (label, mean, interval, color, marker) in enumerate(specs):
        ax.errorbar(
            mean,
            y,
            xerr=np.asarray([[mean - interval[0]], [interval[1] - mean]]),
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=2.5,
            markersize=4.5,
            zorder=3,
        )
        ax.text(interval[1] + 0.6, y, label, va="center", fontsize=7)
    ax.set_yticks([])
    ax.set_xlim(-17, 12)
    ax.set_xlabel("K3 minus raw (pp)")
    ax.set_title("(B) Matched and mismatch effects", loc="left", fontweight="bold")
    style_axis(ax)
    save(fig, "final_matched_augmentation")


def external_forest_figure() -> None:
    rows = []
    with (ROOT / "qualification_demo.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    rows.append(
        {
            "dataset": "PHM2012 (healthy/degraded)",
            "delta": "0.2450980392156863",
            "ci_low": "0.19199346405228756",
            "ci_high": "0.2973856209150327",
            "status": "Not qualified",
        }
    )
    labels = [row["dataset"] for row in rows]
    means = np.asarray([float(row["delta"]) * 100 for row in rows])
    lows = np.asarray([float(row["ci_low"]) * 100 for row in rows])
    highs = np.asarray([float(row["ci_high"]) * 100 for row in rows])
    status = [row["status"] for row in rows]
    color_map = {"Insufficient": BLUE, "Not qualified": RED}

    fig, ax = plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)
    y = np.arange(len(rows))
    for yi, mean, low, high, state in zip(y, means, lows, highs, status):
        color = color_map[state]
        ax.errorbar(
            mean,
            yi,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o",
            markersize=4.6,
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=2.8,
            zorder=3,
        )
        ax.text(31.5, yi, state, va="center", fontsize=7, color=color)
    ax.set_yticks(y, labels)
    ax.set_xlim(-58, 41)
    ax.set_xlabel("K3 minus raw (pp)")
    style_axis(ax)
    ax.grid(axis="y", visible=False)
    save(fig, "final_external_forest")


def nested_k_figure() -> None:
    data = load_json("hust_independent_k_results.json")
    freq = data["selected_k_frequency"]
    ks = ["2", "3", "4", "5"]
    counts = [freq[k] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), constrained_layout=True)
    axes[0].bar(ks, counts, color=[GRAY, BLUE, ORANGE, GREEN], width=0.65)
    axes[0].set_xlabel("Selected mode count K")
    axes[0].set_ylabel("Outer selections")
    axes[0].set_ylim(0, 14)
    axes[0].set_title("(A) Selection frequency", loc="left", fontweight="bold")
    style_axis(axes[0])

    vals = [
        data["fixed_k3_mean_delta"] * 100,
        data["selected_k_mean_delta"] * 100,
    ]
    axes[1].bar(["Fixed K3", "Nested K"], vals, color=[BLUE, ORANGE], width=0.6)
    axes[1].set_ylabel("Mean gain over raw (pp)")
    axes[1].set_ylim(0, 12)
    axes[1].set_title("(B) Held-out effect", loc="left", fontweight="bold")
    style_axis(axes[1])
    save(fig, "final_nested_k")


def operating_figure() -> None:
    rows = []
    with (ROOT / "qualification_operating_characteristics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    scenarios = [
        "borderline_benefit",
        "borderline_harm",
        "uncertain",
        "qualified",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    width = 0.24
    x = np.arange(len(scenarios))

    for idx, n_units in enumerate([10, 20, 40]):
        vals_q = []
        vals_i = []
        for scenario in scenarios:
            candidates = [
                row
                for row in rows
                if row["scenario"] == scenario and int(row["n_units"]) == n_units
            ]
            if candidates:
                vals_q.append(float(candidates[0]["qualified_rate"]) * 100)
                vals_i.append(float(candidates[0]["insufficient_rate"]) * 100)
            else:
                vals_q.append(np.nan)
                vals_i.append(np.nan)
        offset = (idx - 1) * width
        axes[0].bar(
            x + offset,
            vals_q,
            width,
            label=f"n={n_units}",
            color=[BLUE, ORANGE, GREEN][idx],
        )
        axes[1].bar(
            x + offset,
            vals_i,
            width,
            label=f"n={n_units}",
            color=[BLUE, ORANGE, GREEN][idx],
        )

    for ax, title in zip(
        axes, ["(A) Qualified rate", "(B) Insufficient rate"]
    ):
        ax.set_xticks(x, ["Border\nbenefit", "Border\nharm", "Uncertain", "Qualified"])
        ax.set_ylim(0, 105)
        ax.set_ylabel("Rate (%)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.legend(frameon=False, ncols=3, loc="upper left")
        style_axis(ax)
    save(fig, "final_operating_characteristics")


def main() -> None:
    same_source_figure()
    matched_augmentation_figure()
    external_forest_figure()
    nested_k_figure()
    operating_figure()


if __name__ == "__main__":
    main()
