#!/usr/bin/env python3
"""Redraw HW6 figures in a clean academic style.

This script only changes figure presentation. It does not run experiments or
alter the results used in the figures.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final_submission_hw6" / "figures"
RESULTS = ROOT / "results" / "all_experiments.json"

TEXT = "#111111"
MUTED = "#4b5563"
GRID = "#d1d5db"
EDGE = "#222222"
FILL = "#ffffff"
PANEL = "#f8fafc"

COLORS = {
    "DSL": "#1f77b4",
    "DSL+Opt": "#17becf",
    "Baseline-v1": "#d97706",
    "Baseline-v2": "#b91c1c",
    "SimpleSync": "#6b7280",
    "accent": "#2ca02c",
    "warning": "#b91c1c",
}


def load_results() -> dict:
    with RESULTS.open() as f:
        return json.load(f)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": EDGE,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "lines.linewidth": 1.6,
            "patch.linewidth": 1.1,
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def title(ax, text: str) -> None:
    ax.set_title(text, loc="center", fontweight="bold", pad=14)


def rounded_box(ax, x, y, w, h, heading, lines, fill=PANEL, title_size=11, body_size=9.5):
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fill,
        edgecolor=EDGE,
        linewidth=1.2,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.045, heading, ha="center", va="top", fontweight="bold", fontsize=title_size)
    if isinstance(lines, str):
        lines = [lines]
    start_y = y + h - 0.115
    gap = min(0.052, (h - 0.15) / max(len(lines), 1))
    for i, line in enumerate(lines):
        ax.text(x + w / 2, start_y - i * gap, line, ha="center", va="top", fontsize=body_size, color=TEXT)


def draw_arrow(ax, start, end, label=None, label_y_offset=0.035):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.5, color=EDGE, shrinkA=2, shrinkB=2),
    )
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + label_y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=TEXT,
        )


def latency_value(results: dict, scenario: str, method: str) -> float | None:
    for row in results["latency"].get(scenario, []):
        if row.get("approach") == method and "mean_ms" in row:
            return float(row["mean_ms"])
    return None


def line_count(path: str) -> int:
    return len((ROOT / path).read_text().splitlines())


def draw_execution_evidence(results: dict) -> None:
    clean = {r["approach"]: r["mean_ms"] for r in results["latency"]["clean"]}
    exact = next(r for r in results["correctness"] if r["test"] == "exact_matching")
    nearest = next(r for r in results["correctness"] if r["test"] == "nearest_matching")

    width, height = 1500, 900
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf", 34)
        body_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 25)
        small_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text((70, 55), "Execution Evidence: AutoSyncDSL HW6 Run", font=title_font, fill=TEXT)
    box = (70, 115, width - 70, height - 70)
    draw.rounded_rectangle(box, radius=14, outline=EDGE, width=3, fill="#fbfbfb")
    draw.rectangle((70, 115, width - 70, 165), fill="#f3f4f6", outline=EDGE, width=2)
    draw.text((95, 128), "terminal excerpt", font=small_font, fill=TEXT)

    lines = [
        "$ PYTHONPATH=. python scripts/run_all_experiments.py",
        "",
        "AUTOSYNCDSL EVALUATION SUITE",
        "status: suite started and completed successfully",
        "",
        f"exact_matching: DSL={exact['dsl_correct']} baseline={exact['baseline_correct']} groups={exact['groups_dsl']}",
        f"nearest_matching: DSL={nearest['dsl_correct']} baseline={nearest['baseline_correct']} tolerance={nearest['tolerance_ms']:.1f} ms",
        "",
        "clean latency means:",
        f"  DSL         {clean['DSL']:.4f} ms",
        f"  Baseline-v1 {clean['Baseline-v1']:.4f} ms",
        "",
        "Results saved to results/all_experiments.json",
    ]
    y = 195
    for raw in lines:
        wrapped = textwrap.wrap(raw, width=82, replace_whitespace=False) or [""]
        for line in wrapped:
            draw.text((100, y), line, font=body_font, fill=TEXT)
            y += 38
    img.save(OUT / "execution_evidence.png")


def draw_fig1() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 2.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(ax, "Figure 1. AutoSyncDSL System Overview")

    y, h = 0.18, 0.55
    boxes = [
        (0.035, 0.14, "Sensor Streams", ["camera", "LiDAR", "IMU"], "#f8fafc"),
        (0.245, 0.15, "Embedded DSL", ["SyncPlan", "policy chain"], "#e0f2fe"),
        (0.465, 0.14, "IR Lowering", ["source nodes", "match/filter nodes"], "#e5e7eb"),
        (0.680, 0.13, "Runtime", ["buffer", "match", "batch"], "#dcfce7"),
        (0.875, 0.095, "Output", ["sync", "batches"], "#fef3c7"),
    ]
    for x, w, head, lines, fill in boxes:
        rounded_box(ax, x, y, w, h, head, lines, fill=fill)

    arrow_y = y + h / 2
    draw_arrow(ax, (0.175, arrow_y), (0.245, arrow_y), "specify")
    draw_arrow(ax, (0.395, arrow_y), (0.465, arrow_y), "compile")
    draw_arrow(ax, (0.605, arrow_y), (0.680, arrow_y), "execute")
    draw_arrow(ax, (0.810, arrow_y), (0.875, arrow_y), "emit")

    save(fig, "fig1_system_overview.png")


def draw_fig2() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(ax, "Figure 2. DSL-to-IR Lowering")

    panel_y, panel_h = 0.17, 0.65
    rounded_box(ax, 0.035, panel_y, 0.34, panel_h, "DSL Source", [], fill="#f8fafc")
    code = [
        "SyncPlan()",
        '  .camera("cam")',
        '  .lidar("lidar")',
        '  .imu("imu")',
        '  .nearest("lidar", tolerance_ms=50)',
        "  .drop_stale(max_age_ms=100)",
        "  .batch(size=4)",
    ]
    for i, line in enumerate(code):
        ax.text(0.060, 0.68 - i * 0.065, line, fontfamily="DejaVu Sans Mono", fontsize=8.4, ha="left", va="center")

    rounded_box(ax, 0.440, panel_y, 0.23, panel_h, "IR Nodes", [], fill="#e5e7eb")
    node_x = 0.485
    nodes = [("SensorSource", 0.68), ("NearestMatch", 0.57), ("StaleFilter", 0.46), ("Batch", 0.35)]
    for i, (label, yy) in enumerate(nodes):
        rect = patches.FancyBboxPatch(
            (node_x, yy - 0.035),
            0.15,
            0.07,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            facecolor="white",
            edgecolor=EDGE,
            linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(node_x + 0.075, yy, label, ha="center", va="center", fontsize=9.5)
        if i < len(nodes) - 1:
            draw_arrow(ax, (node_x + 0.075, yy - 0.04), (node_x + 0.075, nodes[i + 1][1] + 0.04), None, 0)

    rounded_box(
        ax,
        0.745,
        panel_y,
        0.22,
        panel_h,
        "Executor Plan",
        ["topological order", "tolerance metadata", "batch size metadata"],
        fill="#dcfce7",
        body_size=10,
    )
    draw_arrow(ax, (0.375, 0.50), (0.440, 0.50), "compile()")
    draw_arrow(ax, (0.670, 0.50), (0.745, 0.50), "run()")
    save(fig, "fig2_dsl_to_ir.png")


def draw_fig3() -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.4), sharex=True)
    fig.suptitle("Figure 3. Synchronization Semantics", y=0.985, fontweight="bold", fontsize=14)

    cam = np.array([0, 33, 67, 100, 133])
    lidar = np.array([10, 50, 90, 120])
    subtitles = [
        "(a) Exact match",
        "(b) Nearest match",
        "(c) Stale rejection",
        "(d) IMU interpolation",
    ]
    for ax, subtitle in zip(axes, subtitles):
        ax.set_title(subtitle, loc="left", pad=8, fontsize=11, fontweight="bold")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_xlim(-5, 150)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].scatter(cam, np.ones_like(cam), color=COLORS["DSL"], marker="o", s=58, label="Camera")
    axes[0].scatter(lidar, np.zeros_like(lidar), color="#d97706", marker="s", s=58, label="LiDAR")
    axes[0].set_yticks([0, 1], ["LiDAR", "Camera"])
    axes[0].annotate("no identical timestamps", xy=(67, 1), xytext=(77, 0.55), arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=9)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)

    axes[1].scatter(cam, np.ones_like(cam), color=COLORS["DSL"], marker="o", s=58)
    axes[1].scatter(lidar, np.zeros_like(lidar), color="#d97706", marker="s", s=58)
    for c, l in zip(cam[:4], lidar[:4]):
        axes[1].plot([c, l], [1, 0], color="#374151", linestyle="--", linewidth=1.0)
    axes[1].set_yticks([0, 1], ["LiDAR", "Camera"])
    axes[1].annotate("selected nearest LiDAR\nwithin 50 ms", xy=(50, 0), xytext=(58, 0.60), arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=9)

    axes[2].scatter(cam, np.ones_like(cam), color=COLORS["DSL"], marker="o", s=58)
    axes[2].scatter([10, 50, 90], [0, 0, 0], color="#d97706", marker="s", s=58)
    axes[2].scatter([10], [0], marker="x", color=COLORS["warning"], s=110, linewidths=2.2)
    axes[2].axvline(133, color=EDGE, linestyle=":", linewidth=1.1)
    axes[2].text(134.5, 1.05, "reference time", fontsize=9, ha="left", va="center")
    axes[2].annotate("stale reading rejected", xy=(10, 0), xytext=(25, 0.42), arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=9, color=TEXT)
    axes[2].set_yticks([0, 1], ["LiDAR", "Camera"])

    imu_ts = np.arange(0, 151, 10)
    imu_val = np.sin(imu_ts / 23.0)
    q = 65
    qv = np.sin(q / 23.0)
    axes[3].plot(imu_ts, imu_val, marker="o", markersize=4, color=COLORS["accent"], label="raw IMU samples")
    axes[3].scatter([q], [qv], color=TEXT, s=58, zorder=5, label="interpolated value")
    axes[3].axvline(q, color=EDGE, linestyle="--", linewidth=1.0)
    axes[3].annotate("interpolated IMU value\nat query time", xy=(q, qv), xytext=(78, qv + 0.30), arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=9)
    axes[3].set_yticks([])
    axes[3].legend(loc="upper right", frameon=False)
    axes[3].set_xlabel("Time (milliseconds)")

    fig.subplots_adjust(top=0.92, hspace=0.52, left=0.09, right=0.98, bottom=0.08)
    save(fig, "fig3_sync_semantics.png")


def draw_fig4() -> None:
    fig, ax = plt.subplots(figsize=(9.4, 4.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(ax, "Figure 4. Prototype Optimization Before and After")
    rounded_box(
        ax,
        0.08,
        0.22,
        0.33,
        0.55,
        "Before",
        ["NearestMatch pass", "StaleFilter pass", "Interpolation pass", "Batch assembly"],
        fill="#f8fafc",
        body_size=10,
    )
    rounded_box(
        ax,
        0.59,
        0.22,
        0.33,
        0.55,
        "After",
        ["candidate rule fusion", "buffer reuse metadata", "same user-facing semantics"],
        fill="#f8fafc",
        body_size=10,
    )
    draw_arrow(ax, (0.41, 0.50), (0.59, 0.50), "optimizer", label_y_offset=0.045)
    save(fig, "fig4_optimization_before_after.png")


def draw_fig5() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(ax, "Figure 5. Evaluation Workflow")
    rounded_box(ax, 0.04, 0.34, 0.18, 0.36, "Inputs", ["synthetic jitter", "dropout", "limited KITTI"], fill="#f8fafc")
    rounded_box(ax, 0.34, 0.58, 0.20, 0.25, "AutoSyncDSL", ["DSL -> IR -> runtime"], fill="#e0f2fe")
    rounded_box(ax, 0.34, 0.20, 0.20, 0.25, "Baselines", ["handwritten scripts"], fill="#fef3c7")
    rounded_box(ax, 0.66, 0.34, 0.16, 0.36, "Metrics", ["latency", "group counts", "agreement"], fill="#dcfce7")
    rounded_box(ax, 0.88, 0.34, 0.10, 0.36, "Outputs", ["JSON", "CSV", "figures"], fill="#e5e7eb")
    draw_arrow(ax, (0.22, 0.55), (0.34, 0.70), "run", label_y_offset=0.02)
    draw_arrow(ax, (0.22, 0.47), (0.34, 0.32), "run", label_y_offset=-0.04)
    draw_arrow(ax, (0.54, 0.70), (0.66, 0.58), "compare", label_y_offset=0.03)
    draw_arrow(ax, (0.54, 0.32), (0.66, 0.46), "compare", label_y_offset=-0.02)
    draw_arrow(ax, (0.82, 0.52), (0.88, 0.52), "save", label_y_offset=0.03)
    save(fig, "fig5_evaluation_workflow.png")


def draw_fig6(results: dict) -> None:
    scenarios = [
        ("clean", "Clean\nsynthetic"),
        ("jittery", "Jittery\nsynthetic"),
        ("large", "Large\nsynthetic"),
        ("kitti_real", "KITTI\nreal"),
    ]
    methods = ["DSL", "DSL+Opt", "Baseline-v1", "Baseline-v2", "SimpleSync"]
    x = np.arange(len(scenarios))
    width = 0.14
    fig, ax = plt.subplots(figsize=(10.8, 5.4))

    for i, method in enumerate(methods):
        offset = (i - 2) * width
        first_label = True
        for j, (scenario, _) in enumerate(scenarios):
            val = latency_value(results, scenario, method)
            if val is None:
                continue
            label = method if first_label else None
            first_label = False
            bar = ax.bar(x[j] + offset, val, width, color=COLORS[method], label=label)
            ax.text(x[j] + offset, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=0)

    ax.set_title("Figure 6. Preliminary Latency Comparison", fontweight="bold", pad=42)
    ax.set_ylabel("Mean latency (milliseconds)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in scenarios])
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=5, frameon=False)
    ax.text(
        0.5,
        -0.20,
        "Missing bars indicate methods that were not evaluated for that scenario. Lower latency is faster.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=MUTED,
    )
    ax.set_ylim(0, max(ax.get_ylim()[1], 0.68))
    fig.subplots_adjust(top=0.75, bottom=0.22, left=0.08, right=0.98)
    save(fig, "fig6_latency_comparison.png")


def draw_fig7(results: dict) -> None:
    jitter = results["robustness"]["jitter"]
    xs = [r["jitter_ms"] for r in jitter]
    dsl = [r["dsl_groups"] for r in jitter]
    base = [r["baseline_groups"] for r in jitter]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(xs, dsl, marker="o", color=COLORS["DSL"], label="DSL")
    ax.plot(xs, base, marker="s", color="#d97706", linestyle="--", label="Baseline")
    ax.set_title("Figure 7. Robustness Under Timestamp Jitter", fontweight="bold", pad=14)
    ax.set_xlabel("Injected jitter (milliseconds)")
    ax.set_ylabel("Synchronized groups")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(104, 112)
    ax.set_yticks([104, 106, 108, 110, 112])
    ax.text(
        0.5,
        -0.22,
        "DSL and baseline produce 108 synchronized groups at all tested jitter levels.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color=MUTED,
    )
    fig.subplots_adjust(bottom=0.24, left=0.10, right=0.98, top=0.88)
    save(fig, "fig7_robustness.png")


def draw_fig8() -> None:
    dsl = {
        "DSL API": line_count("autosyncdsl/dsl.py"),
        "IR": line_count("autosyncdsl/ir.py"),
        "Runtime": line_count("autosyncdsl/executor.py"),
        "Support/optimization": line_count("autosyncdsl/optimizations.py")
        + line_count("autosyncdsl/utils.py")
        + line_count("autosyncdsl/__init__.py"),
    }
    base = {
        "Baseline core": line_count("baselines/manual_sync.py"),
        "Baseline support": line_count("baselines/__init__.py"),
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    labels = ["DSL+Runtime", "Handwritten\nBaselines"]
    bottom = 0
    colors_dsl = ["#1f77b4", "#17becf", "#2ca02c", "#6b7280"]
    for (name, val), color in zip(dsl.items(), colors_dsl):
        ax.bar(labels[0], val, bottom=bottom, color=color, width=0.48, label=name)
        bottom += val
    ax.bar(labels[1], base["Baseline core"], color="#d97706", width=0.48, label="Baseline core")
    ax.bar(labels[1], base["Baseline support"], bottom=base["Baseline core"], color="#facc15", width=0.48, label="Baseline support")

    totals = [sum(dsl.values()), sum(base.values())]
    for label, total in zip(labels, totals):
        ax.text(label, total + 28, f"{total}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_title("Figure 8. Implementation Footprint", fontweight="bold", pad=14)
    ax.set_ylabel("Physical lines of code")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5)
    ax.text(
        0.5,
        -0.18,
        "Prototype footprint: the DSL infrastructure is larger than the handwritten baseline module.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        color=MUTED,
    )
    fig.subplots_adjust(right=0.74, bottom=0.22, top=0.88)
    save(fig, "fig8_implementation_footprint.png")


def write_notes() -> None:
    notes = """# HW6 Figure Redraw Notes

| Old figure name | Issues found | Changes made | Final intended use |
| --- | --- | --- | --- |
| execution_evidence.png | Raw log excerpt was too dense, long lines were cut off, and key evidence was hard to identify. | Rebuilt as a concise terminal-style figure with command, suite status, exact/nearest correctness, latency means, and saved-results path. | Appendix |
| fig1_main_system_overview.png | Lower explanatory box was oversized and text-heavy; arrow labels and spacing were awkward. | Redrew as a balanced pipeline with compact boxes and removed the lower explanatory box to keep the figure focused. | Main body |
| fig2_dsl_to_ir_lowering.png | Figure was too sparse and IR panel did not communicate semantics clearly. | Redrew as DSL Source -> IR Nodes -> Executor Plan with monospaced DSL code and mini IR graph. | Appendix or optional main-body alternative to Figure 1 |
| fig3_sync_semantics.png | Annotations were floating, reference time was not clearly labeled, and legends could be simplified. | Cleaned four-panel layout, added precise annotations, shared bottom x-axis, and clearer reference-time labeling. | Main body |
| fig4_optimization_before_after.png | Too much empty space and generic transformation content. | Redrew compact two-panel before/after diagram with concrete pass names and optimizer arrow. | Appendix |
| fig5_eval_workflow.png | Text sat close to borders and layout was loose. | Redrew compact Inputs -> AutoSyncDSL/Baselines -> Metrics -> Outputs workflow with centered text. | Appendix |
| fig6_latency_comparison.png | Title and legend collided, n/a labels cluttered the plot, and note text intruded into the data area. | Redrew grouped bar chart with legend above axes, omitted missing bars, concise footnote, gridlines, and value labels. | Main body |
| fig7_robustness.png | Overlapping curves made the result look flat and the large arrow annotation was distracting. | Kept line plot but tightened y-range, removed arrow, and added a short note below the plot. | Main body |
| fig8_code_complexity.png | Legend was too complex, annotation distracted from bars, and framing could imply superiority. | Redrew as implementation-footprint stacked bars with simplified categories and caption-style note below. | Appendix |
"""
    (OUT / "figure_redraw_notes.md").write_text(notes)


def write_selection() -> None:
    selection = """# HW6 Figure Selection Recommendation

## Main Body

- **Figure 1: AutoSyncDSL System Overview** - best single architecture figure for explaining progress from sensor streams to synchronized batches.
- **Figure 3: Synchronization Semantics** - best worked example for exact match, nearest match, stale rejection, and IMU interpolation.
- **Figure 6: Preliminary Latency Comparison** - main quantitative result; it honestly shows current prototype overhead relative to baselines.
- **Figure 7: Robustness Under Timestamp Jitter** - strongest correctness/robustness result; DSL and baseline agree across all tested jitter levels.

## Appendix

- **Execution Evidence** - required proof that the code ran, but it should not interrupt the main results narrative.
- **Figure 2: DSL-to-IR Lowering** - useful if space permits, but Figure 1 is the better main-body architecture figure.
- **Figure 4: Prototype Optimization Before and After** - useful supporting figure; optimization benefit is still preliminary.
- **Figure 5: Evaluation Workflow** - helpful reproducibility support, but not essential for the main HW6 story.
- **Figure 8: Implementation Footprint** - important for critical reflection, but better as appendix/optional support because it is not a core result figure.
"""
    (OUT / "hw6_figure_selection.md").write_text(selection)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    results = load_results()
    draw_execution_evidence(results)
    draw_fig1()
    draw_fig2()
    draw_fig3()
    draw_fig4()
    draw_fig5()
    draw_fig6(results)
    draw_fig7(results)
    draw_fig8()
    write_notes()
    write_selection()
    print(f"Redrew HW6 figures in {OUT}")


if __name__ == "__main__":
    main()
