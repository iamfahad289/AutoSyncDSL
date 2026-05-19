#!/usr/bin/env python3
"""Generate schematic SVG/PNG figures for AutoSyncDSL.

Creates original, paper-style schematics for:
- Figure 2: DSL to IR lowering
- Figure 4: Optimization before/after
- Figure 5: Experimental workflow

The figures are intentionally simple and original, tailored to this project.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

COLORS = {
    "camera": "#1f77b4",
    "lidar": "#ff7f0e",
    "imu": "#2ca02c",
    "dsl": "#9467bd",
    "ir": "#7f7f7f",
    "opt": "#17becf",
    "runtime": "#1f77b4",
    "ok": "#2ca02c",
    "warn": "#d62728",
    "gray": "#e9ecef",
    "text": "#1f2937",
}


def svg_box(x, y, w, h, fill, title, subtitle=None, text_color="white"):
    lines = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" ry="10" fill="{fill}" stroke="#222" stroke-width="2"/>',
        f'<text x="{x + w/2}" y="{y + 28}" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="{text_color}">{title}</text>',
    ]
    if subtitle:
        lines.append(f'<text x="{x + w/2}" y="{y + 54}" text-anchor="middle" font-family="Arial" font-size="12" fill="{text_color}">{subtitle}</text>')
    return "\n".join(lines)


def arrow(x1, y1, x2, y2, label=None):
    label_svg = ""
    if label:
        label_svg = f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2 - 8}" text-anchor="middle" font-family="Arial" font-size="11" fill="#374151">{label}</text>'
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#222" stroke-width="2.5" marker-end="url(#arrow)"/>{label_svg}'


def svg_header(w=1600, h=800, title="AutoSyncDSL Figure"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L9,3 L0,6 z" fill="#222"/>
    </marker>
    <style>
      .small {{ font-family: Arial, sans-serif; font-size: 11px; fill: #374151; }}
      .code {{ font-family: 'Courier New', monospace; font-size: 11px; fill: #111827; }}
      .title {{ font-family: Arial, sans-serif; font-size: 22px; font-weight: 700; fill: #111827; }}
      .subtitle {{ font-family: Arial, sans-serif; font-size: 13px; fill: #4b5563; }}
    </style>
  </defs>
  <text x="{w/2}" y="34" text-anchor="middle" class="title">{title}</text>
'''


def close_svg():
    return '</svg>\n'


def write_svg(name: str, body: str):
    path = OUT / f"{name}.svg"
    path.write_text(body)
    return path


def to_png(svg_path: Path):
    png_path = svg_path.with_suffix('.png')
    if shutil.which('rsvg-convert'):
        subprocess.run(['rsvg-convert', str(svg_path), '-o', str(png_path)], check=False)
    elif shutil.which('sips'):
        # SIPS can rasterize SVG to PNG on macOS.
        subprocess.run(['sips', '-s', 'format', 'png', str(svg_path), '--out', str(png_path)], check=False)
    return png_path


def figure_2():
    parts = [svg_header(title='Figure 2: Lowering Embedded Synchronization Rules to an Executable Alignment Graph')]
    parts.append('<text x="210" y="78" class="subtitle">Embedded Python DSL</text>')
    parts.append(svg_box(70, 110, 360, 250, COLORS['dsl'], 'DSL Source', 'fluent embedded Python'))
    code = [
        'plan = (',
        '  SyncPlan()',
        '    .camera("cam")',
        '    .lidar("lidar")',
        '    .imu("imu")',
        '    .nearest("cam", tolerance_ms=50)',
        '    .interpolate("imu", window_ms=20)',
        '    .drop_stale(max_age_ms=100)',
        '    .batch(size=4)',
        ')'
    ]
    for i, line in enumerate(code):
        parts.append(f'<text x="95" y="160" class="code">{line}</text>' if i == 0 else f'<text x="95" y="{160 + i*22}" class="code">{line}</text>')
    parts.append(arrow(440, 235, 550, 235, 'compile()'))
    parts.append(svg_box(560, 90, 520, 300, COLORS['ir'], 'IR / Dataflow Graph', 'nodes for match, interpolate, filter, batch'))
    # simple DAG nodes
    node_fill = {'camera': COLORS['camera'], 'lidar': COLORS['lidar'], 'imu': COLORS['imu']}
    xs = [620, 760, 900]
    labels = ['camera', 'lidar', 'imu']
    for x, lbl in zip(xs, labels):
        parts.append(f'<rect x="{x}" y="145" width="90" height="36" rx="6" fill="{node_fill[lbl]}" stroke="#222"/>')
        parts.append(f'<text x="{x+45}" y="168" text-anchor="middle" font-family="Arial" font-size="12" fill="white">Source</text>')
        parts.append(f'<text x="{x+45}" y="183" text-anchor="middle" font-family="Arial" font-size="10" fill="white">{lbl}</text>')
    parts.append(arrow(665, 181, 770, 235))
    parts.append(arrow(805, 181, 800, 235))
    parts.append(arrow(945, 181, 830, 235))
    parts.append(svg_box(725, 245, 150, 40, '#d97706', 'NearestMatch', 'tol=50ms'))
    parts.append(svg_box(725, 300, 150, 40, '#dc2626', 'StaleFilter', 'max_age=100ms'))
    parts.append(svg_box(725, 355, 150, 40, '#6b7280', 'Batch', 'size=4'))
    parts.append(arrow(800, 275, 800, 300))
    parts.append(arrow(800, 340, 800, 355))
    parts.append(arrow(1080, 235, 1180, 235, 'optimize + execute'))
    parts.append(svg_box(1200, 95, 330, 290, COLORS['runtime'], 'Execution Runtime', 'lightweight executor'))
    runtime_text = [
        '• stream buffering',
        '• exact / nearest matching',
        '• IMU interpolation',
        '• stale-frame rejection',
        '• fixed-size batching',
        '• trace logging'
    ]
    for i, line in enumerate(runtime_text):
        parts.append(f'<text x="1225" y="160" class="small">{line}</text>' if i == 0 else f'<text x="1225" y="{160 + i*28}" class="small">{line}</text>')
    parts.append(svg_box(1230, 425, 290, 160, COLORS['ok'], 'Synchronized Batches', 'ready for downstream perception'))
    for i in range(3):
        y = 465 + i*35
        parts.append(f'<rect x="1250" y="{y}" width="250" height="24" rx="4" fill="#ffffff" stroke="#ffffff" opacity="0.9"/>')
        parts.append(f'<text x="1265" y="{y+16}" class="code">Batch {i+1}: cam + lidar + imu</text>')
    parts.append(close_svg())
    svg = write_svg('dsl_to_ir_lowering', ''.join(parts))
    to_png(svg)


def figure_4():
    parts = [svg_header(title='Figure 4: Optimization of Alignment Plans Through Rule Fusion and Buffer Reuse')]
    parts.append(svg_box(60, 95, 680, 560, '#ef4444', 'Before', 'unoptimized execution plan', text_color='white'))
    parts.append(svg_box(860, 95, 680, 560, '#10b981', 'After', 'optimized execution plan', text_color='white'))
    # before nodes
    before_nodes = [('Match', 110, 170), ('Filter', 110, 255), ('Interp', 110, 340), ('Batch', 110, 425)]
    for lbl, x, y in before_nodes:
        parts.append(f'<rect x="{x}" y="{y}" width="180" height="48" rx="8" fill="#ffffff" stroke="#111827" stroke-width="1.5"/>')
        parts.append(f'<text x="{x+90}" y="{y+30}" text-anchor="middle" class="small" font-weight="700">{lbl}</text>')
    parts.append(arrow(290, 194, 420, 194, 'temp buffers'))
    parts.append(arrow(290, 279, 420, 279, 'temp buffers'))
    parts.append(arrow(290, 364, 420, 364, 'temp buffers'))
    parts.append(arrow(290, 449, 420, 449, 'temp buffers'))
    # after nodes
    after_nodes = [('Fused Match+Filter', 915, 205), ('Interp', 915, 320), ('Batch', 915, 435)]
    for lbl, x, y in after_nodes:
        parts.append(f'<rect x="{x}" y="{y}" width="210" height="54" rx="8" fill="#ffffff" stroke="#111827" stroke-width="1.5"/>')
        parts.append(f'<text x="{x+105}" y="{y+32}" text-anchor="middle" class="small" font-weight="700">{lbl}</text>')
    parts.append(arrow(1125, 232, 1250, 232, 'reused buffers'))
    parts.append(arrow(1125, 347, 1250, 347, 'reused buffers'))
    parts.append(arrow(1125, 462, 1250, 462, 'single buffer'))
    parts.append(svg_box(1235, 180, 245, 330, '#ffffff', 'Impact', 'qualitative improvement', text_color='#111827'))
    impact = ['• fewer passes', '• fewer allocations', '• simpler execution path', '• measurable latency reduction', '• same semantics']
    for i, line in enumerate(impact):
        parts.append(f'<text x="1260" y="230" class="small" fill="#111827">{line}</text>' if i == 0 else f'<text x="1260" y="{230 + i*46}" class="small" fill="#111827">{line}</text>')
    parts.append(close_svg())
    svg = write_svg('optimization_before_after', ''.join(parts))
    to_png(svg)


def figure_5():
    parts = [svg_header(title='Figure 5: End-to-End Evaluation Workflow and Benchmark Setup')]
    parts.append(svg_box(70, 100, 300, 140, COLORS['gray'], 'perturbed KITTI Data', 'jitter + dropout + out-of-order', text_color='#111827'))
    parts.append(svg_box(70, 275, 300, 140, COLORS['gray'], 'Optional KITTI Raw', 'camera + LiDAR + OXTS fallback', text_color='#111827'))
    parts.append(arrow(220, 240, 220, 275, 'load'))
    parts.append(svg_box(430, 155, 280, 150, COLORS['dsl'], 'Baseline', 'handwritten Python', text_color='white'))
    parts.append(svg_box(430, 340, 280, 150, COLORS['runtime'], 'AutoSyncDSL', 'DSL → IR → runtime', text_color='white'))
    parts.append(arrow(370, 170, 430, 220, 'evaluate'))
    parts.append(arrow(370, 345, 430, 395, 'evaluate'))
    parts.append(svg_box(785, 145, 245, 150, COLORS['opt'], 'Metrics', 'correctness, latency, robustness', text_color='white'))
    parts.append(arrow(710, 220, 785, 220, 'compare'))
    parts.append(arrow(710, 405, 785, 265, 'compare'))
    parts.append(svg_box(1080, 120, 450, 420, '#ffffff', 'Outputs', 'results and plots', text_color='#111827'))
    out_lines = ['• results/all_experiments.json', '• latency comparison plot', '• robustness table', '• code complexity comparison', '• figure-ready summaries']
    for i, line in enumerate(out_lines):
        parts.append(f'<text x="1115" y="195" class="small" fill="#111827">{line}</text>' if i == 0 else f'<text x="1115" y="{195 + i*52}" class="small" fill="#111827">{line}</text>')
    parts.append(arrow(1030, 220, 1080, 220, 'results'))
    parts.append(arrow(1030, 395, 1080, 330, 'results'))
    parts.append(close_svg())
    svg = write_svg('eval_workflow', ''.join(parts))
    to_png(svg)


def main():
    figure_2()
    figure_4()
    figure_5()
    print('Generated schematic figures: Figure 2, Figure 4, Figure 5')


if __name__ == '__main__':
    main()

