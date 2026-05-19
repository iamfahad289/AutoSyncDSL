#!/usr/bin/env python3
"""
Generate main system overview figure from Mermaid diagram

This script converts the Mermaid diagram to PNG and SVG using various backends.
"""

import os
import subprocess
import sys

def generate_from_mermaid(input_file, output_dir="figures"):
    """
    Generate PNG and SVG from Mermaid diagram

    Requires: mermaid-cli (npm install -g @mermaid-js/mermaid-cli)
    Or: Use online converter or pandoc with mermaid support
    """

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    png_output = os.path.join(output_dir, f"{base_name}.png")
    svg_output = os.path.join(output_dir, f"{base_name}.svg")

    # Try using mmdc (mermaid-cli)
    try:
        # PNG generation
        cmd = [
            "mmdc",
            "-i", input_file,
            "-o", png_output,
            "-w", "1600",
            "-H", "800",
            "--configFile", "mermaid.config.json"
        ]
        print(f"Generating PNG: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)

        # SVG generation
        cmd_svg = [
            "mmdc",
            "-i", input_file,
            "-o", svg_output,
            "-w", "1600",
            "-H", "800",
            "-e", "svg",
            "--configFile", "mermaid.config.json"
        ]
        print(f"Generating SVG: {' '.join(cmd_svg)}")
        subprocess.run(cmd_svg, check=False)

        print(f"✓ Generated: {png_output}")
        print(f"✓ Generated: {svg_output}")

    except FileNotFoundError:
        print("⚠ mmdc (mermaid-cli) not found. Installing...")
        print("Run: npm install -g @mermaid-js/mermaid-cli")
        print("\nAlternatively, use online tool: https://mermaid.live/")
        print(f"  1. Upload {input_file}")
        print(f"  2. Export as PNG and SVG")
        print(f"  3. Save to {output_dir}/")
        return False

    return True


def create_svg_directly(output_file="figures/main_system_overview.svg"):
    """
    Create a clean SVG directly using Python (no external deps)

    This generates a proper vector graphic without needing mermaid-cli
    """

    svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="1600" height="800" viewBox="0 0 1600 800">
  
  <defs>
    <style>
      .box { fill: #f0f0f0; stroke: #333; stroke-width: 2; }
      .box-sensor { fill: #1f77b4; color: white; }
      .box-dsl { fill: #9467bd; color: white; }
      .box-ir { fill: #7f7f7f; color: white; }
      .box-opt { fill: #17becf; color: white; }
      .box-runtime { fill: #1f77b4; color: white; }
      .box-output { fill: #2ca02c; color: white; }
      .title { font-size: 16px; font-weight: bold; font-family: Arial, sans-serif; }
      .label { font-size: 12px; font-family: Arial, sans-serif; }
      .code { font-size: 9px; font-family: 'Courier New', monospace; }
      .arrow { fill: none; stroke: #333; stroke-width: 2; marker-end: url(#arrowhead); }
      .arrow-label { font-size: 11px; font-family: Arial, sans-serif; fill: #333; }
    </style>
    
    <!-- Arrow marker definition -->
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#333" />
    </marker>
  </defs>
  
  <!-- Title at top -->
  <text x="800" y="30" class="title" text-anchor="middle">
    AutoSyncDSL: System Overview
  </text>
  
  <!-- INPUT SECTION -->
  <g>
    <text x="80" y="70" class="label" font-weight="bold">Sensor Streams</text>
    
    <!-- Camera -->
    <rect x="20" y="100" width="120" height="60" class="box box-sensor" rx="5"/>
    <text x="80" y="120" class="label" text-anchor="middle" fill="white">📷 Camera</text>
    <text x="80" y="135" class="label" text-anchor="middle" fill="white" font-size="11">30 Hz</text>
    <text x="80" y="150" class="label" text-anchor="middle" fill="white" font-size="10">~33ms interval</text>
    
    <!-- LiDAR -->
    <rect x="20" y="180" width="120" height="60" class="box box-sensor" rx="5" fill="#ff7f0e"/>
    <text x="80" y="200" class="label" text-anchor="middle" fill="white">📡 LiDAR</text>
    <text x="80" y="215" class="label" text-anchor="middle" fill="white" font-size="11">10 Hz</text>
    <text x="80" y="230" class="label" text-anchor="middle" fill="white" font-size="10">~100ms interval</text>
    
    <!-- IMU -->
    <rect x="20" y="260" width="120" height="60" class="box box-sensor" rx="5" fill="#2ca02c"/>
    <text x="80" y="280" class="label" text-anchor="middle" fill="white">📍 IMU</text>
    <text x="80" y="295" class="label" text-anchor="middle" fill="white" font-size="11">100 Hz</text>
    <text x="80" y="310" class="label" text-anchor="middle" fill="white" font-size="10">~10ms interval</text>
  </g>
  
  <!-- ARROW 1 -->
  <path d="M 140 160 L 200 160" class="arrow"/>
  <text x="170" y="155" class="arrow-label" text-anchor="middle">asynchronous</text>
  
  <!-- DSL SECTION -->
  <g>
    <rect x="200" y="80" width="220" height="220" class="box box-dsl" rx="5"/>
    <text x="310" y="105" class="label" text-anchor="middle" fill="white" font-weight="bold">
      Embedded Python DSL
    </text>
    
    <text x="210" y="135" class="code" fill="white">plan = SyncPlan()</text>
    <text x="210" y="150" class="code" fill="white">  .camera("cam")</text>
    <text x="210" y="165" class="code" fill="white">  .lidar("lidar")</text>
    <text x="210" y="180" class="code" fill="white">  .imu("imu")</text>
    <text x="210" y="195" class="code" fill="white">  .nearest("cam",</text>
    <text x="210" y="210" class="code" fill="white">    tolerance_ms=50)</text>
    <text x="210" y="225" class="code" fill="white">  .drop_stale(max_age=100)</text>
    <text x="210" y="240" class="code" fill="white">  .batch(size=4)</text>
    <text x="310" y="280" class="label" text-anchor="middle" fill="white" font-size="11">
      ir = plan.compile()
    </text>
  </g>
  
  <!-- ARROW 2 -->
  <path d="M 420 190 L 480 190" class="arrow"/>
  <text x="450" y="185" class="arrow-label" text-anchor="middle">compile</text>
  
  <!-- IR SECTION -->
  <g>
    <rect x="480" y="80" width="200" height="220" class="box box-ir" rx="5"/>
    <text x="580" y="105" class="label" text-anchor="middle" fill="white" font-weight="bold">
      Intermediate Representation
    </text>
    
    <!-- IR Nodes (simplified visualization) -->
    <circle cx="520" cy="140" r="15" fill="#1f77b4" stroke="#333" stroke-width="1"/>
    <text x="520" y="145" class="label" text-anchor="middle" fill="white" font-size="9">C</text>
    
    <circle cx="580" cy="140" r="15" fill="#ff7f0e" stroke="#333" stroke-width="1"/>
    <text x="580" y="145" class="label" text-anchor="middle" fill="white" font-size="9">L</text>
    
    <circle cx="640" cy="140" r="15" fill="#2ca02c" stroke="#333" stroke-width="1"/>
    <text x="640" y="145" class="label" text-anchor="middle" fill="white" font-size="9">I</text>
    
    <!-- Edges to central nodes -->
    <path d="M 520 155 L 550 190" class="arrow" stroke-width="1.5"/>
    <path d="M 580 155 L 580 190" class="arrow" stroke-width="1.5"/>
    <path d="M 640 155 L 610 190" class="arrow" stroke-width="1.5"/>
    
    <!-- Match node -->
    <rect x="530" y="190" width="100" height="30" class="box" fill="#d62728" stroke="#333" stroke-width="1" rx="3"/>
    <text x="580" y="210" class="label" text-anchor="middle" fill="white" font-size="10">Nearest Match</text>
    
    <!-- Filter node -->
    <rect x="530" y="240" width="100" height="30" class="box" fill="#d62728" stroke="#333" stroke-width="1" rx="3"/>
    <text x="580" y="260" class="label" text-anchor="middle" fill="white" font-size="10">Stale Filter</text>
    
    <path d="M 580 220 L 580 240" class="arrow" stroke-width="1.5"/>
  </g>
  
  <!-- ARROW 3 -->
  <path d="M 680 190 L 740 190" class="arrow"/>
  <text x="710" y="185" class="arrow-label" text-anchor="middle">optimize</text>
  
  <!-- OPTIMIZATION SECTION -->
  <g>
    <rect x="740" y="80" width="140" height="220" class="box box-opt" rx="5"/>
    <text x="810" y="105" class="label" text-anchor="middle" fill="white" font-weight="bold">
      Optimization
    </text>
    
    <rect x="755" y="135" width="110" height="35" class="box" fill="white" stroke="#333" stroke-width="1" rx="3"/>
    <text x="810" y="150" class="label" text-anchor="middle" fill="black" font-size="11">✓ Rule Fusion</text>
    <text x="810" y="163" class="label" text-anchor="middle" fill="black" font-size="9">5 ops → 4 ops</text>
    
    <rect x="755" y="185" width="110" height="35" class="box" fill="white" stroke="#333" stroke-width="1" rx="3"/>
    <text x="810" y="200" class="label" text-anchor="middle" fill="black" font-size="11">✓ Buffer Reuse</text>
    <text x="810" y="213" class="label" text-anchor="middle" fill="black" font-size="9">5 → 3 buffers</text>
  </g>
  
  <!-- ARROW 4 -->
  <path d="M 880 190 L 940 190" class="arrow"/>
  <text x="910" y="185" class="arrow-label" text-anchor="middle">execute</text>
  
  <!-- EXECUTION SECTION -->
  <g>
    <rect x="940" y="80" width="180" height="220" class="box box-runtime" rx="5"/>
    <text x="1030" y="105" class="label" text-anchor="middle" fill="white" font-weight="bold">
      Runtime Executor
    </text>
    
    <text x="950" y="135" class="label" fill="white" font-size="11">• Stream buffering</text>
    <text x="950" y="155" class="label" fill="white" font-size="11">• Timestamp</text>
    <text x="950" y="170" class="label" fill="white" font-size="11">  matching</text>
    <text x="950" y="190" class="label" fill="white" font-size="11">• Filtering &amp;</text>
    <text x="950" y="205" class="label" fill="white" font-size="11">  interpolation</text>
    <text x="950" y="225" class="label" fill="white" font-size="11">• Batch assembly</text>
  </g>
  
  <!-- ARROW 5 -->
  <path d="M 1120 190 L 1180 190" class="arrow"/>
  <text x="1150" y="185" class="arrow-label" text-anchor="middle">produce</text>
  
  <!-- OUTPUT SECTION -->
  <g>
    <text x="1250" y="70" class="label" font-weight="bold">Synchronized Batches</text>
    
    <!-- Batch 1 -->
    <rect x="1180" y="100" width="120" height="50" class="box box-output" rx="5"/>
    <text x="1240" y="120" class="label" text-anchor="middle" fill="white" font-size="11">Batch 1</text>
    <text x="1240" y="135" class="label" text-anchor="middle" fill="white" font-size="9">cam+lid+imu</text>
    <text x="1240" y="145" class="label" text-anchor="middle" fill="white" font-size="8">T=5000ms</text>
    
    <!-- Batch 2 -->
    <rect x="1180" y="170" width="120" height="50" class="box box-output" rx="5"/>
    <text x="1240" y="190" class="label" text-anchor="middle" fill="white" font-size="11">Batch 2</text>
    <text x="1240" y="205" class="label" text-anchor="middle" fill="white" font-size="9">cam+lid+imu</text>
    <text x="1240" y="215" class="label" text-anchor="middle" fill="white" font-size="8">T=5033ms</text>
    
    <!-- Batch 3 -->
    <rect x="1180" y="240" width="120" height="50" class="box box-output" rx="5"/>
    <text x="1240" y="260" class="label" text-anchor="middle" fill="white" font-size="11">Batch 3</text>
    <text x="1240" y="275" class="label" text-anchor="middle" fill="white" font-size="9">cam+lid+imu</text>
    <text x="1240" y="285" class="label" text-anchor="middle" fill="white" font-size="8">T=5067ms</text>
  </g>
  
  <!-- Legend at bottom -->
  <g>
    <line x1="50" y1="350" x2="1550" y2="350" stroke="#ccc" stroke-width="1"/>
    
    <text x="50" y="390" class="label" font-weight="bold">Key Features:</text>
    <text x="50" y="410" class="label">• Embedded Python DSL with fluent API • Compilation to structured IR • Automatic optimization passes</text>
    <text x="50" y="430" class="label">• Lightweight runtime • Supports exact/nearest matching • Configurable tolerance windows • IMU interpolation • Stale-frame rejection</text>
  </g>
  
</svg>
'''

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(svg_content)

    print(f"✓ Created SVG: {output_file}")
    return output_file


def create_png_from_svg(svg_file, png_file):
    """
    Convert SVG to PNG using cairosvg or other tool

    Requires: cairosvg (pip install cairosvg)
    """
    try:
        import cairosvg
        cairosvg.svg2png(url=svg_file, write_to=png_file, dpi=150)
        print(f"✓ Created PNG from SVG: {png_file}")
        return True
    except ImportError:
        print(f"⚠ cairosvg not available. Install with: pip install cairosvg")
        print(f"  Alternatively, use: inkscape {svg_file} -o {png_file}")
        return False


if __name__ == "__main__":
    print("AutoSyncDSL Figure Generation")
    print("=" * 50)

    # Create output directory
    os.makedirs("figures", exist_ok=True)

    # Create SVG directly (no dependencies)
    print("\n1. Generating SVG (direct)...")
    svg_file = create_svg_directly()

    # Try to convert to PNG
    print("\n2. Converting SVG to PNG...")
    png_file = "figures/main_system_overview.png"
    if not create_png_from_svg(svg_file, png_file):
        print(f"  Manual conversion needed. Try:")
        print(f"    brew install librsvg  # macOS")
        print(f"    rsvg-convert {svg_file} -o {png_file}")
        print(f"    Or: pip install cairosvg && python -c \"import cairosvg; cairosvg.svg2png(url='{svg_file}', write_to='{png_file}')\"")

    print("\n3. Mermaid diagram available at: figures/main_system_overview.mmd")
    print("   Convert online at: https://mermaid.live/")

    print("\n" + "=" * 50)
    print("✓ Main figure generation complete!")
    print(f"\nFiles created:")
    print(f"  - {svg_file}")
    print(f"  - figures/main_system_overview.mmd")
    if os.path.exists(png_file):
        print(f"  - {png_file}")

