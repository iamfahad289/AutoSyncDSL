#!/usr/bin/env python3
"""
Master Figure Generation Pipeline for AutoSyncDSL

This is the main script that orchestrates generation of all figures
used in proposal, final report, and presentation.

Run this script to regenerate all figures from source.
"""

import subprocess
import os
import sys
from pathlib import Path

def run_script(script_path, description):
    """Run a Python script and report status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print('='*70)

    try:
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=False)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"✗ Script not found: {script_path}")
        return False


def check_dependencies():
    """Verify required Python packages are installed."""
    print("Checking dependencies...")

    required = {
        'matplotlib': 'Charting library',
        'numpy': 'Numerical computing',
    }

    optional = {
        'cairosvg': 'SVG to PNG conversion (optional)',
    }

    missing = []

    for package, description in required.items():
        try:
            __import__(package)
            print(f"  ✓ {package}: {description}")
        except ImportError:
            print(f"  ✗ {package}: {description} [MISSING]")
            missing.append(package)

    for package, description in optional.items():
        try:
            __import__(package)
            print(f"  ✓ {package}: {description}")
        except ImportError:
            print(f"  ⚠ {package}: {description} [OPTIONAL, not installed]")

    if missing:
        print(f"\n⚠ Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    return True


def verify_output_files():
    """List all generated figure files."""
    print("\n" + "="*70)
    print("Generated Figure Files")
    print("="*70)

    figures_dir = Path("figures")

    if not figures_dir.exists():
        print("No figures directory found.")
        return

    files = sorted(figures_dir.glob("*"))

    if not files:
        print("No files found in figures/ directory")
        return

    print(f"\nTotal files: {len(files)}\n")

    for file in files:
        size_kb = file.stat().st_size / 1024
        print(f"  {file.name:<50} {size_kb:>8.1f} KB")

    # Summary
    png_files = list(figures_dir.glob("*.png"))
    svg_files = list(figures_dir.glob("*.svg"))
    pdf_files = list(figures_dir.glob("*.pdf"))

    print(f"\nSummary:")
    print(f"  PNG files: {len(png_files)}")
    print(f"  SVG files: {len(svg_files)}")
    print(f"  PDF files: {len(pdf_files)}")

    return len(files) > 0


def main():
    """Main pipeline."""

    print("\n" + "="*70)
    print("AutoSyncDSL: Complete Figure Generation Pipeline")
    print("="*70)

    # Check dependencies
    if not check_dependencies():
        print("\n⚠ Some required dependencies are missing.")
        print("Install with: pip install matplotlib numpy scipy")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return False

    # Create output directory
    os.makedirs("figures", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    print("\n" + "="*70)
    print("Generating Figures...")
    print("="*70)

    results = {}

    # 1. Main system overview figure
    results['main_figure'] = run_script(
        "scripts/generate_main_figure.py",
        "Main System Overview (Figure 1)"
    )

    # 2. Comparison and analysis figures
    results['comparison_figures'] = run_script(
        "scripts/generate_comparison_figures.py",
        "Comparison Figures (Figures 3, 6, 7, 8)"
    )

    # Summary of results
    print("\n" + "="*70)
    print("Generation Summary")
    print("="*70)

    print("\nResults:")
    for key, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {key}: {status}")

    # List generated files
    if verify_output_files():
        print("\n✓ Figure generation complete!")
    else:
        print("\n⚠ No figures were generated.")

    # Remaining figures
    print("\n" + "="*70)
    print("Remaining Figures (Manual Creation)")
    print("="*70)
    print("""
The following figures require manual creation in draw.io:
  - Figure 2: DSL to IR Lowering (draw.io recommended)
  - Figure 4: Optimization Before/After (draw.io recommended)
  - Figure 5: Evaluation Workflow (draw.io recommended)

Instructions:
  1. Open draw.io (https://draw.io or desktop app)
  2. Create figures using the specifications in:
     - docs/figure_plan.md
     - docs/main_figure_layout_notes.md
  3. Export as SVG and PNG to figures/ directory
  4. Optionally: Export to PDF for direct report embedding

Alternatively:
  - Use the Mermaid diagram (figures/main_system_overview.mmd)
  - Convert with: mmdc -i file.mmd -o file.png
""")

    print("="*70)
    print("Next Steps:")
    print("="*70)
    print("""
1. Review generated figures in figures/ directory
2. Create remaining figures in draw.io (Figures 2, 4, 5)
3. Verify all figures match docs/figure_plan.md specifications
4. Check figure captions in docs/figure_titles_and_captions.md
5. Embed figures in proposal/report/PPT using docs/figure_to_section_mapping.md
6. Use docs/final_ppt_figure_storyboard.md for presentation layout
""")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

