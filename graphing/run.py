"""CLI entry point for the graphing module.

Usage:
    python -m graphing.run --entity ppo_crafter
    python -m graphing.run --entity ppo_crafter --run_id abc123 --output_dir graphs/
    python -m graphing.run --entity ppo_crafter --window 8 --dpi 200

Generates:
    - Standard metric timeseries (opposition, coherence, gradient magnitude,
      activation separation/cosine, RSA group alignments)
    - Zoomed achievement-event plots for each standard metric
    - Entity-specific plots (if graphing/entities/{entity_id}.py exists)
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots from an MLflow run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--entity",     required=True,  help="Entity ID (= MLflow experiment name)")
    parser.add_argument("--run_id",     default=None,   help="MLflow run ID (default: latest run)")
    parser.add_argument("--output_dir", default="graphs/", help="Output directory for PDFs")
    parser.add_argument("--tracking_uri", default="mlruns/", help="MLflow tracking URI")
    parser.add_argument("--window",     type=int, default=5,   help="Zoomed plot half-window (checkpoints)")
    parser.add_argument("--dpi",        type=int, default=150, help="Figure DPI")
    args = parser.parse_args()

    from graphing.loader import load_run
    import graphing.standard_plots as standard_plots
    import graphing.zoomed_plots as zoomed_plots

    print(f"\nLoading MLflow run — experiment: {args.entity!r}, run_id: {args.run_id or 'latest'}")
    run_data = load_run(args.entity, args.run_id, args.tracking_uri)
    print(f"  run_id: {run_data.run_id}")
    print(f"  checkpoints: {len(run_data.steps)}  "
          f"(steps {run_data.steps[0]:,} → {run_data.steps[-1]:,})" if run_data.steps else "  (no steps)")
    print(f"  achievements with first-unlock: {len(run_data.achievement_first_steps)}\n")

    output_dir = args.output_dir
    dpi = args.dpi
    saved: list[str] = []

    # Standard timeseries plots
    print("--- Standard metric plots ---")
    saved.extend(standard_plots.plot_all(run_data, output_dir, dpi))

    # Zoomed achievement-event plots
    print("\n--- Zoomed plots ---")
    saved.extend(zoomed_plots.plot_zoomed_all_metrics(
        run_data, output_dir, args.window, dpi))

    # Entity-specific soft-coded plots
    entity_module_name = f"graphing.entities.{args.entity}"
    try:
        entity_mod = importlib.import_module(entity_module_name)
        print(f"\n--- Entity-specific plots ({args.entity}) ---")
        entity_mod.plot(run_data, output_dir, dpi)
    except ModuleNotFoundError:
        print(f"\n(no entity-specific plots: {entity_module_name} not found)")
    except Exception as exc:
        print(f"\nWARN: entity plot failed: {exc}")

    print(f"\nDone — {len(saved)} files saved to {os.path.abspath(output_dir)}/")


if __name__ == "__main__":
    main()
