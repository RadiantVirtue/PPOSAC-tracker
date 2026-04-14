"""dissertation_graphs/run_all.py

Master runner: generates all dissertation graphs in one command.

Primary RQ graphs (6):
  shared/output/rq1_cross_algorithm_opposition.png
  rainbow/output/rq2_weight_delta_rainbow_averaged.png
  ppo/output/rq3_rsa_alignment_ppo_averaged.png
  rainbow/output/rq4_mora_opp_pos_neutral_rainbow_averaged.png
  ppo/output/milestone_delta_bar_ppo.png
  shared/output/milestone_delta_comparison.png

Supplementary metric timeseries (one PNG per metric):
  ppo/output/metrics/*.png          (~18 graphs)
  rainbow/output/metrics/*.png      (~26 graphs)

Achievement zoom charts (+/- 200k-step window around each unlock):
  ppo/output/zooms/<achievement>_<metric>.png
  rainbow/output/zooms/<achievement>_<metric>.png

Usage (run from PPOSAC-tracker/):
    python dissertation_graphs/run_all.py \\
        --ppo_root ppo_experiment_root \\
        --rainbow_root rainbow_experiment_root

    # Custom DPI:
    python dissertation_graphs/run_all.py \\
        --ppo_root ppo_experiment_root \\
        --rainbow_root rainbow_experiment_root \\
        --dpi 200
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

# Import each sub-module's plot() function
sys.path.insert(0, os.path.join(_HERE, "shared"))
sys.path.insert(0, os.path.join(_HERE, "ppo"))
sys.path.insert(0, os.path.join(_HERE, "rainbow"))

import importlib.util as _ilu
import types


def _import_script(path: str, name: str) -> types.ModuleType:
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo_root", required=True,
                        help="Path to PPO experiment root")
    parser.add_argument("--rainbow_root", required=True,
                        help="Path to Rainbow experiment root")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    def _abs(path: str) -> str:
        return os.path.join(_ROOT, path) if not os.path.isabs(path) else path

    ppo_root = _abs(args.ppo_root)
    rbw_root = _abs(args.rainbow_root)
    dpi = args.dpi

    scripts = [
        # (module_path, out_dir, call_fn)
        (
            os.path.join(_HERE, "shared", "plot_rq1_cross_algorithm.py"),
            os.path.join(_HERE, "shared", "output"),
            lambda mod, od: mod.plot(ppo_root, rbw_root, od, dpi),
        ),
        (
            os.path.join(_HERE, "rainbow", "plot_rq2_weight_delta.py"),
            os.path.join(_HERE, "rainbow", "output"),
            lambda mod, od: mod.plot(rbw_root, od, dpi),
        ),
        (
            os.path.join(_HERE, "ppo", "plot_rq3_rsa_alignment.py"),
            os.path.join(_HERE, "ppo", "output"),
            lambda mod, od: mod.plot(ppo_root, od, dpi),
        ),
        (
            os.path.join(_HERE, "rainbow", "plot_rq4_mora_opp_pos_neutral.py"),
            os.path.join(_HERE, "rainbow", "output"),
            lambda mod, od: mod.plot(rbw_root, od, dpi),
        ),
        (
            os.path.join(_HERE, "ppo", "plot_milestone_delta_bar.py"),
            os.path.join(_HERE, "ppo", "output"),
            lambda mod, od: mod.plot(ppo_root, od, dpi),
        ),
        (
            os.path.join(_HERE, "shared", "plot_milestone_delta_comparison.py"),
            os.path.join(_HERE, "shared", "output"),
            lambda mod, od: mod.plot(ppo_root, rbw_root, od, dpi),
        ),
        # ── Supplementary: per-metric timeseries ───────────────────────────────
        (
            os.path.join(_HERE, "ppo", "plot_all_metrics.py"),
            os.path.join(_HERE, "ppo", "output", "metrics"),
            lambda mod, od: mod.plot(ppo_root, od, dpi),
        ),
        (
            os.path.join(_HERE, "rainbow", "plot_all_metrics.py"),
            os.path.join(_HERE, "rainbow", "output", "metrics"),
            lambda mod, od: mod.plot(rbw_root, od, dpi),
        ),
        # ── Supplementary: achievement zoom charts ─────────────────────────────
        (
            os.path.join(_HERE, "ppo", "plot_achievement_zooms.py"),
            os.path.join(_HERE, "ppo", "output", "zooms"),
            lambda mod, od: mod.plot(ppo_root, od, dpi),
        ),
        (
            os.path.join(_HERE, "rainbow", "plot_achievement_zooms.py"),
            os.path.join(_HERE, "rainbow", "output", "zooms"),
            lambda mod, od: mod.plot(rbw_root, od, dpi),
        ),
    ]

    results = {}
    for script_path, out_dir, call_fn in scripts:
        script_name = os.path.splitext(os.path.basename(script_path))[0]
        print(f"\n[{script_name}]")
        try:
            mod = _import_script(script_path, script_name)
            ret = call_fn(mod, out_dir)
            results[script_name] = ret
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results[script_name] = f"FAILED: {exc}"

    print("\n" + "=" * 60)
    print("Summary:")
    for name, ret in results.items():
        if isinstance(ret, list):
            status = "OK" if ret else "EMPTY"
            print(f"  [{status}] {name}  ({len(ret)} files)")
        else:
            failed = not ret or str(ret).startswith("FAILED")
            status = "FAIL" if failed else "OK"
            print(f"  [{status}] {name}")
            if ret and not failed:
                print(f"         {ret}")
    print("=" * 60)


if __name__ == "__main__":
    main()
