"""dissertation_graphs/run_all.py

Master runner: generates all dissertation figures in one command.

All PDFs are written under PPOSAC-tracker/GRAPHS/:

  Dissertation graphs/
    7.2.1/  rq1_cross_algorithm_opposition.pdf
    7.2.2/  coherence_success_ppo.pdf, coherence_failure_ppo.pdf
            coherence_success_rainbow.pdf, coherence_failure_rainbow.pdf
    7.2.3/  activation_separation_ppo.pdf, activation_cosine_dist_ppo.pdf
            activation_separation_rainbow.pdf, activation_cosine_dist_rainbow.pdf
    7.3/    cos_uniform_is_success_rainbow.pdf, cos_uniform_is_failure_rainbow.pdf
            cos_is_reward_rainbow.pdf
    7.4.1/  mor_ratio_rainbow.pdf, mora_grad_mag_all_rainbow.pdf
    7.4.2/  mora_opp_joint_panel.pdf
    7.5.1/  ppo_euclidean_separation.pdf, rainbow_euclidean_separation.pdf
            ppo_cosine_distance.pdf, rainbow_cosine_distance.pdf
    7.5.2/  rsa_alignment_ppo.pdf
            rsa_fighting_resource_rainbow.pdf, rsa_crafting_housing_rainbow.pdf
    7.5.3/  achievement_rsa_group_opposition_ppo.pdf
            achievement_rsa_group_opposition_rainbow.pdf
    7.5.4/  make_stone_sword_case_study_ppo.pdf
            make_stone_sword_case_study_rainbow.pdf
    A.1/    weight_delta_rainbow.pdf

  Ablation graphs/
    PPO/      abl_ppo_cross_seed_opposition.pdf, abl_ppo_cross_seed_coherence.pdf
              abl_no_gae_opposition.pdf, abl_no_gae_coherence.pdf  (--no_gae_ppo_root)
    Rainbow/  abl_scalar_dqn_mora_ratio.pdf
              abl_fixed_threshold_mora_opp.pdf, abl_fixed_threshold_mora_bar.pdf  (--rainbow_v2_root)
              abl_growing_rsa_rainbow.pdf  (--rainbow_v2_root)

Usage (run from PPOSAC-tracker/):
    python dissertation_graphs/run_all.py \\
        --ppo_root ppo_experiment_root \\
        --rainbow_root rainbow_experiment_root \\
        [--rainbow_v2_root rainbow_v2] \\
        [--no_gae_ppo_root no_gae_ppo_root] \\
        [--scalar_json scalar_ablation_results/scalar_ablation_result.json] \\
        [--marker_every 3]
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import importlib.util as _ilu
import types


def _import_script(path: str, name: str) -> types.ModuleType:
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_D = {
    "ppo_root":        "ppo_experiment_root",
    "rainbow_root":    "rainbow_v2",
    "rainbow_v2_root": "rainbow_v2",
    "no_gae_ppo_root": "experiment_root_no_gae",
    "scalar_json":     "rainbow_v2/scalar_ablation/scalar_ablation_result.json",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo_root",        default=_D["ppo_root"])
    parser.add_argument("--rainbow_root",    default=_D["rainbow_root"])
    parser.add_argument("--rainbow_v2_root", default=_D["rainbow_v2_root"])
    parser.add_argument("--no_gae_ppo_root", default=_D["no_gae_ppo_root"])
    parser.add_argument("--scalar_json",     default=_D["scalar_json"])
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--marker_every", type=int, default=None,
                        help="Markers per N data points. 0 = no markers. "
                             "Default: value in shared/graphing.py.")
    args = parser.parse_args()

    def _abs(path: str) -> str:
        return os.path.join(_ROOT, path) if not os.path.isabs(path) else path

    ppo_root    = _abs(args.ppo_root)
    rbw_root    = _abs(args.rainbow_root)
    rbw_v2_root = _abs(args.rainbow_v2_root) if args.rainbow_v2_root else None
    no_gae_root = _abs(args.no_gae_ppo_root) if args.no_gae_ppo_root else None
    scalar_json = _abs(args.scalar_json)
    dpi = args.dpi

    # Patch MARKER_EVERY before scripts are dynamically imported.
    # 0 is a valid value meaning "no markers" — handled in _make_shaded_line.
    if args.marker_every is not None:
        import shared.graphing as _gph
        _gph.MARKER_EVERY = args.marker_every
        print(f"INFO: MARKER_EVERY = {args.marker_every}")

    # Each entry: (script_path, call_fn) — out_dir is now ignored by all scripts
    scripts: list[tuple[str, callable]] = [
        (
            os.path.join(_HERE, "shared", "plot_rq1_cross_algorithm.py"),
            lambda mod: mod.plot(ppo_root, rbw_root, dpi=dpi),
        ),
        (
            os.path.join(_HERE, "shared", "plot_activation_comparison.py"),
            lambda mod: mod.plot(ppo_root, rbw_root, dpi=dpi),
        ),
        (
            os.path.join(_HERE, "rainbow", "plot_rq2_weight_delta.py"),
            lambda mod: mod.plot(rbw_root, dpi=dpi),
        ),
        (
            os.path.join(_HERE, "ppo", "plot_rq3_rsa_alignment.py"),
            lambda mod: mod.plot(ppo_root, dpi=dpi),
        ),
        (
            os.path.join(_HERE, "rainbow", "plot_mor_ratio_vs_return.py"),
            lambda mod: mod.plot(rbw_root, dpi=dpi),
        ),
        (
            os.path.join(_HERE, "rainbow", "plot_cos_is_reward.py"),
            lambda mod: mod.plot(rbw_root, dpi=dpi),
        ),
        (
            os.path.join(_HERE, "rainbow", "plot_mora_combined.py"),
            lambda mod: mod.plot(rbw_root, dpi=dpi),
        ),
        (
            os.path.join(_HERE, "shared", "plot_achievement_rsa_group_opposition.py"),
            lambda mod: mod.plot(ppo_root, rbw_root, dpi=dpi),
        ),
        (
            os.path.join(_HERE, "shared", "plot_make_stone_sword_case_study.py"),
            lambda mod: mod.plot(ppo_root, rbw_root, dpi=dpi),
        ),
        (
            os.path.join(_HERE, "shared", "plot_centroid_decomposition.py"),
            lambda mod: mod.plot(dpi=dpi),
        ),
        (
            os.path.join(_HERE, "ppo", "plot_all_metrics.py"),
            lambda mod: mod.plot(ppo_root, dpi=dpi),
        ),
        (
            os.path.join(_HERE, "rainbow", "plot_all_metrics.py"),
            lambda mod: mod.plot(rbw_root, dpi=dpi),
        ),
        (
            os.path.join(_HERE, "ablations", "ablation_ppo_cross_seed.py"),
            lambda mod: mod.plot(ppo_root, dpi=dpi),
        ),
        (
            os.path.join(_HERE, "ablations", "ablation_scalar_dqn.py"),
            lambda mod, sj=scalar_json: mod.plot(rbw_root, sj, dpi=dpi),
        ),
    ]

    if rbw_v2_root:
        scripts += [
            (
                os.path.join(_HERE, "ablations", "ablation_fixed_threshold.py"),
                lambda mod, r2=rbw_v2_root: mod.plot(rbw_root, r2, dpi=dpi),
            ),
            (
                os.path.join(_HERE, "ablations", "ablation_growing_rsa.py"),
                lambda mod, r2=rbw_v2_root: mod.plot(rbw_root, r2, dpi=dpi),
            ),
        ]
    else:
        print("INFO: --rainbow_v2_root not provided; "
              "skipping fixed-threshold and growing-RSA ablations.")

    if no_gae_root:
        scripts += [
            (
                os.path.join(_HERE, "ablations", "ablation_no_gae_ppo.py"),
                lambda mod, ng=no_gae_root: mod.plot(ppo_root, ng, dpi=dpi),
            ),
        ]
    else:
        print("INFO: --no_gae_ppo_root not provided; skipping no-GAE ablation.")

    results = {}
    for script_path, call_fn in scripts:
        script_name = os.path.splitext(os.path.basename(script_path))[0]
        print(f"\n[{script_name}]")
        if not os.path.isfile(script_path):
            print(f"  SKIP: script not found at {script_path}")
            results[script_name] = "SKIP"
            continue
        try:
            mod = _import_script(script_path, script_name)
            ret = call_fn(mod)
            results[script_name] = ret
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results[script_name] = f"FAILED: {exc}"

    print("\n" + "=" * 60)
    print("Summary:")
    for name, ret in results.items():
        if isinstance(ret, list):
            status = "OK" if ret else "EMPTY"
            print(f"  [{status}] {name}  ({len(ret)} files)")
        elif ret == "SKIP":
            print(f"  [SKIP] {name}")
        else:
            failed = not ret or str(ret).startswith("FAILED")
            status = "FAIL" if failed else "OK"
            print(f"  [{status}] {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
