"""Re-run RSA on existing PPO analysis JSONs using the updated 4-group scheme.

Iterates over all analysis JSONs in <experiment_root>/analysis_logs/ppo/,
locates the corresponding PPO checkpoint, evaluates N episodes for stimulus
frame collection, runs the updated shared/rsa.py RSA pipeline, and patches
the JSON in-place with the new rsa_* fields.

Usage:
    python redo_ppo_rsa.py --experiment_root ppo_experiment_root/seed_1
    python redo_ppo_rsa.py --experiment_root ppo_experiment_root/seed_1 --n_episodes 200 --device cpu
"""
import argparse
import glob
import json
import os
import sys
import traceback


def _find_checkpoint(experiment_root: str, basename: str) -> str | None:
    """Search for <basename>.pt under checkpoints/ppo/ (any subdirectory)."""
    search_root = os.path.join(experiment_root, "checkpoints", "ppo")
    candidates = glob.glob(os.path.join(search_root, "**", f"{basename}.pt"), recursive=True)
    return candidates[0] if candidates else None


def _redo_rsa_for_json(json_path: str, experiment_root: str, n_episodes: int, device: str):
    from ppo.sampling import evaluate_frozen_policy, load_ppo_agent
    from ppo.activations import PPO_HOOK_LAYER
    from shared.rsa import run_rsa
    from shared.storage import save_analysis_results

    basename = os.path.splitext(os.path.basename(json_path))[0]
    ckpt_path = _find_checkpoint(experiment_root, basename)

    if ckpt_path is None:
        print(f"  [SKIP] Checkpoint not found for {basename}")
        return

    print(f"  Processing {basename} ...")

    # Collect episodes for RSA stimulus detection
    _, _, episodes_with_transitions = evaluate_frozen_policy(
        ckpt_path, n_episodes=n_episodes, device=device,
    )

    # Load policy for activation extraction
    model, _ = load_ppo_agent(ckpt_path, device=device)
    policy = model.policy.to(device)

    rsa_results = run_rsa(
        policy, episodes_with_transitions,
        layer_name=PPO_HOOK_LAYER, device=device,
    )
    print(
        f"    RSA: {rsa_results['n_stimuli']} stimuli — "
        f"fighting={rsa_results['alignment_fighting']}, "
        f"resource={rsa_results['alignment_resource']}, "
        f"crafting={rsa_results['alignment_crafting']}, "
        f"housing={rsa_results['alignment_housing']}"
    )

    # Patch existing JSON
    with open(json_path) as f:
        record = json.load(f)

    record["rsa_alignment"]          = None  # superseded
    record["rsa_alignment_fighting"] = rsa_results.get("alignment_fighting")
    record["rsa_alignment_resource"] = rsa_results.get("alignment_resource")
    record["rsa_alignment_crafting"] = rsa_results.get("alignment_crafting")
    record["rsa_alignment_housing"]  = rsa_results.get("alignment_housing")
    record["rsa_n_stimuli"]          = rsa_results["n_stimuli"]
    record["rsa_labels"]             = rsa_results["labels"]
    record["rsa_rdm"]                = rsa_results["rdm"]
    record["rsa_n_frames"]           = rsa_results["n_frames"]

    save_analysis_results(record, json_path)
    print(f"    Updated: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-run PPO RSA with updated 4-group scheme on existing analysis JSONs"
    )
    parser.add_argument("--experiment_root", required=True,
                        help="Path to a single seed experiment root (e.g. ppo_experiment_root/seed_1)")
    parser.add_argument("--n_episodes", type=int, default=200,
                        help="Episodes to collect per checkpoint for RSA (default: 200)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    log_dir = os.path.join(args.experiment_root, "analysis_logs", "ppo")
    if not os.path.isdir(log_dir):
        print(f"No analysis logs found at {log_dir}")
        sys.exit(1)

    json_files = sorted(glob.glob(os.path.join(log_dir, "*.json")))
    if not json_files:
        print(f"No JSON files found in {log_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} analysis JSONs in {log_dir}")
    ok = 0
    for json_path in json_files:
        try:
            _redo_rsa_for_json(json_path, args.experiment_root, args.n_episodes, args.device)
            ok += 1
        except Exception:
            print(f"  [ERROR] {json_path}")
            traceback.print_exc()

    print(f"\nDone: {ok}/{len(json_files)} files updated.")


if __name__ == "__main__":
    main()
