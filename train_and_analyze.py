"""Thin dispatcher — routes to ppo/train_and_analyze.py or rainbow/train_and_analyze.py.

Usage:
    python train_and_analyze.py --algorithm ppo   [ppo args...]
    python train_and_analyze.py --algorithm rainbow [rainbow args...]
    python train_and_analyze.py --help            (defaults to rainbow help)

Sub-scripts can also be invoked directly:
    python ppo/train_and_analyze.py     [ppo args...]
    python rainbow/train_and_analyze.py [rainbow args...]
"""
import sys

_ALGORITHMS = {"ppo", "rainbow"}


def main():
    algorithm = "rainbow"
    filtered = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--algorithm" and i + 1 < len(sys.argv):
            algorithm = sys.argv[i + 1]
            i += 2
        elif arg.startswith("--algorithm="):
            algorithm = arg.split("=", 1)[1]
            i += 1
        else:
            filtered.append(arg)
            i += 1

    if algorithm not in _ALGORITHMS:
        print(f"Error: --algorithm must be one of {sorted(_ALGORITHMS)}, got {algorithm!r}",
              file=sys.stderr)
        sys.exit(1)

    # Strip --algorithm before the sub-script's tyro.cli() parses argv
    sys.argv = [sys.argv[0]] + filtered

    if algorithm == "ppo":
        from ppo.train_and_analyze import main as _main
    else:
        from rainbow.train_and_analyze import main as _main

    _main()


if __name__ == "__main__":
    main()
