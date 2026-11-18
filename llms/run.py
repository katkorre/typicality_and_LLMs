"""Play Scattergories with one model, in every language.

    python run.py llama                         # the games without strategies, resuming
    python run.py llama --use-strategies        # the games with strategies
    python run.py llama --rounds with_history   # the games with history
    python run.py claude --language german      # the German games only
    python run.py gemma --smoke                 # 3 categories, into results/smoke

Run from the repository root, since the registry scans it.
"""

import argparse
from pathlib import Path

from cinnamon.registry import Registry

from configurations.keys import GAMES


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", choices=GAMES)
    parser.add_argument("--language", choices=["english", "german", "spanish"])
    parser.add_argument("--use-strategies", action="store_true")
    parser.add_argument(
        "--rounds",
        choices=["without_history", "with_history", "with_history_sampling"],
        default="without_history",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    valid, invalid = Registry.build(directory=Path.cwd())
    assert not invalid, sorted(str(key) for key in invalid)

    overrides = {}
    if args.language:
        overrides["languages"] = [args.language]
    if args.smoke:
        overrides |= {"limit": 3, "results_dir": "results/smoke"}
    variant = {}
    if args.use_strategies:
        variant["use_strategies"] = True
    if args.rounds != "without_history":
        variant["rounds"] = args.rounds
    key = GAMES[args.model]
    if variant:
        key = key.from_variant(variant)
    Registry.from_key(key, **overrides).run()


if __name__ == "__main__":
    main()
