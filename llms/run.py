"""Play the Scattergories games that match every given filter, one after another.

    python run.py --model llama                    # every Llama game
    python run.py --model llama --language german  # the German Llama games
    python run.py --model llama gemma --rounds with_history --use-strategies true
    python run.py --model claude --language english --rounds without_history \\
        --use-strategies false                     # exactly one game
    python run.py --model gemma --rounds with_history_sampling --num-rounds 52
    python run.py --model claude --seeds 42        # only the games of seed 42
    python run.py --model llama --list             # list the matching games
    python run.py --model random --sanity-check    # 3 categories, results/sanity_check

A filter that is not given matches every value. The option --seeds replaces
the seeds of every selected configuration, and a seed names its game directory.
A game finished earlier is skipped, and an unfinished game resumes at its first
unfinished round. Run from the repository root, since the registry scans it.
"""

import argparse
from pathlib import Path

from cinnamon.registry import Registry

from configurations.keys import GAMES, NAMESPACE

FILTERS = {
    "language": ["english", "german", "spanish"],
    "rounds": ["without_history", "with_history", "with_history_sampling"],
    "use_strategies": ["true", "false"],
    "num_rounds": ["26", "52"],
}


def select_keys(models: list[str], filters: dict[str, list[str] | None]) -> list:
    """Return the game keys of ``models`` whose configuration matches ``filters``.

    A key search finds the candidates of each model. The filters then compare
    the configuration values, since a default value such as ``english`` adds no
    tag to its key.
    """
    selected = []
    for model in models:
        for key in Registry.retrieve_keys(
            names="scattergories", namespaces=NAMESPACE, tags={model}
        ):
            config = Registry.retrieve_configuration(registration_key=key)
            if all(
                values is None or str(getattr(config, name)).lower() in values
                for name, values in filters.items()
            ):
                selected.append(key)
    return selected


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", nargs="+", choices=GAMES, required=True)
    for name, choices in FILTERS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", nargs="+", choices=choices)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--sanity-check", action="store_true")
    args = parser.parse_args()

    # Invalid keys are the variant combinations that a condition excludes.
    Registry.build(directory=Path.cwd())

    keys = select_keys(args.model, {name: getattr(args, name) for name in FILTERS})
    print(f"{len(keys)} matching configurations", flush=True)
    overrides = {}
    if args.seeds:
        overrides["seeds"] = args.seeds
    if args.sanity_check:
        overrides |= {"limit": 3, "results_dir": "results/sanity_check"}
    for key in keys:
        print(key, flush=True)
        if not args.list:
            Registry.from_key(key, **overrides).run()


if __name__ == "__main__":
    main()
