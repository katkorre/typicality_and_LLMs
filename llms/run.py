"""Play Scattergories with one model, in every language and strategy arm.

    python run.py llama                       # every game, resuming a previous run
    python run.py claude --language german    # the German games only
    python run.py gemma --smoke               # 3 categories, into results/smoke

Run from the repository root, since the registry scans it.
"""

import argparse
from pathlib import Path

from cinnamon.registry import Registry

from configurations.keys import MODELS, scattergories


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--language", choices=["english", "german", "spanish"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    valid, invalid = Registry.build(directory=Path.cwd())
    assert not invalid, sorted(str(key) for key in invalid)

    overrides = {}
    if args.language:
        overrides["languages"] = [args.language]
    if args.smoke:
        overrides |= {"limit": 3, "results_dir": "results/smoke"}
    Registry.from_key(scattergories(args.model), **overrides).run()


if __name__ == "__main__":
    main()
