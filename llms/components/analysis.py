"""Report the points of every agent, per language, over the finished games.

    python -m components.analysis                     # game points
    python -m components.analysis --points valid      # valid points only

A game scores the sum of its rounds. A cell holds the mean and the sample
standard deviation over the seeds of that game, followed by the number of
seeds. One seed has no standard deviation, shown as ``-``. Totals of different
languages are not comparable, since each language plays a different number of
categories.
"""

import argparse
from pathlib import Path

import pandas as pd

COLUMNS = {"game": "total", "valid": "total_valid"}


def summarize(results_dir: Path, points: str = "game") -> pd.DataFrame:
    """Return one row per model, strategy setting, round setting and agent.

    A game lives under
    ``<model>/<language>/<strategy setting>/<round setting>/seed_<seed>``, and
    only games with ``game_info.txt`` count. The columns pair a statistic,
    ``mean``, ``std`` or ``count`` over seeds, with a language.
    """
    games = []
    for info in sorted(results_dir.glob("*/*/*/*/seed_*/game_info.txt")):
        game = info.parent
        model, language, strategies, rounds, seed = game.relative_to(results_dir).parts
        totals = (
            pd.concat(
                pd.read_csv(path, usecols=["agent", COLUMNS[points]])
                for path in game.glob("round_*/scores.csv")
            )
            .groupby("agent", as_index=False)[COLUMNS[points]]
            .sum()
        )
        games.append(
            totals.assign(
                model=model,
                language=language,
                strategies=strategies,
                rounds=rounds,
                seed=seed,
            )
        )
    if not games:
        raise FileNotFoundError(f"no finished game under {results_dir}")
    return (
        pd.concat(games)
        .groupby(["model", "strategies", "rounds", "agent", "language"])[
            COLUMNS[points]
        ]
        .agg(["mean", "std", "count"])
        .unstack("language")
    )


def format_cells(summary: pd.DataFrame) -> pd.DataFrame:
    """Write each language as ``mean ± std (seeds)``."""
    return pd.DataFrame(
        {
            language: [
                f"{mean:.1f} ± {'-' if pd.isna(std) else f'{std:.1f}'} ({count:.0f})"
                for mean, std, count in zip(
                    summary["mean", language],
                    summary["std", language],
                    summary["count", language],
                )
            ]
            for language in summary["mean"].columns
        },
        index=summary.index,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default="results/scattergories", type=Path)
    parser.add_argument("--points", choices=COLUMNS, default="game")
    args = parser.parse_args()
    print(format_cells(summarize(args.results_dir, args.points)).to_string())


if __name__ == "__main__":
    main()
