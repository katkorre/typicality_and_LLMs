import re
from typing import Any

import pandas as pd

from components.agent import AgentAnswer


def normalize_text(value: str) -> str:
    """Normalize text for matching model answers against the gold dataset."""
    value = value.strip().lower()
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^[\"']|[\"']$", "", value)
    return value


def score_answers(
    agent_answers: list[AgentAnswer],
    categories: list[str],
    members: dict[str, set[str]],
    round_letter: str,
    used_words: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Score a round according to the rules of the game.

    For each category, a blank answer, an answer with another initial, or an
    answer in ``used_words`` of the category scores 0 points. The remaining
    answers score points and are compared only with each other. An answer
    shared with another agent scores 5 points. The only answer that scores
    points gets 20 points, and any other unique answer gets 10 points. The
    valid points repeat the points of an answer found among ``members`` of its
    own category, the gold words that the dataset lists for it, and are 0
    otherwise.
    """
    used_words = used_words or {}

    def scores_points(value: str, category: str) -> bool:
        return value.startswith(round_letter) and value not in used_words.get(
            category, set()
        )

    rows = []

    for answer in agent_answers:
        row: dict[str, Any] = {"agent": answer.agent_name}
        total = 0
        total_valid = 0

        for category in categories:
            raw_value = answer.answers.get(category, "")
            value = raw_value.strip().lower()

            scoring_values = [
                v
                for a in agent_answers
                if scores_points(
                    v := a.answers.get(category, "").strip().lower(), category
                )
            ]

            if not scores_points(value, category):
                points = 0
            elif scoring_values.count(value) > 1:
                points = 5
            elif len(scoring_values) == 1:
                points = 20
            else:
                points = 10

            valid_points = points if value in members.get(category, set()) else 0

            row[category] = raw_value
            row[f"{category}_points"] = points
            row[f"{category}_valid_points"] = valid_points
            total += points
            total_valid += valid_points

        row["total"] = total
        row["total_valid"] = total_valid
        rows.append(row)

    return pd.DataFrame(rows)


def prepare_gold(gold: pd.DataFrame) -> pd.DataFrame:
    """Add the rank and the percentile of each member within its category.

    ``gold`` holds the columns ``category``, ``member`` and ``typicality``.
    Rank 1 is the most typical member, and a percentile close to 1 is typical.
    """
    gold = gold.assign(
        category_norm=gold["category"].map(normalize_text),
        member_norm=gold["member"].map(normalize_text),
    )
    by_category = gold.groupby("category_norm")["typicality"]
    return gold.assign(
        typicality_rank=by_category.rank(ascending=False, method="min"),
        typicality_percentile=by_category.rank(pct=True),
    )


def score_answers_with_typicality(
    agent_answers: list[AgentAnswer],
    gold: pd.DataFrame,
    letter: str,
    categories: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score answers against the gold typicality of their category.

    An answer is valid when it is filled, starts with ``letter``, and is a
    member of its category in ``gold``. A valid answer carries the gold
    typicality with its rank and percentile. Returns one row per agent and
    category, and one summary row per agent. The penalized means count an
    invalid answer as 0.
    """
    gold = prepare_gold(gold)
    rows = []

    for agent_answer in agent_answers:
        for category in categories:
            raw_answer = agent_answer.answers.get(category, "")
            answer_norm = normalize_text(raw_answer)

            is_blank = answer_norm == ""
            starts_with_letter = not is_blank and answer_norm.startswith(letter.lower())
            match = gold[
                (gold["category_norm"] == normalize_text(category))
                & (gold["member_norm"] == answer_norm)
            ]
            in_gold_category = len(match) > 0
            is_valid = starts_with_letter and in_gold_category
            gold_row = match.iloc[0] if is_valid else None

            rows.append(
                {
                    "agent": agent_answer.agent_name,
                    "strategy": agent_answer.strategy,
                    "category": category,
                    "answer": raw_answer,
                    "answer_norm": answer_norm,
                    "is_blank": is_blank,
                    "starts_with_letter": starts_with_letter,
                    "in_gold_category": in_gold_category,
                    "is_valid": is_valid,
                    **{
                        column: float("nan") if gold_row is None else gold_row[column]
                        for column in (
                            "typicality",
                            "typicality_rank",
                            "typicality_percentile",
                        )
                    },
                }
            )

    item_level_df = pd.DataFrame(rows)

    agent_summary_df = item_level_df.groupby(["agent", "strategy"], as_index=False).agg(
        n_categories=("category", "count"),
        n_valid=("is_valid", "sum"),
        n_blank=("is_blank", "sum"),
        n_wrong_letter=("starts_with_letter", lambda x: (~x).sum()),
        n_not_in_gold=("in_gold_category", lambda x: (~x).sum()),
        mean_typicality=("typicality", "mean"),
        median_typicality=("typicality", "median"),
        mean_typicality_percentile=("typicality_percentile", "mean"),
        mean_typicality_rank=("typicality_rank", "mean"),
    )
    agent_summary_df["validity_rate"] = (
        agent_summary_df["n_valid"] / agent_summary_df["n_categories"]
    )

    penalized = (
        item_level_df.assign(
            penalized_typicality=lambda df: df["typicality"].fillna(0),
            penalized_percentile=lambda df: df["typicality_percentile"].fillna(0),
        )
        .groupby(["agent", "strategy"], as_index=False)
        .agg(
            penalized_mean_typicality=("penalized_typicality", "mean"),
            penalized_mean_percentile=("penalized_percentile", "mean"),
        )
    )

    return item_level_df, agent_summary_df.merge(
        penalized, on=["agent", "strategy"], how="left"
    )
