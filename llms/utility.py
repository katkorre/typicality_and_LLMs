from __future__ import annotations

import re
from typing import Any

import pandas as pd

from components.agent import AgentAnswer


def normalize_answer(value: str) -> str:
    """Normalize answers before comparing duplicates."""
    return str(value).strip().lower()


def score_answers(
        agent_answers: list[AgentAnswer],
        slots: list[str],
        instances: list[str]
) -> pd.DataFrame:
    """
    Score a round according to the rules.

    For each category:
    - blank / missing answer: 0 points
    - same answer as another agent: 5 points
    - unique valid answer: 10 points
    - only filled answer while all other agents are blank: 20 points
    """
    rows = []

    for answer in agent_answers:
        row: dict[str, Any] = {"agent": answer.agent_name}
        total = 0
        total_valid = 0

        for slot in slots:
            raw_value = answer.answers.get(slot, "")
            value = normalize_answer(raw_value)

            filled_values = [
                normalize_answer(a.answers.get(slot, ""))
                for a in agent_answers
                if normalize_answer(a.answers.get(slot, ""))
            ]

            if not value:
                points = 0
            elif filled_values.count(value) > 1:
                points = 5
            elif len(filled_values) == 1:
                points = 20
            else:
                points = 10

            valid_points = points if value in instances else 0

            row[slot] = raw_value
            row[f"{slot}_points"] = points
            row[f"{slot}_valid_points"] = valid_points
            total += points
            total_valid += valid_points

        row["total"] = total
        row['total_valid'] = total_valid
        rows.append(row)

    return pd.DataFrame(rows)


def normalize_text(value: str) -> str:
    """
    Normalize text for matching model answers against the gold dataset.
    """
    value = str(value).strip().lower()
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^[\"']|[\"']$", "", value)
    return value


def prepare_gold(gold_df: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    Prepare gold dataframe for matching.
    """
    gold = gold_df.copy()

    gold["category_norm"] = gold["category"].apply(normalize_text)
    gold["member_norm"] = gold[f"norm_rating_{language}"].apply(normalize_text)

    # Rank within each category: 1 = most prototypical
    gold["typicality_rank"] = gold.groupby("category_norm")["typicality_score"].rank(
        ascending=False, method="min"
    )

    # Percentile within category: closer to 1 = more prototypical
    gold["typicality_percentile"] = gold.groupby("category_norm")[
        "typicality_score"
    ].rank(pct=True)

    return gold


def score_answers_with_typicality(
        agent_answers: list[AgentAnswer],
        gold_df: pd.DataFrame,
        game_round: 'components.game.GameRound',
        slots: list[str],
        language: str = 'English'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate agent answers using:
    - starting-letter validity
    - category membership in gold data
    - gold typicality score
    - category-relative typicality rank and percentile

    Returns:
    - item_level_df: one row per agent/category answer
    - agent_summary_df: aggregated scores per agent
    """

    gold = prepare_gold(gold_df, language=language)

    rows = []

    for agent_answer in agent_answers:
        for category in slots:
            raw_answer = agent_answer.answers.get(category, "")
            answer_norm = normalize_text(raw_answer)
            category_norm = normalize_text(category)

            is_blank = answer_norm == ""
            starts_with_letter = not is_blank and answer_norm.startswith(
                game_round.letter.lower()
            )

            match = gold[
                (gold["category_norm"] == category_norm)
                & (gold["member_norm"] == answer_norm)
                ]

            in_gold_category = len(match) > 0

            if in_gold_category:
                gold_row = match.iloc[0]
                typicality_score = gold_row["typicality_score"]
                typicality_rank = gold_row["typicality_rank"]
                typicality_percentile = gold_row["typicality_percentile"]
                nratings = gold_row["Nratings"]
            else:
                typicality_score = None
                typicality_rank = None
                typicality_percentile = None
                nratings = None

            is_valid = not is_blank and starts_with_letter and in_gold_category

            rows.append(
                {
                    "agent": agent_answer.agent_name,
                    "strategy": agent_answer.notes.replace("Strategy: ", ""),
                    "category": category,
                    "answer": raw_answer,
                    "answer_norm": answer_norm,
                    "is_blank": is_blank,
                    "starts_with_letter": starts_with_letter,
                    "in_gold_category": in_gold_category,
                    "is_valid": is_valid,
                    "typicality_score": typicality_score if is_valid else None,
                    "typicality_rank": typicality_rank if is_valid else None,
                    "typicality_percentile": typicality_percentile
                    if is_valid
                    else None,
                    "Nratings": nratings if is_valid else None,
                }
            )

    item_level_df = pd.DataFrame(rows)

    agent_summary_df = item_level_df.groupby(["agent", "strategy"], as_index=False).agg(
        n_categories=("category", "count"),
        n_valid=("is_valid", "sum"),
        n_blank=("is_blank", "sum"),
        n_wrong_letter=("starts_with_letter", lambda x: (~x).sum()),
        n_not_in_gold=("in_gold_category", lambda x: (~x).sum()),
        mean_typicality=("typicality_score", "mean"),
        median_typicality=("typicality_score", "median"),
        mean_typicality_percentile=("typicality_percentile", "mean"),
        mean_typicality_rank=("typicality_rank", "mean"),
    )

    agent_summary_df["validity_rate"] = (
            agent_summary_df["n_valid"] / agent_summary_df["n_categories"]
    )

    # Penalized score: invalid or blank answers count as 0
    penalized = (
        item_level_df.assign(
            penalized_typicality=lambda df: df["typicality_score"].fillna(0),
            penalized_percentile=lambda df: df["typicality_percentile"].fillna(0),
        )
        .groupby(["agent", "strategy"], as_index=False)
        .agg(
            penalized_mean_typicality=("penalized_typicality", "mean"),
            penalized_mean_percentile=("penalized_percentile", "mean"),
        )
    )

    agent_summary_df = agent_summary_df.merge(
        penalized, on=["agent", "strategy"], how="left"
    )

    return item_level_df, agent_summary_df
