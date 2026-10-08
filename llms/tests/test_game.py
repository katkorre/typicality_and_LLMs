from pathlib import Path

import pandas as pd
import pytest

from components import models
from components.agent import AgentAnswer, parse_output
from components.game import Scattergories, load_gold, play_game
from components.scoring import score_answers, score_answers_with_typicality

DATASET = "data/typicalities_dataset.csv"


class ScriptedModel:
    """Answers every prompt with the next scripted output."""

    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def reseed(self, seed):
        pass

    def generate(self, prompts):
        return [next(self.outputs) for _ in prompts]


def test_parse_output_reads_the_first_word():
    assert parse_output("Cat.") == "cat"
    assert parse_output('"Bär" ist ein Tier') == "bär"
    assert parse_output("t-shirt") == "tshirt"
    assert parse_output("  ") == ""


def test_load_gold_keeps_the_members_of_one_language():
    gold = load_gold(DATASET, "german")
    assert list(gold.columns) == ["category", "member", "typicality"]
    assert gold["member"].notna().all()
    assert "Bär" in set(gold["member"])
    # Categories without German members are not played in German.
    assert "profession" in set(gold["category"])
    assert "jewelry" not in set(gold["category"])


def test_score_answers_follows_the_rules():
    answers = [
        AgentAnswer("a", {"animal": "cat", "tool": "", "fruit": "cherry"}),
        AgentAnswer("b", {"animal": "cat", "tool": "chisel", "fruit": "dog"}),
    ]
    scores = score_answers(
        answers, ["animal", "tool", "fruit"], {"cat", "chisel"}, "c"
    ).set_index("agent")
    assert scores.loc["a", "animal_points"] == 5
    assert scores.loc["b", "tool_points"] == 20
    assert scores.loc["a", "fruit_points"] == 10
    assert scores.loc["b", "fruit_points"] == 0
    assert scores.loc["a", "fruit_valid_points"] == 0
    assert scores.loc["b", "total_valid"] == 25


def test_typicality_is_read_within_the_category():
    gold = pd.DataFrame(
        {
            "category": ["animal", "animal", "tool"],
            "member": ["Cat", "Cow", "Chisel"],
            "typicality": [0.9, 0.5, 0.7],
        }
    )
    answers = [AgentAnswer("a", {"animal": "cow", "tool": "cat"}, "random")]
    items, summary = score_answers_with_typicality(
        answers, gold, "C", ["animal", "tool"]
    )
    items = items.set_index("category")
    assert items.loc["animal", "is_valid"]
    assert items.loc["animal", "typicality_rank"] == 2
    assert not items.loc["tool", "in_gold_category"]
    assert summary.loc[0, "penalized_mean_typicality"] == pytest.approx(0.25)


def test_play_game_writes_every_round(tmp_path):
    gold = load_gold(DATASET, "english")
    slots = ["animal", "fruit"]
    play_game(
        ScriptedModel(["Cat", "Cherry", "Cow", "Cherry"] * 2),
        gold=gold,
        slots=slots,
        language="english",
        use_strategies=False,
        num_agents=2,
        letters="cb",
        save_path=tmp_path,
    )
    for name in ("outputs", "scores", "typicality_scores", "summary"):
        assert (tmp_path / "round_1" / f"{name}.csv").exists()
    scores = pd.read_csv(tmp_path / "round_1" / "scores.csv")
    assert scores["animal_points"].tolist() == [10, 10]
    assert scores["fruit_points"].tolist() == [5, 5]
    assert (tmp_path / "round_2" / "scores.csv").exists()
    assert (tmp_path / "game_info.txt").exists()


def test_a_finished_game_is_skipped(tmp_path, monkeypatch, capsys):
    for arm in ("no_strategies", "strategies"):
        path = tmp_path / "llama" / "english" / arm
        path.mkdir(parents=True)
        (path / "game_info.txt").write_text("done")

    def refuse(**kwargs):
        raise AssertionError("the model loads although every game is played")

    monkeypatch.setitem(models.BACKENDS, "hf", refuse)
    Scattergories(
        dataset_path=DATASET,
        results_dir=str(tmp_path),
        languages=["english"],
        strategy_arms=[False, True],
        num_agents=4,
        letters="c",
        seed=42,
        limit=None,
        backend="hf",
        model_tag="llama",
        model_name="unused",
    ).run()
    assert "2 already played" in capsys.readouterr().out


def test_every_model_is_registered():
    from cinnamon.registry import Registry

    from configurations.keys import MODELS, scattergories

    valid, invalid = Registry.build(directory=Path.cwd())
    assert not invalid, sorted(str(key) for key in invalid)
    for model in MODELS:
        game = Registry.from_key(scattergories(model))
        assert isinstance(game, Scattergories) and game.model_tag == model
