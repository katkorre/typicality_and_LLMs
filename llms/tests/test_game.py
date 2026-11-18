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
        self.prompts = []

    def reseed(self, seed):
        pass

    def generate(self, prompts):
        self.prompts += prompts
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
    assert scores.loc["a", "fruit_points"] == 20
    assert scores.loc["b", "fruit_points"] == 0
    assert scores.loc["a", "fruit_valid_points"] == 0
    assert scores.loc["b", "total_valid"] == 25


def test_a_used_word_scores_zero_even_when_shared():
    answers = [
        AgentAnswer("a", {"animal": "cat"}),
        AgentAnswer("b", {"animal": "cat"}),
        AgentAnswer("c", {"animal": "cow"}),
    ]
    scores = score_answers(
        answers, ["animal"], {"cat", "cow"}, "c", used_words={"animal": {"cat"}}
    ).set_index("agent")
    assert scores.loc["a", "animal_points"] == 0
    assert scores.loc["b", "animal_points"] == 0
    assert scores.loc["c", "animal_points"] == 20


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
        history=False,
        save_path=tmp_path,
    )
    for name in ("outputs", "scores", "typicality_scores", "summary"):
        assert (tmp_path / "round_1" / f"{name}.csv").exists()
    scores = pd.read_csv(tmp_path / "round_1" / "scores.csv")
    assert scores["animal_points"].tolist() == [10, 10]
    assert scores["fruit_points"].tolist() == [5, 5]
    assert (tmp_path / "round_2" / "scores.csv").exists()
    assert (tmp_path / "game_info.txt").exists()


def test_history_gives_the_scores_and_the_used_words(tmp_path):
    gold = load_gold(DATASET, "english")
    model = ScriptedModel(["Cat", "Cow", "Cat", "Crab"])
    play_game(
        model,
        gold=gold,
        slots=["animal"],
        language="english",
        use_strategies=False,
        num_agents=2,
        letters="cc",
        history=True,
        save_path=tmp_path,
    )
    assert "Your score" not in model.prompts[0]
    assert "Your score after the previous rounds: 10 points." in model.prompts[2]
    assert "Scores of the other agents: 10 points." in model.prompts[2]
    assert "in previous rounds: cat, cow." in model.prompts[2]
    scores = pd.read_csv(tmp_path / "round_2" / "scores.csv")
    assert scores["animal_points"].tolist() == [0, 20]


@pytest.mark.parametrize("rounds", ["without_history", "with_history"])
def test_every_letter_is_played_once(rounds):
    game = Scattergories(
        dataset_path=DATASET,
        results_dir="unused",
        languages=["english"],
        use_strategies=False,
        rounds=rounds,
        num_rounds=5,
        num_agents=2,
        letters=set("hgfedcba"),
        seeds=[1],
        limit=None,
        backend="hf",
        model_tag="llama",
    )
    assert sorted(game.letter_sequence(1)) == list("abcdefgh")
    if rounds == "without_history":
        assert game.letter_sequence(1) == "abcdefgh"
    assert game.letter_sequence(1) == game.letter_sequence(1)
    if rounds == "with_history":
        assert game.letter_sequence(1) != game.letter_sequence(2)
    game.rounds = "with_history_sampling"
    assert len(game.letter_sequence(1)) == 5


def test_a_finished_game_is_skipped(tmp_path, monkeypatch, capsys):
    for seed in (1, 2):
        path = tmp_path / "llama" / "english" / "strategies" / "with_history"
        path = path / f"seed_{seed}"
        path.mkdir(parents=True)
        (path / "game_info.txt").write_text("done")

    def refuse(**kwargs):
        raise AssertionError("the model loads although every game is played")

    monkeypatch.setitem(models.BACKENDS, "hf", refuse)
    Scattergories(
        dataset_path=DATASET,
        results_dir=str(tmp_path),
        languages=["english"],
        use_strategies=True,
        rounds="with_history",
        num_rounds=26,
        num_agents=4,
        letters={"c"},
        seeds=[1, 2],
        limit=None,
        backend="hf",
        model_tag="llama",
        model_name="unused",
    ).run()
    assert "2 already played" in capsys.readouterr().out


def test_every_model_is_registered():
    from cinnamon.registry import Registry

    from configurations.keys import GAMES

    valid, invalid = Registry.build(directory=Path.cwd())
    assert not invalid, sorted(str(key) for key in invalid)
    for model, key in GAMES.items():
        for use_strategies in (False, True):
            if use_strategies:
                key = key.from_variant({"use_strategies": True})
            game = Registry.from_key(key)
            assert isinstance(game, Scattergories) and game.model_tag == model
            assert game.use_strategies is use_strategies


def test_summary_averages_the_finished_games_over_seeds(tmp_path):
    from components.analysis import format_cells, summarize

    for seed, points, finished in ((1, 10, True), (2, 30, True), (3, 99, False)):
        game = tmp_path / "llama" / "english" / "strategies" / "with_history"
        game = game / f"seed_{seed}"
        for round_idx in (1, 2):
            (game / f"round_{round_idx}").mkdir(parents=True)
            pd.DataFrame(
                {"agent": ["a", "b"], "total": [points, 0], "total_valid": [5, 0]}
            ).to_csv(game / f"round_{round_idx}" / "scores.csv", index=False)
        if finished:
            (game / "game_info.txt").write_text("done")
    summary = summarize(tmp_path)
    row = ("llama", "strategies", "with_history", "a")
    assert summary.loc[row, ("mean", "english")] == 40
    assert summary.loc[row, ("std", "english")] == pytest.approx(28.28, abs=0.01)
    assert summary.loc[row, ("count", "english")] == 2
    assert summarize(tmp_path, "valid").loc[row, ("mean", "english")] == 10
    assert format_cells(summary).loc[row, "english"] == "40.0 ± 28.3 (2)"
