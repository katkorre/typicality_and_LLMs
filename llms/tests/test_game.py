from pathlib import Path

import pandas as pd
import pytest

from components import models
from components.agent import Agent, AgentAnswer, parse_output
from components.game import Scattergories, load_gold
from components.scoring import score_answers, score_answers_with_typicality

DATASET = "data/typicalities_dataset.csv"


class ScriptedModel(models.ScattergoriesModel):
    """Answers every prompt with the next scripted output."""

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def reseed(self, seed):
        pass

    def generate(self, prompts):
        self.prompts += prompts
        return [next(self.outputs) for _ in prompts]


def make_game(**overrides) -> Scattergories:
    """Build a game with small defaults, changed by ``overrides``."""
    params = dict(
        dataset_path=DATASET,
        results_dir="unused",
        language="english",
        use_strategies=False,
        rounds="without_history",
        num_rounds=5,
        num_agents=2,
        letters=set("abc"),
        seeds=[1],
        limit=None,
        backend="hf",
        model_tag="llama",
        model_name="unused",
    )
    return Scattergories(**params | overrides)


def test_parse_output_reads_the_first_word():
    assert parse_output("Cat.") == "cat"
    assert parse_output('"Bär" ist ein Tier') == "bär"
    assert parse_output("t-shirt") == "tshirt"
    assert parse_output("  ") == ""
    assert parse_output('" cat') == "cat"


def test_prompts_name_the_category_in_their_language():
    german = Agent("a", None, ["furniture"], "german").build_prompt("t", "furniture")
    assert "Kategorie Möbel" in german and "furniture" not in german
    spanish = Agent("a", None, ["body part"], "spanish").build_prompt("p", "body part")
    assert "categoría parte del cuerpo" in spanish


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
    answers = [AgentAnswer("a", {"animal": "cow", "tool": "cat"}, "neutral")]
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
    categories = ["animal", "fruit"]
    make_game().play_game(
        ScriptedModel(["Cat", "Cherry", "Cow", "Cherry"] * 2),
        gold=gold,
        categories=categories,
        letters="cb",
        seed=1,
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
    make_game(rounds="with_history").play_game(
        model,
        gold=gold,
        categories=["animal"],
        letters="cc",
        seed=1,
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
    game = make_game(rounds=rounds, letters=set("hgfedcba"))
    assert sorted(game.letter_sequence(1)) == list("abcdefgh")
    if rounds == "without_history":
        assert game.letter_sequence(1) == "abcdefgh"
    assert game.letter_sequence(1) == game.letter_sequence(1)
    if rounds == "with_history":
        assert game.letter_sequence(1) != game.letter_sequence(2)
    game.rounds = "with_history_sampling"
    assert len(game.letter_sequence(1)) == 5


def test_random_model_answers_with_the_letter_of_the_prompt():
    model = models.RandomModel(DATASET)
    prompts = [
        Agent("a", None, ["animal"], language, None).build_prompt(letter, "animal")
        for language in ("english", "german", "spanish")
        for letter in "cbx"
    ]
    model.reseed(1)
    first = model.generate(prompts)
    model.reseed(1)
    assert model.generate(prompts) == first
    assert all(not word or word[0] == "cbx"[i % 3] for i, word in enumerate(first))
    assert all(word in model.words for word in first if word)


class TemplateTokenizer:
    """Writes ``prefix`` before every chat prompt."""

    def __init__(self, bos_token, prefix):
        self.bos_token = bos_token
        self.prefix = prefix

    def apply_chat_template(self, messages, **kwargs):
        return self.prefix + messages[0]["content"]


def test_bos_is_added_only_when_the_template_lacks_it():
    assert not models.needs_bos(TemplateTokenizer("<bos>", "<bos><user>"))
    assert models.needs_bos(TemplateTokenizer("<bos>", "<user>"))
    assert not models.needs_bos(TemplateTokenizer(None, "<user>"))


def test_a_random_game_runs_end_to_end(tmp_path):
    make_game(
        results_dir=str(tmp_path),
        language="spanish",
        use_strategies=True,
        rounds="with_history_sampling",
        num_rounds=4,
        seeds=[1, 2],
        backend="random",
        model_tag="random",
    ).run()
    games = sorted(tmp_path.glob("random/spanish/strategies/*/seed_*/game_info.txt"))
    assert len(games) == 2
    assert len(list(games[0].parent.glob("round_*/scores.csv"))) == 4


def test_a_finished_game_is_skipped(tmp_path, monkeypatch, capsys):
    for seed in (1, 2):
        path = tmp_path / "llama" / "english" / "strategies" / "with_history"
        path = path / f"seed_{seed}"
        path.mkdir(parents=True)
        (path / "game_info.txt").write_text("done")

    def refuse(**kwargs):
        raise AssertionError("the model loads although every game is played")

    monkeypatch.setitem(models.BACKENDS, "hf", refuse)
    make_game(
        results_dir=str(tmp_path),
        use_strategies=True,
        rounds="with_history",
        seeds=[1, 2],
    ).run()
    assert "2 already played" in capsys.readouterr().out


def test_every_condition_is_registered_and_selectable():
    from cinnamon.registry import Registry

    from configurations.keys import GAMES
    from run import select_keys

    valid, invalid = Registry.build(directory=Path.cwd())
    assert len(invalid) == len(GAMES) * 3 * 2 * 2
    assert all(
        "num_rounds=52" in key.tags and "rounds=with_history_sampling" not in key.tags
        for key in invalid
    )
    no_filter = dict.fromkeys(["language", "rounds", "use_strategies", "num_rounds"])
    for model in GAMES:
        keys = select_keys([model], no_filter)
        assert len(keys) == 3 * 2 * 4
        game = Registry.from_key(keys[0])
        assert isinstance(game, Scattergories) and game.model_tag == model

    defaults = {
        "language": ["english"],
        "rounds": ["without_history"],
        "use_strategies": ["false"],
        "num_rounds": ["26"],
    }
    assert select_keys(["llama"], defaults) == [GAMES["llama"]]
    german = select_keys(["llama", "gemma"], no_filter | {"language": ["german"]})
    assert len(german) == 2 * 2 * 4
    assert all(Registry.from_key(key).language == "german" for key in german)


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


def test_unexpected_stops_are_reported(capsys):
    from collections import Counter

    models.report_stops(Counter({"end_turn": 5}), expected="end_turn")
    assert capsys.readouterr().out == ""
    models.report_stops(Counter({"end_turn": 3, "max_tokens": 2}), expected="end_turn")
    assert "{'max_tokens': 2}" in capsys.readouterr().out


def test_an_interrupted_game_resumes_exactly(tmp_path):
    import shutil

    def play(results_dir):
        make_game(
            results_dir=str(results_dir),
            language="spanish",
            use_strategies=True,
            rounds="with_history_sampling",
            num_rounds=6,
            backend="random",
            model_tag="random",
        ).run()
        return results_dir / "random/spanish/strategies/with_history_sampling_6/seed_1"

    full = play(tmp_path / "full")
    resumed = play(tmp_path / "resumed")
    (resumed / "game_info.txt").unlink()
    for round_idx in (4, 5, 6):
        shutil.rmtree(resumed / f"round_{round_idx}")
    (resumed / "round_3" / "summary.csv").unlink()
    play(tmp_path / "resumed")

    for name in ("outputs", "scores", "summary"):
        for round_idx in range(1, 7):
            path = f"round_{round_idx}/{name}.csv"
            assert (full / path).read_text() == (resumed / path).read_text(), path


def fake_sdk(monkeypatch, module, client_class, create):
    """Install a fake SDK ``module`` whose client records its kwargs."""
    import sys
    import types
    from types import SimpleNamespace

    created = {}

    def make_client(**kwargs):
        created.update(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            messages=SimpleNamespace(create=create),
        )

    monkeypatch.setitem(
        sys.modules, module, types.SimpleNamespace(**{client_class: make_client})
    )
    return created


def test_openai_model_sends_the_sampling_and_reports_stops(monkeypatch, capsys):
    from types import SimpleNamespace

    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        reason = "length" if len(requests) == 2 else "stop"
        message = SimpleNamespace(content=None if reason == "length" else "Cat")
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=reason, message=message)]
        )

    created = fake_sdk(monkeypatch, "openai", "OpenAI", create)
    model = models.OpenAIModel(
        model_name="gpt", temperature=0.9, base_url=None, reasoning_effort="none"
    )
    assert created["max_retries"] == models.API_RETRIES
    assert model.generate(["a", "b"]) == ["Cat", ""]
    assert requests[0]["temperature"] == 0.9
    assert requests[0]["reasoning_effort"] == "none"
    assert "{'length': 1}" in capsys.readouterr().out

    requests.clear()
    models.OpenAIModel(
        model_name="gpt", temperature=None, base_url=None, reasoning_effort=None
    ).generate(["a"])
    assert "temperature" not in requests[0] and "reasoning_effort" not in requests[0]


def test_claude_model_keeps_the_text_and_reports_stops(monkeypatch, capsys):
    from types import SimpleNamespace

    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        if len(requests) == 2:
            return SimpleNamespace(stop_reason="refusal", content=[])
        content = [
            SimpleNamespace(type="thinking", text="ignored"),
            SimpleNamespace(type="text", text="Cow"),
        ]
        return SimpleNamespace(stop_reason="end_turn", content=content)

    created = fake_sdk(monkeypatch, "anthropic", "Anthropic", create)
    model = models.ClaudeModel(model_name="claude", max_new_tokens=100, effort="low")
    assert created["max_retries"] == models.API_RETRIES
    assert model.generate(["a", "b"]) == ["Cow", ""]
    assert requests[0]["output_config"] == {"effort": "low"}
    assert "temperature" not in requests[0]
    assert "{'refusal': 1}" in capsys.readouterr().out

    requests.clear()
    models.ClaudeModel(model_name="claude", max_new_tokens=100, effort=None).generate(
        ["a"]
    )
    assert "output_config" not in requests[0]


def test_run_applies_the_filters_and_overrides(monkeypatch, capsys):
    import sys

    import run

    played = []

    class Game:
        def __init__(self, key, overrides):
            self.key, self.overrides = key, overrides

        def run(self):
            played.append(self)

    monkeypatch.setattr(
        run.Registry, "from_key", lambda key, **overrides: Game(key, overrides)
    )
    argv = ["run.py", "--model", "random", "--language", "spanish", "--rounds"]
    argv += ["with_history", "--use-strategies", "false", "--seeds", "7", "8"]
    monkeypatch.setattr(sys, "argv", argv + ["--sanity-check"])
    run.main()
    assert len(played) == 1
    assert played[0].overrides == {
        "seeds": [7, 8],
        "limit": 3,
        "results_dir": "results/sanity_check",
    }

    played.clear()
    monkeypatch.setattr(sys, "argv", ["run.py", "--model", "random", "--list"])
    run.main()
    assert not played
    assert "24 matching configurations" in capsys.readouterr().out


def test_analysis_needs_a_finished_game(tmp_path, monkeypatch, capsys):
    import sys

    from components import analysis

    with pytest.raises(FileNotFoundError):
        analysis.summarize(tmp_path)
    game = tmp_path / "random/english/no_strategies/without_history/seed_1"
    (game / "round_1").mkdir(parents=True)
    pd.DataFrame({"agent": ["a"], "total": [5], "total_valid": [0]}).to_csv(
        game / "round_1/scores.csv", index=False
    )
    (game / "game_info.txt").write_text("done")
    monkeypatch.setattr(sys, "argv", ["analysis", "--results-dir", str(tmp_path)])
    analysis.main()
    assert "5.0 ± - (1)" in capsys.readouterr().out
