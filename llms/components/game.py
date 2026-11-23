"""Scattergories games between agents that share one model.

A game gives every agent the same categories and one letter per round. Each
agent answers every category with one word starting with that letter. The
dataset lists the members of each category with their typicality, per
language. A language plays only the categories that hold members in it.

A game plays its rounds in one of three settings. Without history, every round
is independent and the letters follow alphabetical order. With history,
each agent also receives its cumulative points and those of the other agents,
and the letters follow a seeded permutation. With history and sampling, each
round draws its letter at random, so a letter can return. A word that any agent
used in an earlier round with the same letter and category then scores 0
points, and the prompt lists these words.
"""

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from components.agent import Agent, AgentAnswer
from components.models import BACKENDS, ScattergoriesModel
from components.prompts import STRATEGIES
from components.scoring import score_answers, score_answers_with_typicality

LANGUAGE_CODES = {"english": "en", "german": "de", "spanish": "es"}


def load_gold(dataset_path: str, language: str) -> pd.DataFrame:
    """Return the members of every category in ``language``, with typicality."""
    df = pd.read_csv(dataset_path, index_col=0)
    gold = df[["concept", language, f"typ_{LANGUAGE_CODES[language]}"]].dropna(
        subset=[language]
    )
    gold.columns = ["category", "member", "typicality"]
    return gold.reset_index(drop=True)


def build_agents(
    llm: ScattergoriesModel,
    categories: list[str],
    language: str,
    use_strategies: bool,
    num_agents: int,
):
    if use_strategies:
        return [
            Agent(f"{strategy}_agent", llm, categories, language, strategy=strategy)
            for strategy in STRATEGIES
        ]
    return [
        Agent(f"agent_{idx}", llm, categories, language) for idx in range(num_agents)
    ]


class Scattergories:
    """Play every seed of one language with one model, in one round setting.

    The model loads once. A finished game is skipped, and an unfinished game
    resumes at its first unfinished round, so a run that stops continues where
    it stopped when started again. Every round reseeds the model, so its outcome
    does not depend on the games or rounds played before it in the same run.
    """

    def __init__(
        self,
        dataset_path: str,
        results_dir: str,
        language: str,
        use_strategies: bool,
        rounds: str,
        num_rounds: int,
        num_agents: int,
        letters: set[str],
        seeds: list[int],
        limit: int | None,
        backend: str,
        model_tag: str,
        **model_args,
    ):
        self.dataset_path = dataset_path
        self.results_dir = Path(results_dir)
        self.language = language
        self.use_strategies = use_strategies
        self.rounds = rounds
        self.num_rounds = num_rounds
        self.num_agents = num_agents
        self.letters = "".join(sorted(letters))
        self.seeds = seeds
        self.limit = limit
        self.backend = backend
        self.model_tag = model_tag
        self.model_args = model_args

    def letter_sequence(self, seed: int) -> str:
        """Return the letter of every round of a game played with ``seed``."""
        rng = random.Random(seed)
        if self.rounds == "without_history":
            return self.letters
        if self.rounds == "with_history":
            return "".join(rng.sample(self.letters, len(self.letters)))
        return "".join(rng.choices(self.letters, k=self.num_rounds))

    def play_game(
        self,
        llm: ScattergoriesModel,
        gold: pd.DataFrame,
        categories: list[str],
        letters: str,
        seed: int,
        save_path: Path,
    ):
        """Play one round per letter, and write each round under ``save_path``.

        In a setting with history, every agent receives the cumulative points of
        all agents from the second round on. ``game_info.txt`` is written last, so
        its presence marks a finished game.

        A round is finished once its ``summary.csv`` exists. A finished round is
        read back instead of played, so an interrupted game resumes at its first
        unfinished round. Every round reseeds the model from ``seed`` and its
        index, so a resumed game samples like an uninterrupted one.
        """
        history = self.rounds != "without_history"
        agents = build_agents(
            llm, categories, self.language, self.use_strategies, self.num_agents
        )
        members = {
            category: set(words.str.strip().str.lower())
            for category, words in gold.groupby("category")["member"]
        }

        game_info = (
            f"Model: {self.model_args['model_name']}\n"
            f"Round setting: {self.rounds}\n"
            f"Seed: {seed}\n"
            f"Rounds: {len(letters)}\n"
            f"Language: {self.language}\n"
            f"Agents: {len(agents)}\n"
            f"Strategies enabled: {self.use_strategies}\n"
            f"History: {history}\n"
            f"Letters: {letters}\n"
            f"Categories: {len(categories)}\n"
            f"Valid words: {sum(len(words) for words in members.values())}\n"
        )
        print(game_info, flush=True)

        totals = {agent.name: 0 for agent in agents}
        used: defaultdict[tuple[str, str], set[str]] = defaultdict(set)

        def record(scores_df: pd.DataFrame, answers: list[AgentAnswer], letter: str):
            for agent_name, points in zip(scores_df["agent"], scores_df["total"]):
                totals[agent_name] += int(points)
            for answer in answers:
                for category, word in answer.answers.items():
                    if word.startswith(letter.lower()):
                        used[letter, category].add(word)

        for round_idx, letter in enumerate(letters, start=1):
            round_path = save_path / f"round_{round_idx}"
            if (round_path / "summary.csv").exists():
                print(f"Round #{round_idx} - Letter {letter}, already played")
                scores_df = pd.read_csv(
                    round_path / "scores.csv", keep_default_na=False, dtype=str
                )
                answers = [
                    AgentAnswer(row["agent"], {c: row[c] for c in categories})
                    for _, row in scores_df.iterrows()
                ]
                record(scores_df, answers, letter)
                continue

            print(f"Round #{round_idx} - Letter {letter}", flush=True)
            llm.reseed(seed * 100 + round_idx)
            used_words = {category: used[letter, category] for category in categories}
            answers, outputs = [], []
            for agent in agents:
                scores = None
                if history and round_idx > 1:
                    others = [
                        points for name, points in totals.items() if name != agent.name
                    ]
                    scores = (totals[agent.name], others)
                answer, raw = agent.play(
                    letter,
                    scores,
                    {
                        category: tuple(sorted(words))
                        for category, words in used_words.items()
                    },
                )
                answers.append(answer)
                outputs += [
                    {"agent": agent.name, "category": category, "output": out}
                    for category, out in zip(categories, raw)
                ]

            round_path.mkdir(parents=True, exist_ok=True)
            round_path.joinpath("round_info.txt").write_text(
                f"Round #{round_idx} - Letter {letter}"
            )
            pd.DataFrame(outputs).to_csv(round_path / "outputs.csv", index=False)
            scores_df = score_answers(
                answers,
                categories=categories,
                members=members,
                round_letter=letter.lower(),
                used_words=used_words,
            )
            scores_df.to_csv(round_path / "scores.csv", index=False)
            record(scores_df, answers, letter)
            typicality, summary = score_answers_with_typicality(
                answers, gold=gold, letter=letter, categories=categories
            )
            typicality.to_csv(round_path / "typicality_scores.csv", index=False)
            summary.to_csv(round_path / "summary.csv", index=False)

        save_path.joinpath("game_info.txt").write_text(game_info)

    def run(self):
        strategy_dir = "strategies" if self.use_strategies else "no_strategies"
        rounds_dir = self.rounds
        if self.rounds == "with_history_sampling":
            rounds_dir = f"{self.rounds}_{self.num_rounds}"
        game_dir = self.results_dir / self.model_tag / self.language
        game_dir = game_dir / strategy_dir / rounds_dir
        games = [(seed, game_dir / f"seed_{seed}") for seed in self.seeds]
        pending = [game for game in games if not (game[1] / "game_info.txt").exists()]
        done = len(games) - len(pending)
        print(f"{len(games)} games, {done} already played", flush=True)
        if not pending:
            return

        llm = BACKENDS[self.backend](dataset_path=self.dataset_path, **self.model_args)
        print(f"Model loaded: {self.model_args['model_name']}", flush=True)

        gold = load_gold(self.dataset_path, self.language)
        categories = gold["category"].unique().tolist()[: self.limit]
        for seed, save_path in pending:
            self.play_game(
                llm,
                gold=gold,
                categories=categories,
                letters=self.letter_sequence(seed),
                seed=seed,
                save_path=save_path,
            )
            print(f"Game written to {save_path}", flush=True)
