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

from components.agent import STRATEGIES, Agent
from components.models import BACKENDS
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
    llm, slots: list[str], language: str, use_strategies: bool, num_agents: int
):
    if use_strategies:
        return [
            Agent(f"{strategy}_agent", llm, slots, language, strategy=strategy)
            for strategy in STRATEGIES
        ]
    return [Agent(f"agent_{idx}", llm, slots, language) for idx in range(num_agents)]


def play_game(
    llm,
    gold: pd.DataFrame,
    slots: list[str],
    language: str,
    use_strategies: bool,
    num_agents: int,
    letters: str,
    history: bool,
    save_path: Path,
):
    """Play one round per letter, and write each round under ``save_path``.

    With ``history``, every agent receives the cumulative points of all agents
    from the second round on. ``game_info.txt`` is written last, so its
    presence marks a finished game.
    """
    agents = build_agents(llm, slots, language, use_strategies, num_agents)
    instances = set(gold["member"].str.strip().str.lower())

    game_info = (
        f"Rounds: {len(letters)}\n"
        f"Language: {language}\n"
        f"Agents: {len(agents)}\n"
        f"Strategies enabled: {use_strategies}\n"
        f"History: {history}\n"
        f"Letters: {letters}\n"
        f"Categories: {len(slots)}\n"
        f"Valid words: {len(instances)}\n"
    )
    print(game_info, flush=True)

    totals = {agent.name: 0 for agent in agents}
    used: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for round_idx, letter in enumerate(letters, start=1):
        print(f"Round #{round_idx} - Letter {letter}", flush=True)
        used_words = {slot: used[letter, slot] for slot in slots}
        answers, outputs = [], []
        for agent in agents:
            scores = None
            if history and round_idx > 1:
                others = [p for name, p in totals.items() if name != agent.name]
                scores = (totals[agent.name], others)
            answer, raw = agent.play(
                letter,
                scores,
                {slot: tuple(sorted(words)) for slot, words in used_words.items()},
            )
            answers.append(answer)
            outputs += [
                {"agent": agent.name, "category": slot, "output": out}
                for slot, out in zip(slots, raw)
            ]

        round_path = save_path / f"round_{round_idx}"
        round_path.mkdir(parents=True, exist_ok=True)
        round_path.joinpath("round_info.txt").write_text(
            f"Round #{round_idx} - Letter {letter}"
        )
        pd.DataFrame(outputs).to_csv(round_path / "outputs.csv", index=False)
        scores_df = score_answers(
            answers,
            slots=slots,
            instances=instances,
            round_letter=letter.lower(),
            used_words=used_words,
        )
        scores_df.to_csv(round_path / "scores.csv", index=False)
        for agent_name, points in zip(scores_df["agent"], scores_df["total"]):
            totals[agent_name] += int(points)
        for answer in answers:
            for slot, word in answer.answers.items():
                if word.startswith(letter.lower()):
                    used[letter, slot].add(word)
        typicality, summary = score_answers_with_typicality(
            answers, gold=gold, letter=letter, slots=slots
        )
        typicality.to_csv(round_path / "typicality_scores.csv", index=False)
        summary.to_csv(round_path / "summary.csv", index=False)

    save_path.joinpath("game_info.txt").write_text(game_info)


class Scattergories:
    """Play every language and seed with one model, in one round setting.

    The model loads once. A finished game is skipped, so a run that stops
    continues where it stopped when started again. Each game reseeds the model
    first, so its outcome does not depend on the games before it.
    """

    def __init__(
        self,
        dataset_path: str,
        results_dir: str,
        languages: list[str],
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
        self.languages = languages
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

    def run(self):
        strategy_dir = "strategies" if self.use_strategies else "no_strategies"
        games = [
            (
                language,
                seed,
                self.results_dir
                / self.model_tag
                / language
                / strategy_dir
                / self.rounds
                / f"seed_{seed}",
            )
            for language in self.languages
            for seed in self.seeds
        ]
        pending = [game for game in games if not (game[2] / "game_info.txt").exists()]
        done = len(games) - len(pending)
        print(f"{len(games)} games, {done} already played", flush=True)
        if not pending:
            return

        llm = BACKENDS[self.backend](**self.model_args)
        print(f"Model loaded: {self.model_args['model_name']}", flush=True)

        for language, seed, save_path in pending:
            gold = load_gold(self.dataset_path, language)
            slots = gold["category"].unique().tolist()[: self.limit]
            llm.reseed(seed)
            play_game(
                llm,
                gold=gold,
                slots=slots,
                language=language,
                use_strategies=self.use_strategies,
                num_agents=self.num_agents,
                letters=self.letter_sequence(seed),
                history=self.rounds != "without_history",
                save_path=save_path,
            )
            print(f"Game written to {save_path}", flush=True)
