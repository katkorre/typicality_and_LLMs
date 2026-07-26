import random
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightning import seed_everything

from components.agent import STRATEGIES, LLMGeneratorAgent, StackAgentAnswer
from utility import score_answers


@dataclass
class GameRound:
    """
    Represents one round of the game.

    A round constrains agents by:
    - letter: the first letter required for every category answer
    """

    letter: str


class Game:
    def __init__(
            self,
            llm,
            slots: list[str],
            instances: list[str],
            rounds: int = 1,
            use_strategies: bool = False,
            num_agents: int = 1,
            language: str = 'English'
    ):
        self.llm = llm
        self.slots = slots
        self.instances = instances
        self.language = language

        self.rounds = rounds
        self.use_strategies = use_strategies
        self.num_agents = num_agents

    def initialize_agents(self) -> list[LLMGeneratorAgent]:
        if not self.use_strategies:
            return [
                LLMGeneratorAgent(name=f"agent_{idx}",
                                  llm=self.llm,
                                  slots=self.slots,
                                  language=self.language)
                for idx in range(self.num_agents)
            ]

        return [
            LLMGeneratorAgent(
                name=f"{strategy}_agent",
                strategy=strategy,
                llm=self.llm,
                slots=self.slots,
                language=self.language
            )
            for strategy in STRATEGIES
        ]

    def play(self, df: pd.DataFrame, save_path: Path):
        agents = self.initialize_agents()
        agent_seed = int(np.random.randint(low=1, high=2 ** 31 - 1))
        seed_everything(agent_seed)

        print(f"""Game starts!
                Rounds: {self.rounds}
                Language: {self.language}
                Agents: {len(agents)}
                Strategies enabled: {self.use_strategies}
                Categories: {len(self.slots)}
                Valid words: {len(self.instances)}
        """)

        game_status = [StackAgentAnswer(answers={}, agent_name=agent.name) for agent in agents]
        for _ in range(self.rounds):
            print("Round #", _ + 1)

            letter = random.choice(string.ascii_uppercase)
            game_round = GameRound(letter=letter)

            answers = []
            for agent, agent_status in zip(agents, game_status):
                agent_answer = agent.play(game_round=game_round, agent_status=agent_status)
                agent_status.update(agent_answer)
                print(agent_answer)

            scores = score_answers(answers, slots=self.slots, instances=self.instances)
            # typicality_scores, summary_df = score_answers_with_typicality(
            #     agent_answers=answers,
            #     slots=self.slots,
            #     game_round=game_round,
            #     gold_df=df,
            # )

            round_path = save_path / f"round_{_ + 1}"
            if not round_path.exists():
                round_path.mkdir(parents=True, exist_ok=True)

            scores.to_csv(round_path / "scores.csv", index=None)
            # typicality_scores.to_csv(round_path / "typicality_scores.csv", index=None)
            # summary_df.to_csv(round_path / "summary_df.csv", index=None)

        print(f'Game ended! Check {save_path} for results.')
