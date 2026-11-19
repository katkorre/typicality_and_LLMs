from dataclasses import dataclass

from components.models import ScattergoriesModel
from components.prompts import (
    CATEGORY_NAMES,
    SCORES,
    STRATEGIES,
    STRATEGY_HEADER,
    USED_WORDS,
    build_prompt,
)


@dataclass
class AgentAnswer:
    """The answers of one agent in one round, by category."""

    agent_name: str
    answers: dict[str, str]
    strategy: str = ""


def parse_output(text: str) -> str:
    """Return the first word of a model output, lowercased and letters only.

    Tokens without a letter, such as an opening quote, are skipped. A word is
    split on whitespace, so ``ice cream`` reads as ``ice``.
    """
    for token in text.split():
        word = "".join(ch for ch in token if ch.isalpha())
        if word:
            return word.lower()
    return ""


class Agent:
    def __init__(
        self,
        name: str,
        llm: ScattergoriesModel,
        categories: list[str],
        language: str,
        strategy: str | None = None,
    ):
        self.name = name
        self.llm = llm
        self.categories = categories
        self.language = language
        self.strategy = strategy

    def build_prompt(
        self,
        letter: str,
        category: str,
        scores: tuple[int, list[int]] | None = None,
        used_words: tuple[str, ...] = (),
    ) -> str:
        """Fill the prompt of one category.

        ``scores`` holds the points of this agent and of the other agents after
        the previous rounds. ``used_words`` lists the words that previous rounds
        with this letter used for this category.
        """
        language = self.language.capitalize()
        name = CATEGORY_NAMES.get(language, {}).get(category, category)
        blocks = []
        if self.strategy is not None:
            blocks.append(
                f"{STRATEGY_HEADER[language]}\n{STRATEGIES[self.strategy][language]}"
            )
        if scores is not None:
            own, others = scores
            blocks.append(
                SCORES[language].format(own=own, others=", ".join(map(str, others)))
            )
        if used_words:
            blocks.append(
                USED_WORDS[language].format(
                    category=name, letter=letter, words=", ".join(used_words)
                )
            )
        return build_prompt(
            category=name,
            letter=letter,
            instructions="\n\n".join(blocks),
            language=self.language,
        ).strip()

    def play(
        self,
        letter: str,
        scores: tuple[int, list[int]] | None = None,
        used_words: dict[str, tuple[str, ...]] | None = None,
    ) -> tuple[AgentAnswer, list[str]]:
        """Answer every category, and return the raw outputs alongside."""
        used_words = used_words or {}
        outputs = self.llm.generate(
            [
                self.build_prompt(
                    letter, category, scores, used_words.get(category, ())
                )
                for category in self.categories
            ]
        )
        answer = AgentAnswer(
            agent_name=self.name,
            answers={
                category: parse_output(out)
                for category, out in zip(self.categories, outputs)
            },
            strategy=self.strategy or "",
        )
        return answer, outputs
