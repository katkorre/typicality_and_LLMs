from dataclasses import dataclass

from components.prompts import SCORES, USED_WORDS, build_prompt

STRATEGIES = {
    "prototypical": (
        "Choose the most typical, central, and obvious valid answer for the given "
        "category."
    ),
    "peripheral": (
        "Choose valid but less typical or more marginal category members. "
        "Avoid the most obvious answers."
    ),
    "frequency_based": (
        "Choose valid answers that are common, familiar, and frequently encountered "
        "in everyday language and experience."
    ),
    "random": "Choose randomly among valid candidates.",
}


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
        llm,
        slots: list[str],
        language: str,
        strategy: str | None = None,
    ):
        self.name = name
        self.llm = llm
        self.slots = slots
        self.language = language
        self.strategy = strategy

    def build_prompt(
        self,
        letter: str,
        slot: str,
        scores: tuple[int, list[int]] | None = None,
        used_words: tuple[str, ...] = (),
    ) -> str:
        """Fill the prompt of one category.

        ``scores`` holds the points of this agent and of the other agents after
        the previous rounds. ``used_words`` lists the words that previous rounds
        with this letter used for this category.
        """
        language = self.language.capitalize()
        blocks = []
        if self.strategy is not None:
            blocks.append(f"Your strategy is:\n {STRATEGIES[self.strategy]}")
        if scores is not None:
            own, others = scores
            blocks.append(SCORES[language].format(own, ", ".join(map(str, others))))
        if used_words:
            blocks.append(
                USED_WORDS[language].format(slot, letter, ", ".join(used_words))
            )
        return build_prompt(
            slot=slot,
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
                self.build_prompt(letter, slot, scores, used_words.get(slot, ()))
                for slot in self.slots
            ]
        )
        answer = AgentAnswer(
            agent_name=self.name,
            answers={slot: parse_output(out) for slot, out in zip(self.slots, outputs)},
            strategy=self.strategy or "",
        )
        return answer, outputs
