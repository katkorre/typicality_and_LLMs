from dataclasses import dataclass

from components.prompts import build_prompt

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

    def build_prompt(self, letter: str, slot: str) -> str:
        strategy_instruction = ""
        if self.strategy is not None:
            strategy_instruction = f"Your strategy is:\n {STRATEGIES[self.strategy]}"
        return build_prompt(
            slot=slot,
            letter=letter,
            strategy_instructions=strategy_instruction,
            language=self.language,
        ).strip()

    def play(self, letter: str) -> tuple[AgentAnswer, list[str]]:
        """Answer every category, and return the raw outputs alongside."""
        outputs = self.llm.generate(
            [self.build_prompt(letter=letter, slot=slot) for slot in self.slots]
        )
        answer = AgentAnswer(
            agent_name=self.name,
            answers={slot: parse_output(out) for slot, out in zip(self.slots, outputs)},
            strategy=self.strategy or "",
        )
        return answer, outputs
