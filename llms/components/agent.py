import re
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


# One to three words. A word starts with a letter or digit and may hold
# hyphens, apostrophes, ampersands and periods, as in ``No. 2`` or ``M&M's``.
WORD = r"[^\W_](?:[^\W_]|['&.-])*"
COMPOUND = rf"{WORD}(?: {WORD}){{0,2}}"
# A compound, optionally after a label such as the category name and a colon.
LABELLED_COMPOUND = re.compile(rf"(?:{COMPOUND}: *)?({COMPOUND})")


def normalize_word(token: str) -> str:
    """Keep the letters, digits and inner hyphens of a word, lowercased."""
    return "".join(ch for ch in token if ch.isalnum() or ch == "-").strip("-").lower()


def parse_output(text: str) -> str:
    """Return the answer in a model output.

    When the first line of the output holds one to three words and nothing
    else, the answer is that whole line, so ``Ice cream`` reads as
    ``ice cream`` and ``Queen Mary 2`` as ``queen mary 2``. A label before a
    colon is dropped, so ``fruit: Fuji`` reads as ``fuji``. Otherwise the
    answer is the first word of the output, and tokens without a letter or
    digit, such as an opening quote, are skipped. Each word keeps its
    letters, digits and inner hyphens, so ``T-shirt.`` reads as ``t-shirt``,
    as in the gold members.
    """
    lines = text.strip().splitlines() or [""]
    line = " ".join(lines[0].strip().strip("\"'").split())
    match = LABELLED_COMPOUND.fullmatch(line)
    if match:
        tokens = match.group(1).split()
    else:
        tokens = [next((t for t in text.split() if normalize_word(t)), "")]
    return " ".join(w for w in map(normalize_word, tokens) if w)


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
