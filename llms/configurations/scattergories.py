from typing import Annotated, List, Literal, Set

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_class
from pydantic import Field

from configurations.keys import CLAUDE, GEMMA, GPT, LLAMA, RANDOM

Language = Literal["english", "german", "spanish"]
Letter = Annotated[str, Field(pattern=r"^[a-z]$")]
Rounds = Literal["without_history", "with_history", "with_history_sampling"]


class ScattergoriesConfig(Configuration):
    dataset_path: str = Param(
        "data/typicalities_dataset.csv",
        description="CSV file with the category members and their gold typicality.",
    )
    results_dir: str = Param(
        "results/scattergories",
        description="A game writes under "
        "results_dir/<model tag>/<language>/<strategies or no_strategies>/"
        "<rounds>/seed_<seed>. A with_history_sampling game appends its number of "
        "rounds to <rounds>.",
    )
    language: Language = Param(
        "english",
        variants=["german", "spanish"],
        description="Language of the prompts and of the gold members.",
    )
    use_strategies: bool = Param(
        False,
        description="Give each agent its own strategy, one agent per strategy.",
        variants=[True],
    )
    rounds: Rounds = Param(
        "without_history",
        variants=["with_history", "with_history_sampling"],
        description="Round setting. without_history plays independent rounds, one "
        "per letter in alphabetical order. with_history also gives each agent "
        "the cumulative points of every agent, one round per letter in a seeded "
        "order. with_history_sampling draws the letter of each round at random, "
        "and a word already used with the same letter and category scores 0.",
    )
    num_rounds: int = Param(
        26,
        ge=1,
        variants=[52],
        description="Rounds of a with_history_sampling game. "
        "The other settings play one round per letter.",
    )
    num_agents: int = Param(
        4,
        ge=2,
        description="Agents in a game without strategies. "
        "A game with strategies has one agent per strategy.",
    )
    letters: Set[Letter] = Param(
        set("abcdefghijklmnopqrstuvwxyz"),
        min_length=1,
        description="Lowercase letters to play. "
        "A game without sampling plays one round per letter.",
    )
    seeds: List[int] = Param(
        [42, 2026, 1337],
        min_length=1,
        description="Seeds to play, one game per seed and language. A seed sets the "
        "letter order and the sampling of Hugging Face models. API models take no "
        "seed.",
    )
    limit: int | None = Param(
        None,
        ge=1,
        description="Play only the first limit categories, for a sanity check. "
        "None plays every category.",
    )

    backend: str = Param(description="One of components.models.BACKENDS.")
    model_tag: str = Param(description="Names the results directory of the model.")
    model_name: str = Param(description="Model identifier passed to the backend.")
    temperature: float | None = Param(
        None,
        gt=0,
        description="Sampling temperature. "
        "None sends no temperature, which some API models require.",
    )
    max_new_tokens: int = Param(30, ge=1, description="Token budget of each answer.")
    top_p: float = Param(
        0.9,
        gt=0,
        le=1,
        description="Nucleus sampling threshold of a Hugging Face model.",
    )
    batch_size: int = Param(
        32,
        ge=1,
        description="Prompts generated together by a Hugging Face model.",
    )
    base_url: str | None = Param(
        None,
        description="A custom OpenAI-compatible endpoint. "
        "None reads OPENAI_BASE_URL, or falls back to the OpenAI API.",
    )
    reasoning_effort: str | None = Param(
        None,
        description="Reasoning effort of an OpenAI model. None sends no effort.",
    )
    effort: str | None = Param(
        None,
        description="Claude's effort level. Claude Opus 5.5 always thinks, "
        "so effort is its only control over reasoning depth.",
    )

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        self.add_condition(
            name="num_rounds_only_with_sampling",
            condition=lambda config: (
                config.rounds == "with_history_sampling" or config.num_rounds == 26
            ),
            description="Only a with_history_sampling game has a free number of "
            "rounds; the other settings play one round per letter",
        )


@register_class(key=LLAMA, component="components.game.Scattergories")
class LlamaConfig(ScattergoriesConfig):
    backend: str = Param("hf")
    model_tag: str = Param("llama")
    model_name: str = Param("meta-llama/Llama-3.1-8B-Instruct")
    temperature: float | None = Param(0.8, gt=0)


@register_class(key=GEMMA, component="components.game.Scattergories")
class GemmaConfig(ScattergoriesConfig):
    backend: str = Param("hf")
    model_tag: str = Param("gemma")
    model_name: str = Param("google/gemma-3-12b-it")
    temperature: float | None = Param(0.8, gt=0)


@register_class(key=GPT, component="components.game.Scattergories")
class GPTConfig(ScattergoriesConfig):
    backend: str = Param("openai")
    model_tag: str = Param("gpt")
    model_name: str = Param("gpt-5.4")
    temperature: float | None = Param(0.9, gt=0)
    reasoning_effort: str | None = Param(
        "none",
        description="GPT-5.4 is reported to accept a temperature only without "
        "reasoning, so the effort is pinned to none, its documented default.",
    )


@register_class(key=CLAUDE, component="components.game.Scattergories")
class ClaudeConfig(ScattergoriesConfig):
    backend: str = Param("anthropic")
    model_tag: str = Param("claude")
    model_name: str = Param("claude-opus-5-5")
    max_new_tokens: int = Param(
        4000,
        ge=1,
        description="Token budget of each answer, "
        "including the adaptive thinking that precedes it.",
    )
    effort: str | None = Param("low")


@register_class(key=RANDOM, component="components.game.Scattergories")
class RandomConfig(ScattergoriesConfig):
    results_dir: str = Param(
        "results/debug",
        description="Games of the random model, kept apart from the experiments.",
    )
    backend: str = Param("random")
    model_tag: str = Param("random")
    model_name: str = Param("random")
