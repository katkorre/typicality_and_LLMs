from typing import List, Literal

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_class

from configurations.keys import NAMESPACE

Language = Literal["english", "german", "spanish"]


class ScattergoriesConfig(Configuration):
    dataset_path: str = Param("data/typicalities_dataset.csv")
    #: A game writes under ``results_dir/<model tag>/<language>/<strategy arm>``.
    results_dir: str = Param("results/scattergories")
    #: Every language is played in both arms, with and without strategies.
    languages: List[Language] = Param(["english", "german", "spanish"])
    strategy_arms: List[bool] = Param([False, True])
    #: Agents in the arm without strategies. The strategy arm has one agent per
    #: strategy.
    num_agents: int = Param(4, ge=1)
    #: One round per letter.
    letters: str = Param("c", min_length=1)
    seed: int = Param(42)
    #: Play only the first ``limit`` categories, for a smoke run.
    limit: int | None = Param(None, ge=1)

    #: One of ``components.models.BACKENDS``.
    backend: str
    #: Names the results directory of the model.
    model_tag: str
    model_name: str
    #: ``None`` sends no temperature, which some API models require.
    temperature: float | None = Param(None, gt=0)
    max_new_tokens: int = Param(30, ge=1)
    #: Prompts generated together by a Hugging Face model.
    batch_size: int = Param(32, ge=1)
    #: A custom OpenAI-compatible endpoint. ``None`` reads ``OPENAI_BASE_URL``,
    #: or falls back to the OpenAI API.
    base_url: str | None = Param(None)
    #: Claude's effort level. Claude Opus 5.5 always thinks, so effort is its
    #: only control over reasoning depth.
    effort: str | None = Param(None)


@register_class(
    name="scattergories",
    tags={"llama"},
    namespace=NAMESPACE,
    component="components.game.Scattergories",
)
class LlamaConfig(ScattergoriesConfig):
    backend: str = Param("hf")
    model_tag: str = Param("llama")
    model_name: str = Param("meta-llama/Llama-3.1-8B-Instruct")
    temperature: float | None = Param(0.8, gt=0)


@register_class(
    name="scattergories",
    tags={"gemma"},
    namespace=NAMESPACE,
    component="components.game.Scattergories",
)
class GemmaConfig(ScattergoriesConfig):
    backend: str = Param("hf")
    model_tag: str = Param("gemma")
    model_name: str = Param("google/gemma-3-12b-it")
    temperature: float | None = Param(0.8, gt=0)


@register_class(
    name="scattergories",
    tags={"gpt"},
    namespace=NAMESPACE,
    component="components.game.Scattergories",
)
class GPTConfig(ScattergoriesConfig):
    backend: str = Param("openai")
    model_tag: str = Param("gpt")
    model_name: str = Param("gpt-5.4")
    temperature: float | None = Param(0.9, gt=0)


@register_class(
    name="scattergories",
    tags={"claude"},
    namespace=NAMESPACE,
    component="components.game.Scattergories",
)
class ClaudeConfig(ScattergoriesConfig):
    backend: str = Param("anthropic")
    model_tag: str = Param("claude")
    model_name: str = Param("claude-opus-5-5")
    #: Covers the adaptive thinking that precedes the answer.
    max_new_tokens: int = Param(4000, ge=1)
    effort: str | None = Param("low")
