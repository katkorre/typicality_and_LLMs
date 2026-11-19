"""Text generation behind one interface, whatever serves the model.

Every model receives the prompt as a single user message. A model generates a
list of prompts and returns one text per prompt, in order. Libraries are
imported inside each class, so a backend needs only its own dependencies.
"""

import random
import re
from abc import ABC, abstractmethod
from collections import Counter

import pandas as pd


class ScattergoriesModel(ABC):
    """A model that plays Scattergories.

    Every backend accepts arbitrary keyword arguments, since a game passes the
    parameters of every backend.
    """

    def reseed(self, seed: int):
        """Seed the sampling of the model. A model without a seed ignores it."""

    @abstractmethod
    def generate(self, prompts: list[str]) -> list[str]:
        """Return one text per prompt, in order."""


# A game sends thousands of requests, so a transient API error must not end it.
API_RETRIES = 8


def report_stops(stops: Counter, expected: str):
    """Print how many answers stopped for a reason other than ``expected``.

    A refused or truncated answer reads as a blank, so the log keeps the count.
    """
    unexpected = {reason: n for reason, n in stops.items() if reason != expected}
    if unexpected:
        print(f"Answers not ended by {expected}: {unexpected}", flush=True)


def needs_bos(tokenizer) -> bool:
    """Tell whether the tokenizer must add the BOS token to a chat prompt.

    A chat template that writes the BOS token of the tokenizer itself needs no
    second one. A tokenizer without a BOS token adds none.
    """
    if tokenizer.bos_token is None:
        return False
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=False
    )
    return not text.startswith(tokenizer.bos_token)


class HFModel(ScattergoriesModel):
    """A Hugging Face chat model on the local GPU, in bfloat16.

    Decoding always samples, since the agents of a game without strategies share
    one prompt and differ only through sampling.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_new_tokens: int,
        batch_size: int,
        top_p: float,
        **kwargs,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.add_bos = needs_bos(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", dtype=torch.bfloat16
        )
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p

    def reseed(self, seed: int):
        from transformers import set_seed

        set_seed(seed)

    def generate(self, prompts: list[str]) -> list[str]:
        import torch

        outputs = []
        for start in range(0, len(prompts), self.batch_size):
            texts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for prompt in prompts[start : start + self.batch_size]
            ]
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=self.add_bos,
            ).to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            outputs += self.tokenizer.batch_decode(
                generated[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )
        return outputs


class OpenAIModel(ScattergoriesModel):
    """A model behind the OpenAI chat completions API.

    The client reads ``OPENAI_API_KEY``, and ``OPENAI_BASE_URL`` when
    ``base_url`` is ``None``. The API takes no seed, so a game with this model
    does not repeat.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float | None,
        base_url: str | None,
        reasoning_effort: str | None,
        **kwargs,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, max_retries=API_RETRIES)
        self.model_name = model_name
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def generate(self, prompts: list[str]) -> list[str]:
        sampling = {} if self.temperature is None else {"temperature": self.temperature}
        if self.reasoning_effort is not None:
            sampling["reasoning_effort"] = self.reasoning_effort
        outputs, stops = [], Counter()
        for prompt in prompts:
            choice = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **sampling,
            ).choices[0]
            stops[choice.finish_reason] += 1
            outputs.append(choice.message.content or "")
        report_stops(stops, expected="stop")
        return outputs


class ClaudeModel(ScattergoriesModel):
    """A Claude model behind the Claude API.

    The client reads ``ANTHROPIC_API_KEY``, or the profile that
    ``ant auth login`` stores. Claude Opus 5.5 accepts no temperature, and its
    thinking cannot be disabled. A refused request answers with an empty text,
    which the game scores as a blank. The API takes no seed, so a game with
    this model does not repeat.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        effort: str | None,
        **kwargs,
    ):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=API_RETRIES)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.effort = effort

    def generate(self, prompts: list[str]) -> list[str]:
        config = {}
        if self.effort is not None:
            config["output_config"] = {"effort": self.effort}
        outputs, stops = [], Counter()
        for prompt in prompts:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_new_tokens,
                messages=[{"role": "user", "content": prompt}],
                **config,
            )
            stops[response.stop_reason] += 1
            outputs.append(
                "".join(b.text for b in response.content if b.type == "text")
            )
        report_stops(stops, expected="end_turn")
        return outputs


class RandomModel(ScattergoriesModel):
    """A local stand-in that plays without a language model.

    Each answer is a member of any category in the dataset, in any language,
    drawn uniformly among the members that start with the letter of the
    prompt. The letter is the first quoted single letter of the prompt, as in
    the rule ``The word must start with the letter "c"``. A letter without
    members gives a blank answer.
    """

    def __init__(self, dataset_path: str, **kwargs):
        df = pd.read_csv(dataset_path, index_col=0)
        members = pd.concat([df[column] for column in ("english", "german", "spanish")])
        self.words = sorted(set(members.dropna().str.strip().str.lower()))
        self.rng = random.Random()

    def reseed(self, seed: int):
        self.rng.seed(seed)

    def generate(self, prompts: list[str]) -> list[str]:
        outputs = []
        for prompt in prompts:
            letter = re.search(r'"(\w)"', prompt).group(1).lower()
            candidates = [word for word in self.words if word.startswith(letter)]
            outputs.append(self.rng.choice(candidates) if candidates else "")
        return outputs


BACKENDS = {
    "hf": HFModel,
    "openai": OpenAIModel,
    "anthropic": ClaudeModel,
    "random": RandomModel,
}
