from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal

import nltk
from huggingface_hub import login

nltk.download("punkt")
nltk.download('punkt_tab')

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from components.prompts import build_prompt

STRATEGY: Literal["prototypical", "peripheral", "frequency_based", "random"]
STRATEGIES = {
    "prototypical": (
        "Choose the most typical, central, and obvious valid answer for the given category."
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
    """
    Stores the output produced by one agent during one round.

    answers is a dictionary where:
    - key = category name, e.g. "fruit", "mammal", "tool"
    - value = agent's answer for that category
    """

    agent_name: str
    answers: dict[str, str]
    notes: str = ""


class Model(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs): ...


class HFModel(Model):
    def __init__(self, model_name: str, hf_token: str, quantize: bool = True):
        login(hf_token)

        self.model_name = model_name

        quantization_config = None
        if quantize:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
        )

        self.model = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=30,
            max_length=None,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            return_full_text=False,
        )

    def generate(self, prompt: str, **kwargs):
        return self.model(prompt, return_full_text=False, **kwargs)[0]["generated_text"]


class OpenAIModel(Model):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-5.4",
    ):
        super().__init__()
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, **kwargs):
        return (
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                **kwargs
            )
            .choices[0]
            .message.content
        )


class LLMGeneratorAgent:
    def __init__(
        self,
        name: str,
        llm,
        slots: list[str],
        strategy: str | None = None,
        language: str = "English",
    ):
        self.name = name
        self.strategy = strategy
        self.llm = llm
        self.slots = slots
        self.language = language

    # TODO: support multiple_rounds
    def build_prompt(
        self,
        game_round: "components.game.GameRound",  # noqa: F821
        slot: str
    ) -> str:
        strategy_instruction = ""
        if self.strategy is not None:
            strategy_instruction = STRATEGIES.get(
                self.strategy, "Choose valid answers that follow all rules."
            )
            strategy_instruction = f"Your strategy is:\n {strategy_instruction}"

        return build_prompt(
            slot=slot,
            letter=game_round.letter,
            strategy_instructions=strategy_instruction,
            language=self.language,
        ).strip()

    # Only take the first word
    def parse_output(self, text: str) -> str:
        words = nltk.word_tokenize(text.strip())
        generated_word = ''
        if len(words):
            generated_word = words[0].strip().lower()
            generated_word = "".join(ch for ch in generated_word if ch.isalpha())
        return generated_word

    def play(
        self, game_round: "components.game.GameRound"  # noqa: F821
    ) -> AgentAnswer:
        generated_answers = {}
        for slot in tqdm(self.slots, desc=f"Agent {self.name} is generating..."):
            prompt = self.build_prompt(
                game_round=game_round, slot=slot
            )

            output = self.llm.generate(prompt)
            answer = self.parse_output(output)
            generated_answers[slot] = answer

        return AgentAnswer(
            agent_name=self.name,
            answers=generated_answers,
            notes=f"Strategy: {self.strategy}" if self.strategy else "",
        )
