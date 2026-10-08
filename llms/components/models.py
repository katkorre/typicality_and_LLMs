"""Text generation behind one interface, whatever serves the model.

Every model receives the prompt as a single user message. A model generates a
list of prompts and returns one text per prompt, in order. Libraries are
imported inside each class, so a backend needs only its own dependencies.
"""


class HFModel:
    """A Hugging Face chat model on the local GPU, in bfloat16."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_new_tokens: int,
        batch_size: int,
        **kwargs,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Left padding keeps every prompt's last token at the end of its row,
        # which is where batched generation continues from.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", dtype=torch.bfloat16
        )
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

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
                for prompt in prompts[start:start + self.batch_size]
            ]
            # The chat template already starts with the BOS token.
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            outputs += self.tokenizer.batch_decode(
                generated[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            )
        return outputs


class OpenAIModel:
    """A model behind the OpenAI chat completions API.

    The client reads ``OPENAI_API_KEY``, and ``OPENAI_BASE_URL`` when
    ``base_url`` is ``None``.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float | None,
        base_url: str | None,
        **kwargs,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url)
        self.model_name = model_name
        self.temperature = temperature

    def reseed(self, seed: int):
        """The API takes no seed, so a game with this model does not repeat."""

    def generate(self, prompts: list[str]) -> list[str]:
        sampling = {} if self.temperature is None else {"temperature": self.temperature}
        return [
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **sampling,
            )
            .choices[0]
            .message.content
            or ""
            for prompt in prompts
        ]


class ClaudeModel:
    """A Claude model behind the Claude API.

    The client reads ``ANTHROPIC_API_KEY``, or the profile that
    ``ant auth login`` stores. Claude Opus 5.5 accepts no temperature, and its
    thinking cannot be disabled. A refused request answers with an empty text,
    which the game scores as a blank.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        effort: str | None,
        **kwargs,
    ):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.effort = effort

    def reseed(self, seed: int):
        """The API takes no seed, so a game with this model does not repeat."""

    def generate(self, prompts: list[str]) -> list[str]:
        config = {}
        if self.effort is not None:
            config["output_config"] = {"effort": self.effort}
        outputs = []
        for prompt in prompts:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_new_tokens,
                messages=[{"role": "user", "content": prompt}],
                **config,
            )
            outputs.append(
                "".join(b.text for b in response.content if b.type == "text")
            )
        return outputs


BACKENDS = {"hf": HFModel, "openai": OpenAIModel, "anthropic": ClaudeModel}
