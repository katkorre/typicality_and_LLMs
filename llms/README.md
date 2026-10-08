# Scattergories with LLMs

This project plays Scattergories with instruction models to probe category
typicality. In each round, every agent receives a letter and a list of
categories. It answers each category with one word that starts with the
letter. The models are Llama 3.1 8B Instruct, Gemma 3 12B Instruct, GPT 5.4
and Claude Opus 5.5. The games run in English, German and Spanish.

## Data

`data/typicalities_dataset.csv` lists category members with their typicality.
The `concept` column names the category in English. The `english`, `german`
and `spanish` columns hold a member in each language, and `typ_en`, `typ_de`
and `typ_es` hold its typicality. A language plays only the categories with
members in it: 53 in English, 11 in German and 5 in Spanish. The prompt names
the category in English in every language.

## Games

A model plays every language in two arms. The arm without strategies has four
agents with the same prompt, so their answers differ only through sampling.
The strategy arm has one agent per strategy: prototypical, peripheral,
frequency based and random. All agents of a game share one model.

The first word of an output is the answer. The parser skips tokens without a
letter, keeps only letters, and lowercases the word. An answer of two words,
such as `ice cream`, therefore reads as `ice` and matches no member.

A run writes one directory per game. The directory is
`results/scattergories/<model>/<language>/<arm>`, with one subdirectory per
round.

Each round directory holds four tables. The file `outputs.csv` holds
the raw output of every agent and category. The file `scores.csv` holds the
game points, together with the valid points of the answers that are members of
any category. The file `typicality_scores.csv` checks each answer against the
members of its own category. It reports their typicality, rank and
percentile. The file `summary.csv` aggregates these values per agent.

The file `game_info.txt` is written when a game ends. A run skips every game
that has this file, so a run that stops continues with the unfinished game.
Each game reseeds the Hugging Face models first. The API models take no seed, so their
games do not repeat exactly.

## Running on the cluster

The cluster runs the Hugging Face models inside an Apptainer image built from
`cluster/env.def`. Clone the repository on scratch, then submit every job from
the `llms` directory:

```bash
mkdir -p logs
sbatch cluster/build.sbatch                # image, registry check, model download
sbatch cluster/run.sbatch llama --smoke    # 3 categories into results/smoke
sbatch cluster/run.sbatch llama            # every game
sbatch cluster/run.sbatch gemma
```

Both models are gated. Accept their licences on Hugging Face, then store a
token before the build with `HF_HOME=$SCRATCH/hf_home hf auth login`.
`SCRATCH` defaults to `/scratch.hpc/${USER}` and holds the Apptainer cache and
the Hugging Face cache. The build downloads the models once, and the runs work
offline. The models run in bfloat16. Gemma needs about 24 GB of GPU memory.

## Running the API models

GPT and Claude need no GPU and run outside the cluster, through
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra api
uv run python run.py claude --smoke
uv run python run.py claude
uv run python run.py gpt
```

The OpenAI client reads `OPENAI_API_KEY`, and `OPENAI_BASE_URL` for an
OpenAI-compatible endpoint. The Claude client reads `ANTHROPIC_API_KEY`, or
the profile that `ant auth login` stores. Claude Opus 5.5 accepts no
temperature and always thinks before it answers. The configuration sets its
effort to `low`, which keeps the thinking short. A request that Claude refuses
returns no text, and the game scores it as a blank answer.

## Running locally

A local run of a Hugging Face model needs a GPU with enough memory for it:

```bash
uv sync --extra gpu --extra dev
uv run python run.py llama --smoke
uv run pytest -q
```

## Configuration

The models are registered with
[cinnamon](https://github.com/nlp-unibo/cinnamon). Each key has the name
`scattergories`, the namespace `typicality`, and the model as its only tag.
The file `configurations/scattergories.py` sets the dataset, the languages,
the arms, the number of agents, the letters and the seed. It also sets each
model's name and decoding. The Hugging Face models sample with temperature 0.8
and top-p 0.9, and GPT samples with temperature 0.9. Every model receives the
prompt as one user message through its chat template.
