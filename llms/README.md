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
the category in its own language, for example `Möbel` for `furniture`. The
English name stays the key of the gold members and of the result files.

## Games

Each registered configuration is one experimental condition: a model, a
language, the use of strategies, and a round setting. The model is a tag of
the key, and the other three are variants of a configuration parameter.

A game is either without or with strategies. A game
without strategies has four agents with the same prompt, so their answers
differ only through sampling. A game with strategies has one agent per
strategy: prototypical, peripheral, frequency based and neutral. The neutral
strategy asks for a member without preferring more or less representative
ones, so it controls for the presence of a strategy. All agents of
a game share one model.

A game plays its rounds in one of three settings.

1. In `without_history`, the rounds are independent. An agent receives no
   information about previous rounds. A game plays one round per letter, in
   alphabetical order.
2. In `with_history`, each agent receives its cumulative game points and the
   points of the other agents from the second round on. A game plays one round
   per letter, in an order that the seed permutes.
3. In `with_history_sampling`, each agent receives the same points, and each
   round draws its letter at random. The parameter `num_rounds` sets the number
   of rounds, 26 or 52, so a letter can return. A word that any agent used in an
   earlier round with the same letter and category scores 0 points. The
   prompt lists these words for its category.

A blank answer, an answer with another initial, and a used word score 0
points. The other answers are compared only with each other. An answer shared
with another agent scores 5 points. The only answer of a category that scores
points gets 20 points, and any other unique answer gets 10 points. A shared
used word therefore scores 0 points for every agent that gives it. Games played
before this rule counted every filled answer, so their `scores.csv` can give
10 points where the current rule gives 20.

The prompt of a game without history equals the prompt of earlier versions.

The command `uv run python -m components.analysis` reports the points of every
agent over the finished games. A row is a model, a strategy setting, a round
setting and an agent, and a column is a language. A game scores the sum of its
rounds. A cell holds the mean and the standard deviation over seeds, followed
by the number of seeds. The option `--points valid`
reports the valid points instead.

When the first line of an output holds one to three words and nothing else,
the whole line is the answer, so `Ice cream` reads as `ice cream`. A label
before a colon is dropped, so `fruit: Fuji` reads as `fuji`. Otherwise
the first word of the output is the answer, and the parser skips tokens
without a letter. In both cases the parser keeps only letters inside each word
and lowercases the answer. An output that explains its choice, such as
`Ivory soap is used, so I choose Irrigator`, therefore reads as `ivory`.

A configuration plays one game per seed, and the seeds are 42, 2026 and 1337.
The directory of a game is
`results/scattergories/<model>/<language>/<strategies or no_strategies>/<setting>/seed_<seed>`,
with one subdirectory per round. A game with sampling names its setting
`with_history_sampling_26` or `with_history_sampling_52`.

Each round directory holds four tables. The file `outputs.csv` holds
the raw output of every agent and category. The file `scores.csv` holds the
game points, together with the valid points of the answers that are members of
their own category. The file `typicality_scores.csv` checks each answer against the
members of its own category. It reports their typicality, rank and
percentile. The file `summary.csv` aggregates these values per agent.

The file `game_info.txt` is written when a game ends. It records the model, the
round setting, the seed and the letters. A run skips every game that has this
file. A round is finished once its `summary.csv` exists, and an unfinished game
resumes at its first unfinished round. Every round reseeds the model from the
seed and the round index, so a resumed game samples like an uninterrupted one.
Each game reseeds the Hugging Face models first. The seed also fixes the
letters of a game, so every model plays the same letters for a given seed. The
API models take no seed, so their answers do not repeat exactly.

## Running on the cluster

The cluster runs the Hugging Face models inside an Apptainer image built from
`cluster/env.def`. Clone the repository on scratch, then submit every job from
the `llms` directory. The first argument of `cluster/run.sbatch` names the
model, and the other arguments are the filters of `run.py`. Two jobs must not
select the same configuration, since both would write the same game.

Both models are gated. Accept their licences on Hugging Face, then store a
token before the build with `HF_HOME=$SCRATCH/hf_home hf auth login`.
`SCRATCH` defaults to `/scratch.hpc/${USER}` and holds the Apptainer cache and
the Hugging Face cache. The build downloads the models once, and the runs work
offline. The models run in bfloat16. Gemma needs about 24 GB of GPU memory.

## Debugging locally

The model `random` plays without a language model, so a game runs in seconds on
any machine. It serves only for debugging, and its games go to `results/debug`.
Each answer is a dataset member, in any language, that starts with the letter
of the round. The seed fixes the answers. Every round setting is available.

```bash
uv run python run.py --model random
uv run python -m components.analysis --results-dir results/debug
```

## Running the API models

GPT and Claude need no GPU and run outside the cluster, through
[uv](https://docs.astral.sh/uv/), with `uv sync --extra api`. A model sends one
request per agent, category and round, one after another. An English game of
26 rounds therefore sends 5,512 requests. The clients retry a failed request
eight times. The log counts the answers that ended for another reason than a
normal stop, such as a refusal or the token limit, since these read as blanks.

The OpenAI client reads `OPENAI_API_KEY`, and `OPENAI_BASE_URL` for an
OpenAI-compatible endpoint. The configuration pins the reasoning effort of
GPT-5.4 to `none`, its documented default, because GPT-5.4 is reported to
accept a temperature only without reasoning. The Claude client reads `ANTHROPIC_API_KEY`, or
the profile that `ant auth login` stores. Claude Opus 5.5 accepts no
temperature and always thinks before it answers. The configuration sets its
effort to `low`, which keeps the thinking short. A request that Claude refuses
returns no text, and the game scores it as a blank answer.

## Reproducing the experiments

Every model plays 24 configurations: 3 languages, 2 strategy settings and 4
round settings. Each configuration plays 3 seeds. On the cluster, run from the
`llms` directory:

```bash
mkdir -p logs
sbatch cluster/build.sbatch
sbatch cluster/run.sbatch llama --sanity-check
sbatch cluster/run.sbatch gemma --sanity-check
for model in llama gemma; do
  sbatch cluster/run.sbatch "$model" --rounds without_history
  sbatch cluster/run.sbatch "$model" --rounds with_history
  sbatch cluster/run.sbatch "$model" --rounds with_history_sampling --num-rounds 26
  sbatch cluster/run.sbatch "$model" --rounds with_history_sampling --num-rounds 52
done
```

The build runs once, before the first job. The sanity checks play every
configuration with 3 categories into `results/sanity_check`. Each of the eight
jobs plays one round setting of one model, so the jobs never share a game. A
job that reaches its time limit continues when the same command is submitted
again.

The API models play one seed and the three settings of 26 rounds, which keeps
their cost bounded. They run on a local machine, from the `llms` directory:

```bash
uv sync --extra api
uv run python run.py --model gpt claude --seeds 42 --sanity-check
uv run python run.py --model gpt claude --seeds 42 --num-rounds 26 \
    --rounds without_history with_history with_history_sampling
```

After every game has finished, the summary reports the points per agent:

```bash
uv run python -m components.analysis
uv run python -m components.analysis --points valid
```

## Running locally

A local run of a Hugging Face model needs a GPU with enough memory for it:

```bash
uv sync --extra gpu --extra dev
uv run python run.py --model llama --sanity-check
uv run pytest -q
```

## Configuration

The models are registered with
[cinnamon](https://github.com/nlp-unibo/cinnamon). Each key has the name
`scattergories`, the namespace `typicality`, and the model as its first tag. A
variant adds a tag, such as `language=german` or `rounds=with_history`. A
default value adds no tag, so the key of English games without strategies and
without history carries only the model tag.

The script `run.py` plays every configuration that matches its filters. The
option `--model` is required. The options `--language`, `--rounds`,
`--use-strategies` and `--num-rounds` each take one or more values. An omitted
option matches every value. A key search finds the configurations of each
model. The filters then compare their values, so a default value is selected
like any other.

A condition excludes 52 rounds outside `with_history_sampling`, so those
combinations are never registered. The option `--list` prints the matching keys without playing them.

The file `configurations/scattergories.py` sets the dataset, the language,
the use of strategies, the round setting, the number of agents, the letters
and the seeds. It also sets each model's name and decoding. The Hugging Face
models always sample, with temperature 0.8 and top-p 0.9 by default. Sampling
is what distinguishes the agents of a game without strategies, which share one
prompt. GPT samples with temperature 0.9. Every model receives the
prompt as one user message through its chat template.
