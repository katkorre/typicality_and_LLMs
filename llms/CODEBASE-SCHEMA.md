# Codebase schema

## run.py

`main` builds the registry over the project root, then calls `select_keys`
and `Registry.from_key(key).run()` for every selected key. `select_keys` calls
`Registry.retrieve_keys` and `Registry.retrieve_configuration`. The module
depends on `configurations.keys`.

## configurations

`keys` declares one key per model through `cinnamon.registry.Namespace`, and
`keys.GAMES` maps each model tag to its key. `scattergories.LlamaConfig`,
`GemmaConfig`, `GPTConfig`, `ClaudeConfig` and `RandomConfig` register one key each and bind it to
`components.game.Scattergories`. They share the parameters of
`ScattergoriesConfig`, whose `model_post_init` adds the condition that limits
`num_rounds` to games with sampling.

## components.game

`Scattergories.run` instantiates one class of `components.models.BACKENDS`
with the dataset path,
then calls `load_gold`, `Scattergories.letter_sequence` and `Scattergories.play_game` for
every unfinished game.
`Scattergories.play_game` reads back finished rounds through its local
`record`, reseeds the model every round, and calls `build_agents`, `components.agent.Agent.play`,
`components.scoring.score_answers` and
`components.scoring.score_answers_with_typicality`. The module depends on
pandas.

## components.agent

`Agent.play` calls `Agent.build_prompt`, the `generate` method of its model,
and `parse_output`. `Agent.build_prompt` reads `components.prompts.CATEGORY_NAMES`, `STRATEGY_HEADER`,
`STRATEGIES`, `SCORES` and `USED_WORDS`, then calls
`components.prompts.build_prompt`. `components.game.build_agents` reads
`components.prompts.STRATEGIES`.

## components.analysis

`main` calls `summarize`, which reads the `scores.csv` files of every finished
game. The module depends on pandas.

## components.models

`HFModel`, `OpenAIModel`, `ClaudeModel` and `RandomModel` subclass the abstract
`ScattergoriesModel`, which declares `reseed` and `generate`. The first three
import transformers and torch, openai, and anthropic respectively, inside
their methods. `HFModel.__init__` calls `needs_bos`. `OpenAIModel.generate` and
`ClaudeModel.generate` call `report_stops`. `RandomModel` reads the dataset
with pandas.

## components.scoring

`score_answers_with_typicality` calls `prepare_gold`, which calls
`normalize_text`. `score_answers` depends on pandas only.

## tests

`tests/test_game.py` calls `parse_output`, `load_gold`, `score_answers`,
`score_answers_with_typicality`, `Scattergories.play_game`, `Scattergories.run` and
`components.analysis.summarize`, with a scripted model where a game is played. It also builds the registry and every registered key.
