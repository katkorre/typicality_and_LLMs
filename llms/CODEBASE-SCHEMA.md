# Codebase schema

## run.py

`main` builds the registry over the project root, then calls
`Registry.from_key(GAMES[model]).run()`. It depends on
`configurations.keys`.

## configurations

`keys` declares one key per model through `cinnamon.registry.Namespace`, and
`keys.GAMES` maps each model tag to its key. `scattergories.LlamaConfig`,
`GemmaConfig`, `GPTConfig` and `ClaudeConfig` register one key each and bind it to
`components.game.Scattergories`. They share the parameters of
`ScattergoriesConfig`.

## components.game

`Scattergories.run` instantiates one class of `components.models.BACKENDS`,
then calls `load_gold` and `play_game` for every unfinished game.
`play_game` calls `build_agents`, `components.agent.Agent.play`,
`components.scoring.score_answers` and
`components.scoring.score_answers_with_typicality`. The module depends on
pandas.

## components.agent

`Agent.play` calls `Agent.build_prompt`, the `generate` method of its model,
and `parse_output`. `Agent.build_prompt` calls
`components.prompts.build_prompt`.

## components.models

`HFModel`, `OpenAIModel` and `ClaudeModel` expose `reseed` and `generate`.
They import transformers and torch, openai, and anthropic respectively,
inside their methods.

## components.scoring

`score_answers_with_typicality` calls `prepare_gold`, which calls
`normalize_text`. `score_answers` depends on pandas only.

## tests

`tests/test_game.py` calls `parse_output`, `load_gold`, `score_answers`,
`score_answers_with_typicality`, `play_game` and `Scattergories.run` with a
scripted model. It also builds the registry and every registered key.
