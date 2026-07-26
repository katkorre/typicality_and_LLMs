from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from lightning.pytorch import seed_everything

from components.agent import HFModel, OpenAIModel
from components.game import Game

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model',
                        '-m',
                        default='gemma',
                        type=str,
                        nargs='?',
                        choices=['mistral', 'llama', 'phi', 'qwen', 'gemma', 'gpt'])
    parser.add_argument('--seed',
                        '-s',
                        default=42,
                        type=int)
    parser.add_argument('--num_agents',
                        '-n',
                        type=int,
                        default=4)
    parser.add_argument('--rounds',
                        '-r',
                        type=int,
                        default=1)
    parser.add_argument('--strategy',
                        '-str',
                        type=int,
                        default=0)
    parser.add_argument('--language',
                        '-l',
                        default='english',
                        type=str,
                        nargs='?',
                        choices=['english', 'german', 'spanish'])
    parser.add_argument('--base_url',
                        '-bl',
                        default="https://api.openai.com/v1",
                        type=str)
    parser.add_argument('--api_key',
                        '-ak',
                        type=str)
    parser.add_argument('--hf_token',
                        '-t',
                        type=str)
    args = parser.parse_args()
    args.strategy = bool(args.strategy)

    seed_everything(args.seed)

    base_path = Path(__file__).parent.parent.resolve()
    data_path = base_path / "data" / "combined_prototypes.csv"
    df = pd.read_csv(data_path)

    instances: list[str] = [item.lower() for item in df[f'instance_{args.language.capitalize()}'].tolist()
                            if isinstance(item, str)]
    slots = df["category"].unique().tolist()

    print(f"All the categories for typicality in this dataset are: {slots}")

    model_map = {
        'mistral': "mistralai/Mistral-7B-Instruct-v0.3",
        'llama': "meta-llama/Meta-Llama-3.1-8B-Instruct",
        'phi': "microsoft/Phi-3-mini-4k-instruct",
        'qwen': "Qwen/Qwen2.5-7B-Instruct",
        'gemma': "google/gemma-3-12b-it",
        'gpt': "gpt-5.4"
    }
    if args.model != 'gpt':
        llm = HFModel(model_map[args.model], hf_token=args.hf_token)
    else:
        llm = OpenAIModel(model_name=model_map[args.model],
                          api_key=args.api_key,
                          base_url=args.base_url)

    print(f"✅ Instruction LLM loaded: {model_map[args.model]}")

    game = Game(llm=llm,
                slots=slots,
                instances=instances,
                use_strategies=args.strategy,
                num_agents=args.num_agents,
                rounds=args.rounds,
                language=args.language)
    save_path = base_path.joinpath("results",
                                   args.model,
                                   args.language,
                                   "strategies" if args.strategy else 'no_strategies')
    if not save_path.exists():
        save_path.mkdir(parents=True)

    game.play(df=df, save_path=save_path)
