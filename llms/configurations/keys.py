from cinnamon.registry import Namespace

NAMESPACE = Namespace("typicality")

LLAMA = NAMESPACE.key("scattergories", "llama")
GEMMA = NAMESPACE.key("scattergories", "gemma")
GPT = NAMESPACE.key("scattergories", "gpt")
CLAUDE = NAMESPACE.key("scattergories", "claude")

#: The game key of each model, by the tag that registers it.
GAMES = {"llama": LLAMA, "gemma": GEMMA, "gpt": GPT, "claude": CLAUDE}
