from cinnamon.registry import RegistrationKey

NAMESPACE = "typicality"

#: The models a game runs, by the tag that registers each of them.
MODELS = ("llama", "gemma", "gpt", "claude")


def scattergories(model: str) -> RegistrationKey:
    return RegistrationKey(name="scattergories", tags={model}, namespace=NAMESPACE)
