# The prompts are experimental material, kept verbatim.
# ruff: noqa: E501
PROMPTS = {
    "English": """
You are an agent playing Scattergories.
Given category {category}, you have to generate one word that fits {category} and starts with letter {letter}.
Other agents are playing with you.
You get 0 points if you do not know a word.
You get 5 points if your generated word is shared with another agent.
You get 20 points if you are the only one generating a word for {category}.
In all the remaining cases, you get 10 points.

{instructions}

Rules:
1. Play to achieve the most points.
2. Generate only one word for the category {category}.
3. The word must start with the letter "{letter}".
4. The word must be a real example of the category {category}.
5. If no valid word exists, leave it blank.
6. Do not add explanations.
7. Do not use markdown.
8. Do not use JSON.
9. Do not make mistakes

Examples for letter B:
mammal: Bear
fastener: Bolt
home decor: Blanket
plant: Bamboo

The category is {category}
Word starting with {letter} for {category}:
""".strip(),

    "German": """Du bist ein Agent, der Scattergories spielt.
Gegeben die Kategorie {category}, musst du ein Wort generieren, das zu {category} passt und mit dem Buchstaben {letter} beginnt.
Andere Agenten spielen mit dir.
Du bekommst 0 Punkte, wenn du kein Wort kennst.
Du bekommst 5 Punkte, wenn dein generiertes Wort mit einem anderen Agenten geteilt wird.
Du bekommst 20 Punkte, wenn du der einzige bist, der ein Wort für {category} generiert.
In allen übrigen Fällen bekommst du 10 Punkte.

{instructions}

Regeln:
1. Spiele so, dass du die meisten Punkte erzielst.
2. Generiere nur ein Wort für die Kategorie {category}.
3. Das Wort muss mit dem Buchstaben "{letter}" beginnen.
4. Das Wort muss ein reales Beispiel für die Kategorie {category} sein.
5. Falls kein gültiges Wort existiert, lass es leer.
6. Füge keine Erklärungen hinzu.
7. Verwende kein Markdown.
8. Verwende kein JSON.
9. Mach keine Fehler

Beispiele für den Buchstaben B:
Säugetier: Bär
Verschluss: Bolzen
Wohndekoration: Bettdecke
Pflanze: Bambus

Die Kategorie ist {category}
Wort beginnend mit {letter} für {category}:""",

    "Spanish": """Eres un agente que juega a Scattergories.
Dada la categoría {category}, debes generar una palabra que encaje con {category} y que empiece por la letra {letter}.
Otros agentes están jugando contigo.
Obtienes 0 puntos si no conoces ninguna palabra.
Obtienes 5 puntos si tu palabra generada es compartida con otro agente.
Obtienes 20 puntos si eres el único que genera una palabra para {category}.
En todos los demás casos, obtienes 10 puntos.

{instructions}

Reglas:
1. Juega para conseguir la mayor cantidad de puntos.
2. Genera solo una palabra para la categoría {category}.
3. La palabra debe empezar por la letra "{letter}".
4. La palabra debe ser un ejemplo real de la categoría {category}.
5. Si no existe ninguna palabra válida, déjalo en blanco.
6. No añadas explicaciones.
7. No uses markdown.
8. No uses JSON.
9. No cometas errores

Ejemplos para la letra B:
mamífero: Búfalo
cierre: Broche
decoración del hogar: Butaca
planta: Bambú

La categoría es {category}
Palabra que empieza por {letter} para {category}:"""
}


# The strategy of an agent in a game with strategies. Prototypical and
# peripheral members differ in how representative they are of the category,
# that is, how many attributes they share with its other members. Frequency of
# occurrence correlates with typicality but is a distinct property. The neutral
# strategy adds a strategy without a direction, as a control for the others.
STRATEGY_HEADER = {
    "English": "Your strategy is:",
    "German": "Deine Strategie ist:",
    "Spanish": "Tu estrategia es:",
}

STRATEGIES = {
    "prototypical": {
        "English": "Generate a word that is a highly representative member of the category, one that shares many attributes with the other members of the category.",
        "German": "Generiere ein Wort, das ein sehr repräsentatives Exemplar der Kategorie ist, also eines, das viele Merkmale mit den anderen Exemplaren der Kategorie teilt.",
        "Spanish": "Genera una palabra que sea un miembro muy representativo de la categoría, es decir, uno que comparta muchos atributos con los demás miembros de la categoría.",
    },
    "peripheral": {
        "English": "Generate a word that is a member of the category but a poorly representative one, one that shares few attributes with the other members of the category.",
        "German": "Generiere ein Wort, das zwar ein Exemplar der Kategorie ist, aber ein kaum repräsentatives, also eines, das nur wenige Merkmale mit den anderen Exemplaren der Kategorie teilt.",
        "Spanish": "Genera una palabra que sea un miembro de la categoría, pero poco representativo, es decir, uno que comparta pocos atributos con los demás miembros de la categoría.",
    },
    "frequency_based": {
        "English": "Generate a word that occurs frequently in everyday language, regardless of how representative it is of the category.",
        "German": "Generiere ein Wort, das in der Alltagssprache häufig vorkommt, unabhängig davon, wie repräsentativ es für die Kategorie ist.",
        "Spanish": "Genera una palabra que aparezca con frecuencia en el lenguaje cotidiano, independientemente de lo representativa que sea de la categoría.",
    },
    "neutral": {
        "English": "Generate a word that is a member of the category, without preferring more or less representative members.",
        "German": "Generiere ein Wort, das ein Exemplar der Kategorie ist, ohne mehr oder weniger repräsentative Exemplare zu bevorzugen.",
        "Spanish": "Genera una palabra que sea un miembro de la categoría, sin preferir miembros más o menos representativos.",
    },
}

# The name of each category in the prompt. The dataset names every category in
# English, which stays the key of the gold members and of the result files.
CATEGORY_NAMES = {
    "German": {
        "animal": "Tier",
        "bird": "Vogel",
        "clothing": "Kleidung",
        "fruit": "Obst",
        "furniture": "Möbel",
        "musical instrument": "Musikinstrument",
        "profession": "Beruf",
        "sport": "Sportart",
        "tool": "Werkzeug",
        "vegetable": "Gemüse",
        "vehicle": "Fahrzeug",
    },
    "Spanish": {
        "animal": "animal",
        "body part": "parte del cuerpo",
        "clothing": "ropa",
        "furniture": "mueble",
        "intelligence": "inteligencia",
    },
}

# Shown from the second round of a game with history.
SCORES = {
    "English": "Your score after the previous rounds: {own} points.\nScores of the other agents: {others} points.",
    "German": "Deine Punktzahl nach den vorherigen Runden: {own} Punkte.\nPunktzahlen der anderen Agenten: {others} Punkte.",
    "Spanish": "Tu puntuación tras las rondas anteriores: {own} puntos.\nPuntuaciones de los otros agentes: {others} puntos.",
}

# Shown when the letter of the round appeared in a previous round.
USED_WORDS = {
    "English": 'Words already used for {category} with the letter "{letter}" in previous rounds: {words}.\nYou get 0 points if you use one of these words again.',
    "German": 'Bereits in vorherigen Runden für {category} mit dem Buchstaben "{letter}" verwendete Wörter: {words}.\nDu bekommst 0 Punkte, wenn du eines dieser Wörter erneut verwendest.',
    "Spanish": 'Palabras ya usadas para {category} con la letra "{letter}" en rondas anteriores: {words}.\nObtienes 0 puntos si vuelves a usar una de estas palabras.',
}


def build_prompt(category: str, letter: str, instructions: str, language: str) -> str:
    return PROMPTS[language.capitalize()].format(
        category=category, letter=letter, instructions=instructions
    )


