
PROMPTS = {
    "English": f"""
You are an agent playing Scattergories.
Given category {0}, you have to generate one word that fits {0} and starts with letter {1}.
Other agents are playing with you.
You get 0 points if you do not know a word.
You get 5 points if your generated word is shared with another agent.
You get 20 points if you are the only one generating a word for {0}.
In all the remaining cases, you get 10 points.

{2}

Rules:
1. Play to achieve the most points.
2. Generate only one word for the category {0}.
3. The word must start with the letter "{1}".
4. The word must be a real example of the category {0}.
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

The category is {0}
Word starting with {1} for {0}: 
""".strip(),

    "German": f"""Du bist ein Agent, der Scattergories spielt.
Gegeben die Kategorie {0}, musst du ein Wort generieren, das zu {0} passt und mit dem Buchstaben {1} beginnt.
Andere Agenten spielen mit dir.
Du bekommst 0 Punkte, wenn du kein Wort kennst.
Du bekommst 5 Punkte, wenn dein generiertes Wort mit einem anderen Agenten geteilt wird.
Du bekommst 20 Punkte, wenn du der einzige bist, der ein Wort für {0} generiert.
In allen übrigen Fällen bekommst du 10 Punkte.

{2}

Regeln:
1. Spiele so, dass du die meisten Punkte erzielst.
2. Generiere nur ein Wort für die Kategorie {0}.
3. Das Wort muss mit dem Buchstaben "{1}" beginnen.
4. Das Wort muss ein reales Beispiel für die Kategorie {0} sein.
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

Die Kategorie ist {0}
Wort beginnend mit {1} für {0}:""",

    "Spanish": f"""Eres un agente que juega a Scattergories.
Dada la categoría {0}, debes generar una palabra que encaje con {0} y que empiece por la letra {1}.
Otros agentes están jugando contigo.
Obtienes 0 puntos si no conoces ninguna palabra.
Obtienes 5 puntos si tu palabra generada es compartida con otro agente.
Obtienes 20 puntos si eres el único que genera una palabra para {0}.
En todos los demás casos, obtienes 10 puntos.

{2}

Reglas:
1. Juega para conseguir la mayor cantidad de puntos.
2. Genera solo una palabra para la categoría {0}.
3. La palabra debe empezar por la letra "{1}".
4. La palabra debe ser un ejemplo real de la categoría {0}.
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

La categoría es {0}
Palabra que empieza por {1} para {0}:"""
}


def build_prompt(slot: str, letter: str, strategy_instructions: str, language: str) -> str:
    return PROMPTS[language.capitalize()].format(slot, letter, strategy_instructions)


