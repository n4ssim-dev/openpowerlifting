"""
Simple Groq chat UI with MLflow tracing.

1) Terminal 1:  mlflow ui --port 5000
2) Terminal 2:  python groq_chat.py
3) Chat at http://localhost:7860, view traces at http://localhost:5000 (Experiments -> "Groq Chat" -> Traces)
"""
import json
import os
import sqlite3
import time

import gradio as gr
import mlflow
from dotenv import load_dotenv
from groq import Groq

from common import DB_PATH
from ml import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI

load_dotenv()

# MLflow setup
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Groq Chat")
mlflow.groq.autolog()  # traces every client.chat.completions.create(...) call

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"
MAX_ROWS = 50
MAX_TOOL_ROUNDS = 5
SQL_TIMEOUT_S = 10

SYSTEM_PROMPT = """Tu es l'assistant du projet OpenPowerlifting : une base SQLite de performances
de powerlifting et trois modèles ML (régression linéaire, Random Forest, KNN) qui prédisent
le total (kg) à partir de l'âge, du poids de corps, du sexe et de l'équipement.

Règles :
- Réponds uniquement à partir des résultats de tes outils, jamais de tes connaissances générales.
  Si les données ne permettent pas de répondre, dis-le en une phrase, sans rien inventer.
- Question hors du projet (culture générale, code, actualité…) : refuse en une phrase,
  par exemple « Je ne réponds qu'aux questions sur les données et les modèles du projet. »
- Sois concis : 3 phrases maximum, ou un petit tableau Markdown pour comparer des chiffres.
  Pas d'introduction, pas de récapitulatif, pas de conseils non demandés.
- Réponds dans la langue du dernier message de l'utilisateur : en anglais s'il écrit en
  anglais, même si ces consignes sont en français.
- Utilise `resultats_ml` pour les scores des modèles (R², MAE, RMSE : plus bas = mieux pour
  MAE et RMSE), et `requete_sql` pour les questions sur les données.

Schéma de la base (SQLite) :
- fait_performance(id_performance, age, poids_kg, categorie_poids_kg, coeff_wilks, total_kg,
  place, best_squat_kg, best_bench_kg, best_deadlift_kg, id_rencontre, id_division,
  id_powerlifter, id_equipement)
- dim_powerlifter(id_powerlifter, sexe)  -- sexe : 'M' ou 'F'
- dim_rencontre(id_rencontre, federation, date_rencontre, pays, etat, ville, nom_rencontre)
- dim_equipement(id_equipement)  -- Raw, Wraps, Single-ply, Multi-ply
- dim_division(id_division)
- participation_rencontre(id_rencontre, id_powerlifter)
`place` peut valoir un rang ou 'DQ', 'DD', 'NS', 'G'."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "resultats_ml",
            "description": "Scores (R², MAE, RMSE) et hyperparamètres des modèles pour les derniers entraînements enregistrés dans MLflow, du plus récent au plus ancien.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nb_entrainements": {
                        "type": "integer",
                        "description": "Nombre d'entraînements à remonter (1 = le dernier).",
                        "default": 1,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "requete_sql",
            "description": f"Exécute une requête SELECT en lecture seule sur la base SQLite du projet. Renvoie au plus {MAX_ROWS} lignes : agrège côté SQL.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "Une seule requête SELECT ou WITH."}},
                "required": ["sql"],
            },
        },
    },
]


@mlflow.trace(span_type="TOOL")
def resultats_ml(nb_entrainements: int = 1) -> dict:
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT],
        filter_string="attributes.run_name = 'comparaison'",
        order_by=["attributes.start_time DESC"],
        max_results=max(1, min(nb_entrainements, 10)),
        output_format="list",
    )
    if not runs:
        return {"erreur": "Aucun entraînement enregistré dans MLflow."}

    entrainements = []
    for parent in runs:
        enfants = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT],
            filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
            output_format="list",
        )
        entrainements.append({
            "date": time.strftime("%Y-%m-%d %H:%M", time.localtime(parent.info.start_time / 1000)),
            "parametres": parent.data.params,
            "modeles": {
                e.info.run_name: {
                    "metriques": {k: round(v, 3) for k, v in e.data.metrics.items()},
                    "hyperparametres": e.data.params,
                }
                for e in enfants
            },
        })
    return {"entrainements": entrainements}


@mlflow.trace(span_type="TOOL")
def requete_sql(sql: str) -> dict:
    if not sql.lstrip().lower().startswith(("select", "with")):
        return {"erreur": "Seules les requêtes SELECT sont autorisées."}
    # Ouverture en lecture seule : aucune écriture possible, même via le SQL généré.
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    debut = time.monotonic()
    conn.set_progress_handler(lambda: time.monotonic() - debut > SQL_TIMEOUT_S, 10_000)
    try:
        cur = conn.execute(sql)
        colonnes = [d[0] for d in cur.description]
        lignes = cur.fetchmany(MAX_ROWS + 1)
    except sqlite3.Error as e:
        return {"erreur": str(e)}
    finally:
        conn.close()
    return {"colonnes": colonnes, "lignes": lignes[:MAX_ROWS], "tronque": len(lignes) > MAX_ROWS}


OUTILS = {"resultats_ml": resultats_ml, "requete_sql": requete_sql}


def _texte(content) -> str:
    # Gradio 6 renvoie l'historique sous forme de blocs [{"type": "text", "text": ...}].
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


@mlflow.trace(name="chat_turn")
def ask_groq(messages: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    for _ in range(MAX_TOOL_ROUNDS):
        # Note: MLflow's Groq autolog does not trace streaming calls,
        # so we use a normal (non-streaming) request here.
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.3,
            max_completion_tokens=1500,
            reasoning_effort="medium",
            stream=False,
        )
        message = completion.choices[0].message
        if not message.tool_calls:
            return message.content or ""

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })
        for tc in message.tool_calls:
            outil = OUTILS.get(tc.function.name)
            try:
                resultat = outil(**json.loads(tc.function.arguments or "{}")) if outil else {"erreur": "Outil inconnu."}
            except (TypeError, json.JSONDecodeError) as e:
                resultat = {"erreur": str(e)}
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(resultat, ensure_ascii=False, default=str),
            })
    return "Je n'ai pas réussi à obtenir une réponse à partir des données du projet."


def respond(message: str, history: list[dict]) -> str:
    messages = [
        {"role": m["role"], "content": _texte(m["content"])}
        for m in history
        if m["role"] in ("user", "assistant") and _texte(m["content"])
    ]
    messages.append({"role": "user", "content": message})
    return ask_groq(messages)


demo = gr.ChatInterface(
    fn=respond,
    title="Interface chat (Groq historisé avec MLflow)",
    description=f"Model: {MODEL} · Traces: http://localhost:5000",
)

if __name__ == "__main__":
    demo.launch()
