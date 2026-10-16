# OpenPowerLifting analytics

Ce projet à produire vise une application streamlit permettant de visualiser les données issue du dataset [powerlifting-database](https://www.kaggle.com/datasets/dansbecker/powerlifting-database) et d'effectuer des prédictions à l'aide de modèles ML prédictifs (régression linéaire, random forest, KNN).

## Lancer le projet

Prérequis : [uv](https://docs.astral.sh/uv/) et `curl`. Pour le chat, ajouter `GROQ_API_KEY=...` dans un fichier `.env`.

```bash
./run.sh               # installe les dépendances, crée la base depuis Kaggle, lance tout
./run.sh --check       # vérifications seulement
./run.sh --rebuild-db  # recrée data/database.db
./run.sh --no-chat     # sans le chat Gradio
```

| Service | URL |
|---|---|
| Streamlit | http://localhost:8501 |
| MLflow | http://127.0.0.1:5000 |
| Chat Gradio | http://localhost:7860 |

Les journaux sont dans `logs/`. `Ctrl+C` arrête tous les services.
