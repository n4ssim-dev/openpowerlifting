import json
import logging
import os
from urllib.parse import quote

import mlflow
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["age", "poids_kg", "sexe", "id_equipement"]
TARGET = "total_kg"
NUMERIC_FEATURES = ["age", "poids_kg"]
CATEGORICAL_FEATURES = ["sexe", "id_equipement"]

MODEL_LABELS = {
    "linear": "Régression linéaire",
    "random_forest": "Random Forest",
    "knn": "KNN",
}

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = "Prédiction total_kg"
# Types internes que skops refuse par défaut ; sûrs ici puisqu'on entraîne nous-mêmes les modèles.
SKOPS_TRUSTED_TYPES = {
    "random_forest": ["sklearn.tree._tree.Tree"],
    "knn": ["sklearn.metrics._dist_metrics.EuclideanDistance64", "sklearn.neighbors._kd_tree.KDTree"],
}

logger = logging.getLogger(__name__)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=FEATURES + [TARGET])[FEATURES + [TARGET]]


def _build_pipeline(model, scale_numeric: bool = False) -> Pipeline:
    # Le KNN raisonne en distances : sans mise à l'échelle, le poids écrase l'âge.
    numeric = StandardScaler() if scale_numeric else "passthrough"
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", numeric, NUMERIC_FEATURES),
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _mlflow_available() -> bool:
    # Vérification rapide : sans serveur, MLflow réessaie pendant plusieurs minutes.
    try:
        return requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=2).ok
    except requests.RequestException:
        return False


def _log_to_mlflow(results: dict, X_test: pd.DataFrame, test_size: float, random_state: int) -> None:
    """Un run parent par entraînement, un run enfant par modèle."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="comparaison"):
        mlflow.log_params({
            "test_size": test_size,
            "random_state": random_state,
            "features": ", ".join(FEATURES),
            "n_test": len(X_test),
        })
        for key, res in results.items():
            with mlflow.start_run(run_name=key, nested=True):
                mlflow.log_params(res["pipeline"].named_steps["model"].get_params())
                mlflow.log_metrics({"r2": res["r2"], "mae": res["mae"], "rmse": res["rmse"]})
                mlflow.sklearn.log_model(
                    res["pipeline"],
                    name=key,
                    input_example=X_test.head(),
                    skops_trusted_types=SKOPS_TRUSTED_TYPES.get(key),
                )


def latest_comparison_url() -> str | None:
    """Lien MLflow comparant les modèles du dernier entraînement, ou None s'il n'y en a pas encore."""
    if not _mlflow_available():
        return None
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    parents = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT],
        filter_string="attributes.run_name = 'comparaison'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not parents:
        return None
    parent = parents[0]
    enfants = mlflow.search_runs(
        experiment_ids=[parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
        output_format="list",
    )
    if not enfants:
        return None
    runs = quote(json.dumps([r.info.run_id for r in enfants], separators=(",", ":")))
    experiments = quote(json.dumps([parent.info.experiment_id]))
    return f"{MLFLOW_TRACKING_URI}/#/compare-runs?runs={runs}&experiments={experiments}"


def train_models(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, track: bool = True
) -> dict:
    """Entraîne les modèles à prédire le total à partir de l'âge, du poids,
    du sexe et de l'équipement. Renvoie pour chacun le pipeline entraîné
    et son R²/MAE/RMSE sur le jeu de test, et trace le tout dans MLflow
    si le serveur répond."""
    data = prepare_dataset(df)
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    candidates = {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=random_state, n_jobs=-1
        ),
        "knn": KNeighborsRegressor(n_neighbors=25, n_jobs=-1),
    }
    scaled = {"knn"}

    results = {}
    for key, model in candidates.items():
        pipeline = _build_pipeline(model, scale_numeric=key in scaled)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        results[key] = {
            "label": MODEL_LABELS[key],
            "pipeline": pipeline,
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "rmse": root_mean_squared_error(y_test, preds),
        }

    if track and _mlflow_available():
        # Un souci de suivi ne doit pas empêcher l'app d'afficher les modèles.
        try:
            _log_to_mlflow(results, X_test, test_size, random_state)
        except Exception:
            logger.exception("Échec de l'enregistrement dans MLflow")
    return results


def predict_single(pipeline: Pipeline, age: float, poids_kg: float, sexe: str, id_equipement: str) -> float:
    row = pd.DataFrame([{
        "age": age,
        "poids_kg": poids_kg,
        "sexe": sexe,
        "id_equipement": id_equipement,
    }])
    return float(pipeline.predict(row)[0])


def age_bracket_comparison(
    df: pd.DataFrame, pipeline: Pipeline, age_bins: list, age_labels: list
) -> pd.DataFrame:
    """Total moyen réel et prédit par tranche d'âge."""
    data = prepare_dataset(df).copy()
    if data.empty:
        return pd.DataFrame(columns=["tranche_age", "total_kg_reel", "total_kg_predit"])
    data["tranche_age"] = pd.cut(data["age"], bins=age_bins, labels=age_labels, right=True)
    data["prediction"] = pipeline.predict(data[FEATURES])
    grouped = (
        data.groupby("tranche_age", observed=True)
        .agg(total_kg_reel=("total_kg", "mean"), total_kg_predit=("prediction", "mean"))
        .reset_index()
    )
    return grouped
