import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["age", "poids_kg", "sexe", "id_equipement"]
TARGET = "total_kg"
NUMERIC_FEATURES = ["age", "poids_kg"]
CATEGORICAL_FEATURES = ["sexe", "id_equipement"]

MODEL_LABELS = {
    "linear": "Régression linéaire",
    "random_forest": "Random Forest",
}


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=FEATURES + [TARGET])[FEATURES + [TARGET]]


def _build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Fit a linear regression and a random forest predicting total_kg
    from age, bodyweight, sex and equipment. Returns, per model key,
    the fitted pipeline plus R2/MAE measured on a held-out test split."""
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
    }

    results = {}
    for key, model in candidates.items():
        pipeline = _build_pipeline(model)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        results[key] = {
            "label": MODEL_LABELS[key],
            "pipeline": pipeline,
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
        }
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
    """Mean actual vs. predicted total_kg per age bracket, for a chart
    comparing the model's trend against the observed data."""
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
