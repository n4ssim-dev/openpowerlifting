import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import ml

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "database.db"

# Reference categorical palette (dataviz skill, light mode) — fixed slot order,
# never cycled or reassigned per filter.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
SEX_COLORS = {"M": BLUE, "F": ORANGE}

st.set_page_config(page_title="OpenPowerlifting Analytics", layout="wide")


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data
def load_data() -> pd.DataFrame:
    query = """
        SELECT
            fp.id_performance,
            fp.age,
            fp.poids_kg,
            fp.coeff_wilks,
            fp.total_kg,
            fp.place,
            fp.best_squat_kg,
            fp.best_bench_kg,
            fp.best_deadlift_kg,
            fp.id_division,
            fp.id_equipement,
            fp.id_powerlifter,
            dp.sexe,
            dr.federation,
            dr.pays,
            dr.date_rencontre
        FROM fait_performance fp
        JOIN dim_rencontre dr ON fp.id_rencontre = dr.id_rencontre
        LEFT JOIN dim_powerlifter dp ON fp.id_powerlifter = dp.id_powerlifter
        WHERE fp.total_kg IS NOT NULL
    """
    df = pd.read_sql_query(query, get_connection())
    df["date_rencontre"] = pd.to_datetime(df["date_rencontre"], errors="coerce")
    df["annee"] = df["date_rencontre"].dt.year
    return df


df = load_data()

st.title("OpenPowerlifting Analytics")
st.caption(
    f"{len(df):,} performances, {df['id_powerlifter'].nunique():,} athlètes, "
    f"{df['federation'].nunique():,} fédérations.".replace(",", " ")
)

# --- Filters ---------------------------------------------------------------
st.sidebar.header("Filtres")

age_min, age_max = int(df["age"].min(skipna=True)), int(df["age"].max(skipna=True))
age_range = st.sidebar.slider("Âge", age_min, age_max, (age_min, age_max))

equip_options = sorted(df["id_equipement"].dropna().unique().tolist())
equip_choice = st.sidebar.multiselect("Équipement", equip_options, default=equip_options)

top_federations = df["federation"].value_counts().head(20).index.tolist()
federation_choice = st.sidebar.multiselect(
    "Fédération (top 20 par défaut = toutes)", top_federations, default=top_federations
)

year_min, year_max = int(df["annee"].min()), int(df["annee"].max())
year_range = st.sidebar.slider("Années", year_min, year_max, (year_min, year_max))

sexe_options = sorted(df["sexe"].dropna().unique().tolist())
sexe_choice = st.sidebar.multiselect("Sexe", sexe_options, default=sexe_options)

mask = (
    (df["age"].isna() | df["age"].between(age_range[0], age_range[1]))
    & df["id_equipement"].isin(equip_choice)
    & df["federation"].isin(federation_choice)
    & df["annee"].between(year_range[0], year_range[1])
    & df["sexe"].isin(sexe_choice)
)
fdf = df[mask]

if fdf.empty:
    st.warning("Aucune donnée pour ces filtres.")
    st.stop()

# --- KPIs --------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Performances", f"{len(fdf):,}".replace(",", " "))
c2.metric("Athlètes uniques", f"{fdf['id_powerlifter'].nunique():,}".replace(",", " "))
c3.metric("Fédérations", fdf["federation"].nunique())
c4.metric("Total moyen (kg)", f"{fdf['total_kg'].mean():.1f}")

st.divider()

# --- Charts --------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Nombre de performances par année")
    per_year = fdf.groupby("annee").size().reset_index(name="performances")
    fig = px.line(
        per_year, x="annee", y="performances", markers=True,
        color_discrete_sequence=[BLUE], template="plotly_white",
    )
    fig.update_layout(xaxis_title="Année", yaxis_title="Performances")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Top 10 fédérations")
    top10 = fdf["federation"].value_counts().head(10).reset_index()
    top10.columns = ["federation", "performances"]
    fig = px.bar(
        top10.sort_values("performances"), x="performances", y="federation", orientation="h",
        color_discrete_sequence=[BLUE], template="plotly_white",
    )
    fig.update_layout(xaxis_title="Performances", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Total (kg) par tranche d'âge")
age_bins = [0, 18, 23, 34, 44, 54, 64, 150]
age_labels = ["-18", "18-23", "24-34", "35-44", "45-54", "55-64", "65+"]
age_df = fdf.dropna(subset=["age"]).copy()
age_df["tranche_age"] = pd.cut(age_df["age"], bins=age_bins, labels=age_labels, right=True)
fig = px.box(
    age_df, x="tranche_age", y="total_kg",
    category_orders={"tranche_age": age_labels},
    color_discrete_sequence=[BLUE], template="plotly_white",
)
fig.update_layout(xaxis_title="Tranche d'âge", yaxis_title="Total (kg)")
st.plotly_chart(fig, use_container_width=True)

# --- ML: prédiction du total (kg) selon l'âge -----------------------------
st.subheader("Prédiction ML : total (kg) selon l'âge")
st.caption(
    "Régression linéaire vs. Random Forest, entraînées sur l'ensemble des données "
    "(âge, poids de corps, sexe, équipement) pour prédire le total (kg)."
)


@st.cache_resource(show_spinner="Entraînement des modèles...")
def get_trained_models(training_df: pd.DataFrame) -> dict:
    return ml.train_models(training_df)


models = get_trained_models(df)

metric_cols = st.columns(len(models))
for col, res in zip(metric_cols, models.values()):
    col.metric(res["label"], f"R² {res['r2']:.2f}", f"MAE {res['mae']:.1f} kg")

model_key = st.selectbox(
    "Modèle à comparer sur les données filtrées",
    list(models.keys()),
    format_func=lambda k: models[k]["label"],
)
comparison = ml.age_bracket_comparison(fdf, models[model_key]["pipeline"], age_bins, age_labels)
comparison_long = comparison.melt(
    id_vars="tranche_age",
    value_vars=["total_kg_reel", "total_kg_predit"],
    var_name="type",
    value_name="total_kg",
)
comparison_long["type"] = comparison_long["type"].map(
    {"total_kg_reel": "Réel", "total_kg_predit": "Prédit"}
)
fig = px.line(
    comparison_long, x="tranche_age", y="total_kg", color="type", markers=True,
    category_orders={"tranche_age": age_labels},
    color_discrete_map={"Réel": BLUE, "Prédit": ORANGE}, template="plotly_white",
)
fig.update_layout(xaxis_title="Tranche d'âge", yaxis_title="Total (kg)", legend_title="")
st.plotly_chart(fig, use_container_width=True)

st.markdown("**Simuler une performance**")
p1, p2, p3, p4 = st.columns(4)
sim_age = p1.number_input("Âge", min_value=10, max_value=100, value=25)
sim_poids = p2.number_input("Poids de corps (kg)", min_value=30.0, max_value=250.0, value=80.0, step=0.5)
sim_sexe = p3.selectbox("Sexe", sexe_options, key="sim_sexe")
sim_equip = p4.selectbox("Équipement", equip_options, key="sim_equip")

if st.button("Prédire le total (kg)"):
    pred_cols = st.columns(len(models))
    for col, res in zip(pred_cols, models.values()):
        prediction = ml.predict_single(res["pipeline"], sim_age, sim_poids, sim_sexe, sim_equip)
        col.metric(res["label"], f"{prediction:.1f} kg")

st.subheader("Total (kg) par équipement")
fig = px.box(
    fdf, x="id_equipement", y="total_kg", color="id_equipement",
    color_discrete_sequence=CATEGORICAL, template="plotly_white",
)
fig.update_layout(xaxis_title="Équipement", yaxis_title="Total (kg)", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top 10 pays organisateurs")
top_pays = fdf["pays"].value_counts().head(10).reset_index()
top_pays.columns = ["pays", "performances"]
fig = px.bar(
    top_pays.sort_values("performances"), x="performances", y="pays", orientation="h",
    color_discrete_sequence=[BLUE], template="plotly_white",
)
fig.update_layout(xaxis_title="Performances", yaxis_title="")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Différenciation par sexe")

col3, col4 = st.columns(2)

with col3:
    st.caption("Distribution du total (kg) par sexe")
    fig = px.histogram(
        fdf, x="total_kg", color="sexe", barmode="overlay", opacity=0.7, nbins=60,
        color_discrete_map=SEX_COLORS, template="plotly_white",
    )
    fig.update_layout(xaxis_title="Total (kg)", yaxis_title="Performances", legend_title="Sexe")
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.caption("Poids de corps vs. total (kg) par sexe")
    sample = fdf.dropna(subset=["poids_kg", "total_kg", "sexe"])
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)
    fig = px.scatter(
        sample, x="poids_kg", y="total_kg", color="sexe", opacity=0.5,
        color_discrete_map=SEX_COLORS, template="plotly_white",
    )
    fig.update_layout(xaxis_title="Poids de corps (kg)", yaxis_title="Total (kg)", legend_title="Sexe")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Voir les données filtrées"):
    st.dataframe(fdf)
