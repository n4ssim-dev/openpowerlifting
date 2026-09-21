import streamlit as st
import plotly.express as px

from common import AGE_BINS, AGE_LABELS, BLUE, ORANGE, load_data
import ml

st.set_page_config(page_title="Prédiction ML — OpenPowerlifting", layout="wide")

df = load_data()

st.title("Prédiction ML : total (kg) selon l'âge")
st.caption(
    "Régression linéaire vs. Random Forest, entraînées sur l'ensemble des données "
    "(âge, poids de corps, sexe, équipement) pour prédire le total (kg)."
)


@st.cache_resource(show_spinner="Entraînement des modèles...")
def get_trained_models(training_df):
    return ml.train_models(training_df)


models = get_trained_models(df)

metric_cols = st.columns(len(models))
for col, res in zip(metric_cols, models.values()):
    col.metric(res["label"], f"R² {res['r2']:.2f}", f"MAE {res['mae']:.1f} kg")

model_key = st.selectbox(
    "Modèle à comparer",
    list(models.keys()),
    format_func=lambda k: models[k]["label"],
)
comparison = ml.age_bracket_comparison(df, models[model_key]["pipeline"], AGE_BINS, AGE_LABELS)
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
    category_orders={"tranche_age": AGE_LABELS},
    color_discrete_map={"Réel": BLUE, "Prédit": ORANGE}, template="plotly_white",
)
fig.update_layout(xaxis_title="Tranche d'âge", yaxis_title="Total (kg)", legend_title="")
st.plotly_chart(fig)

st.markdown("**Simuler une performance**")
sexe_options = sorted(df["sexe"].dropna().unique().tolist())
equip_options = sorted(df["id_equipement"].dropna().unique().tolist())

p1, p2, p3, p4 = st.columns(4)
sim_age = p1.number_input("Âge", min_value=10, max_value=100, value=25)
sim_poids = p2.number_input("Poids de corps (kg)", min_value=30.0, max_value=250.0, value=80.0, step=0.5)
sim_sexe = p3.selectbox("Sexe", sexe_options)
sim_equip = p4.selectbox("Équipement", equip_options)

if st.button("Prédire le total (kg)"):
    pred_cols = st.columns(len(models))
    for col, res in zip(pred_cols, models.values()):
        prediction = ml.predict_single(res["pipeline"], sim_age, sim_poids, sim_sexe, sim_equip)
        col.metric(res["label"], f"{prediction:.1f} kg")
