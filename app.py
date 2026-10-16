import os
import socket
from urllib.parse import urlparse

import plotly.express as px
import streamlit as st

import ml
from common import BLUE, get_trained_models, load_data

CHAT_URL = f"http://localhost:{os.getenv('GRADIO_PORT', '7860')}"


def service_up(url: str) -> bool:
    u = urlparse(url)
    try:
        socket.create_connection((u.hostname, u.port), timeout=0.3).close()
        return True
    except OSError:
        return False


st.set_page_config(page_title="OpenPowerlifting Analytics", layout="wide")

df = load_data()

st.title("OpenPowerlifting Analytics")
st.caption("Utilisez le menu à gauche pour explorer chaque catégorie (âge, équipement, sexe, prédiction ML).")

a1, a2, a3 = st.columns(3)
# La prédiction passe en premier : elle entraîne et enregistre les modèles dans MLflow,
# le lien de comparaison calculé ensuite pointe donc vers ce dernier entraînement.
if a1.button("Lancer une prédiction aléatoire", icon=":material/casino:", width="stretch"):
    models = get_trained_models(df)
    athlete = df.dropna(subset=ml.FEATURES + [ml.TARGET]).sample(1).iloc[0]
    predictions = {
        res["label"]: ml.predict_single(
            res["pipeline"], athlete["age"], athlete["poids_kg"], athlete["sexe"], athlete["id_equipement"]
        )
        for res in models.values()
    }
else:
    athlete = None

if service_up(CHAT_URL):
    a2.link_button("Ouvrir le chat", CHAT_URL, icon=":material/chat:", width="stretch")
else:
    a2.button("Ouvrir le chat", icon=":material/chat:", width="stretch", disabled=True,
              help=f"Le chat ne répond pas sur {CHAT_URL}.")

comparison_url = ml.latest_comparison_url()
if comparison_url:
    a3.link_button("Comparaison des modèles (MLflow)", comparison_url, icon=":material/compare_arrows:", width="stretch")
else:
    a3.button("Comparaison des modèles (MLflow)", icon=":material/compare_arrows:", width="stretch", disabled=True,
              help="Aucun entraînement enregistré : lance d'abord une prédiction (MLflow doit tourner).")

if athlete is not None:
    st.markdown(
        f"**{athlete['id_powerlifter']}** — {athlete['sexe']}, {athlete['age']:g} ans, "
        f"{athlete['poids_kg']:g} kg, {athlete['id_equipement']}, "
        f"{athlete['federation']} ({athlete['date_rencontre']:%Y-%m-%d})"
    )
    p_cols = st.columns(len(predictions) + 1)
    p_cols[0].metric("Total réel", f"{athlete['total_kg']:.1f} kg")
    for col, (label, pred) in zip(p_cols[1:], predictions.items()):
        col.metric(label, f"{pred:.1f} kg", f"{pred - athlete['total_kg']:+.1f} kg", delta_color="off")

st.divider()
st.subheader("Statistiques générales")

with st.expander("Chiffres clés", icon=":material/insights:"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Performances", f"{len(df):,}".replace(",", " "))
    c2.metric("Athlètes uniques", f"{df['id_powerlifter'].nunique():,}".replace(",", " "))
    c3.metric("Fédérations", df["federation"].nunique())
    c4.metric("Total moyen (kg)", f"{df['total_kg'].mean():.1f}")

with st.expander("Nombre de performances par année", icon=":material/show_chart:"):
    per_year = df.groupby("annee").size().reset_index(name="performances")
    fig = px.line(
        per_year, x="annee", y="performances", markers=True,
        color_discrete_sequence=[BLUE], template="plotly_white",
    )
    fig.update_layout(xaxis_title="Année", yaxis_title="Performances")
    st.plotly_chart(fig)

with st.expander("Top 10 fédérations", icon=":material/groups:"):
    top10 = df["federation"].value_counts().head(10).reset_index()
    top10.columns = ["federation", "performances"]
    fig = px.bar(
        top10.sort_values("performances"), x="performances", y="federation", orientation="h",
        color_discrete_sequence=[BLUE], template="plotly_white",
    )
    fig.update_layout(xaxis_title="Performances", yaxis_title="")
    st.plotly_chart(fig)

with st.expander("Top 10 pays organisateurs", icon=":material/public:"):
    top_pays = df["pays"].value_counts().head(10).reset_index()
    top_pays.columns = ["pays", "performances"]
    fig = px.bar(
        top_pays.sort_values("performances"), x="performances", y="pays", orientation="h",
        color_discrete_sequence=[BLUE], template="plotly_white",
    )
    fig.update_layout(xaxis_title="Performances", yaxis_title="")
    st.plotly_chart(fig)

with st.expander("Voir les données", icon=":material/table:"):
    st.dataframe(df)
