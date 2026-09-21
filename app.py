import plotly.express as px
import streamlit as st

from common import BLUE, load_data

st.set_page_config(page_title="OpenPowerlifting Analytics", layout="wide")

df = load_data()

st.title("OpenPowerlifting Analytics")
st.caption(
    f"{len(df):,} performances, {df['id_powerlifter'].nunique():,} athlètes, "
    f"{df['federation'].nunique():,} fédérations.".replace(",", " ")
)
st.caption("Utilisez le menu à gauche pour explorer chaque catégorie (âge, équipement, sexe, prédiction ML).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Performances", f"{len(df):,}".replace(",", " "))
c2.metric("Athlètes uniques", f"{df['id_powerlifter'].nunique():,}".replace(",", " "))
c3.metric("Fédérations", df["federation"].nunique())
c4.metric("Total moyen (kg)", f"{df['total_kg'].mean():.1f}")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Nombre de performances par année")
    per_year = df.groupby("annee").size().reset_index(name="performances")
    fig = px.line(
        per_year, x="annee", y="performances", markers=True,
        color_discrete_sequence=[BLUE], template="plotly_white",
    )
    fig.update_layout(xaxis_title="Année", yaxis_title="Performances")
    st.plotly_chart(fig)

with col2:
    st.subheader("Top 10 fédérations")
    top10 = df["federation"].value_counts().head(10).reset_index()
    top10.columns = ["federation", "performances"]
    fig = px.bar(
        top10.sort_values("performances"), x="performances", y="federation", orientation="h",
        color_discrete_sequence=[BLUE], template="plotly_white",
    )
    fig.update_layout(xaxis_title="Performances", yaxis_title="")
    st.plotly_chart(fig)

st.subheader("Top 10 pays organisateurs")
top_pays = df["pays"].value_counts().head(10).reset_index()
top_pays.columns = ["pays", "performances"]
fig = px.bar(
    top_pays.sort_values("performances"), x="performances", y="pays", orientation="h",
    color_discrete_sequence=[BLUE], template="plotly_white",
)
fig.update_layout(xaxis_title="Performances", yaxis_title="")
st.plotly_chart(fig)

with st.expander("Voir les données"):
    st.dataframe(df)
