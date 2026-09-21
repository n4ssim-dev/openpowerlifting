import plotly.express as px
import streamlit as st

from common import SEX_COLORS, load_data

st.set_page_config(page_title="Sexe — OpenPowerlifting", layout="wide")

df = load_data()

st.title("Différenciation par sexe")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution du total (kg) par sexe")
    fig = px.histogram(
        df, x="total_kg", color="sexe", barmode="overlay", opacity=0.7, nbins=60,
        color_discrete_map=SEX_COLORS, template="plotly_white",
    )
    fig.update_layout(xaxis_title="Total (kg)", yaxis_title="Performances", legend_title="Sexe")
    st.plotly_chart(fig)

with col2:
    st.subheader("Poids de corps vs. total (kg)")
    sample = df.dropna(subset=["poids_kg", "total_kg", "sexe"])
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)
    fig = px.scatter(
        sample, x="poids_kg", y="total_kg", color="sexe", opacity=0.5,
        color_discrete_map=SEX_COLORS, template="plotly_white",
    )
    fig.update_layout(xaxis_title="Poids de corps (kg)", yaxis_title="Total (kg)", legend_title="Sexe")
    st.plotly_chart(fig)
