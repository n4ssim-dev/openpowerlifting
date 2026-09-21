import pandas as pd
import plotly.express as px
import streamlit as st

from common import AGE_BINS, AGE_LABELS, BLUE, load_data

st.set_page_config(page_title="Âge — OpenPowerlifting", layout="wide")

df = load_data()

st.title("Performances par tranche d'âge")

age_df = df.dropna(subset=["age"]).copy()
age_df["tranche_age"] = pd.cut(age_df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)

fig = px.box(
    age_df, x="tranche_age", y="total_kg",
    category_orders={"tranche_age": AGE_LABELS},
    color_discrete_sequence=[BLUE], template="plotly_white",
)
fig.update_layout(xaxis_title="Tranche d'âge", yaxis_title="Total (kg)")
st.plotly_chart(fig)
