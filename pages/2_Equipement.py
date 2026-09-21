import plotly.express as px
import streamlit as st

from common import CATEGORICAL, load_data

st.set_page_config(page_title="Équipement — OpenPowerlifting", layout="wide")

df = load_data()

st.title("Performances par équipement")

fig = px.box(
    df, x="id_equipement", y="total_kg", color="id_equipement",
    color_discrete_sequence=CATEGORICAL, template="plotly_white",
)
fig.update_layout(xaxis_title="Équipement", yaxis_title="Total (kg)", showlegend=False)
st.plotly_chart(fig)
