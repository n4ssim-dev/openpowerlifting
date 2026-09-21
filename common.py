import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

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

AGE_BINS = [0, 18, 23, 34, 44, 54, 64, 150]
AGE_LABELS = ["-18", "18-23", "24-34", "35-44", "45-54", "55-64", "65+"]


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
    # a handful of source rows carry a stray placing code ("DQ") in this column
    df["coeff_wilks"] = pd.to_numeric(df["coeff_wilks"], errors="coerce")
    return df
