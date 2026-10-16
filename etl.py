import sqlite3
from pathlib import Path

import kagglehub
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
KAGGLE_DATASET = "dansbecker/powerlifting-database"
DB_PATH = BASE_DIR / "data" / "database.db"
SQL_PATH = BASE_DIR / "data" / "sql" / "create_tables.sql"


def create_tables(nom_fichier, con):
    with open(nom_fichier, "r") as file:
        sql_query = file.read()
    con.executescript(sql_query)


def download_dataset() -> Path:
    # Téléchargé une fois puis réutilisé depuis le cache de kagglehub (~/.cache/kagglehub).
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def load_meets(csv_dir):
    df = pd.read_csv(csv_dir / "meets.csv")
    return df.rename(columns={
        "MeetID": "id_rencontre",
        "MeetPath": "meet_path",
        "Federation": "federation",
        "Date": "date_rencontre",
        "MeetCountry": "pays",
        "MeetState": "etat",
        "MeetTown": "ville",
        "MeetName": "nom_rencontre",
    })


def load_performances(csv_dir):
    df = pd.read_csv(csv_dir / "openpowerlifting.csv", low_memory=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df.rename(columns={
        "MeetID": "id_rencontre",
        "Name": "id_powerlifter",
        "Sex": "sexe",
        "Equipment": "id_equipement",
        "Division": "id_division",
        "Age": "age",
        "BodyweightKg": "poids_kg",
        "WeightClassKg": "categorie_poids_kg",
        "Squat4Kg": "squat_4kg",
        "BestSquatKg": "best_squat_kg",
        "Bench4Kg": "bench_4kg",
        "BestBenchKg": "best_bench_kg",
        "Deadlift4Kg": "deadlift_4kg",
        "BestDeadliftKg": "best_deadlift_kg",
        "TotalKg": "total_kg",
        "Place": "place",
        "Wilks": "coeff_wilks",
    })


def build_dim_powerlifter(perf):
    return (
        perf[["id_powerlifter", "sexe"]]
        .drop_duplicates(subset="id_powerlifter", keep="first")
        .reset_index(drop=True)
    )


def build_dim_division(perf):
    return perf[["id_division"]].dropna().drop_duplicates().reset_index(drop=True)


def build_dim_equipement(perf):
    return perf[["id_equipement"]].dropna().drop_duplicates().reset_index(drop=True)


def build_participation_rencontre(perf):
    return (
        perf[["id_rencontre", "id_powerlifter"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def build_fait_performance(perf):
    fact = perf.copy()
    fact.insert(0, "id_performance", range(1, len(fact) + 1))
    cols = [
        "id_performance",
        "age",
        "poids_kg",
        "categorie_poids_kg",
        "coeff_wilks",
        "total_kg",
        "place",
        "squat_4kg",
        "bench_4kg",
        "deadlift_4kg",
        "best_squat_kg",
        "best_bench_kg",
        "best_deadlift_kg",
        "id_rencontre",
        "id_division",
        "id_powerlifter",
        "id_equipement",
    ]
    return fact[cols]


def load_data(con, csv_dir):
    dim_rencontre = load_meets(csv_dir)
    dim_rencontre.to_sql("dim_rencontre", con, if_exists="append", index=False)

    perf = load_performances(csv_dir)

    build_dim_powerlifter(perf).to_sql("dim_powerlifter", con, if_exists="append", index=False)
    build_dim_division(perf).to_sql("dim_division", con, if_exists="append", index=False)
    build_dim_equipement(perf).to_sql("dim_equipement", con, if_exists="append", index=False)
    build_participation_rencontre(perf).to_sql(
        "participation_rencontre", con, if_exists="append", index=False
    )
    build_fait_performance(perf).to_sql(
        "fait_performance", con, if_exists="append", index=False
    )


def main():
    csv_dir = download_dataset()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        create_tables(SQL_PATH, con)
        load_data(con, csv_dir)
        con.commit()
    finally:
        con.close()


if __name__ == "__main__":
    main()
