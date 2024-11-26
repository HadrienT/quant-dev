import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from google.cloud import bigquery


# Télécharger la liste des tickers du S&P 500
def get_sp500_tickers():
    """Récupère la liste des tickers du S&P 500 via Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)
    sp500_table = table[0]  # La table principale est la première
    return sp500_table["Symbol"].tolist()


# Télécharger les données du jour précédent
def download_previous_day_data(tickers):
    """
    Télécharge les données daily pour les tickers du jour précédent.
    """
    # start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    # end_date = (datetime.now() - timedelta(days=0)).strftime("%Y-%m-%d")

    current_date = datetime.now() - timedelta(days=1)
    while not pd.Timestamp(current_date).isoweekday() in range(
        1, 6
    ):  # Lundi (1) à Vendredi (5)
        current_date -= timedelta(days=1)

    start_date = current_date.strftime("%Y-%m-%d")
    end_date = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(
        tickers=" ".join(tickers),
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",
        threads=True,
    )
    if data.empty:
        print(f"Aucune donnée disponible pour {start_date}.")
        return pd.DataFrame()  # Retourne un DataFrame vide si aucune donnée
    # Restructurer les données
    data = data.stack(level=0, future_stack=True).reset_index()
    data.columns = [
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]
    return data


# Télécharger les données pour tous les tickers
def download_sp500_data(tickers, start_date="2000-01-01", end_date=None):
    """
    Télécharge les données daily de tous les tickers du S&P 500.
    """
    # Obtenir les données via yfinance
    data = yf.download(
        tickers=" ".join(tickers),
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",  # Organise les données par ticker
        threads=True,  # Téléchargement parallèle
    )

    data = data.stack(
        level=0, future_stack=True
    ).reset_index()  # Rendre les tickers une colonne
    data.columns = [
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    return data


# Charger les données dans BigQuery
def load_data_to_bigquery(df, table_id, project_id):
    """
    Charge un DataFrame dans une table BigQuery.

    Arguments :
        df : pandas.DataFrame contenant les données à charger.
        table_id : ID complet de la table BigQuery (ex : dataset.table_name).
        project_id : ID du projet Google Cloud.
    """
    client = bigquery.Client(project=project_id)

    # Convertir les données au format BigQuery
    job = client.load_table_from_dataframe(df, table_id)

    # Attendre la fin du job
    job.result()
    print(f"Les données ont été chargées dans {table_id}")


# Charger les données dans une table temporaire
def load_to_temp_table(client, df, temp_table_id):
    """
    Charge les données dans une table temporaire BigQuery.
    """
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE"  # Écrase les données existantes
    )
    job = client.load_table_from_dataframe(df, temp_table_id, job_config=job_config)
    job.result()
    print(f"Données chargées dans la table temporaire {temp_table_id}")


# Fusionner les données avec la table principale
def merge_into_main_table(client, temp_table_id, main_table_id):
    """
    Fusionne les données de la table temporaire dans la table principale.
    """
    query = f"""
    MERGE `{main_table_id}` AS main
    USING `{temp_table_id}` AS temp
    ON main.Date = temp.Date AND main.Ticker = temp.Ticker
    WHEN NOT MATCHED THEN
      INSERT (Date, Ticker, Open, High, Low, Close, `Adj Close`, Volume)  -- `Adj Close` avec un espace
      VALUES (temp.Date, temp.Ticker, temp.Open, temp.High, temp.Low, temp.Close, temp.`Adj Close`, temp.Volume)
    """
    job = client.query(query)
    job.result()
    print(f"Données fusionnées dans la table principale {main_table_id}")


def fill_table():
    # Étape 1 : Récupérer les tickers
    print("Téléchargement des tickers du S&P 500...")
    sp500_tickers = get_sp500_tickers()
    print(f"Nombre de tickers récupérés : {len(sp500_tickers)}")

    # Étape 2 : Télécharger les données
    print("Téléchargement des données journalières...")
    sp500_data = download_sp500_data(sp500_tickers[0:1], start_date="2000-01-01")

    # Étape 3 : Charger dans BigQuery
    print("Chargement des données dans BigQuery...")
    PROJECT_ID = "quant-dev-442615"  # Remplacez par votre ID de projet
    DATASET_ID = "financial_data"  # Remplacez par le nom de votre dataset
    TABLE_ID = "sp500_data"  # Nom de la table

    # Charger les données
    load_data_to_bigquery(
        sp500_data,
        table_id=f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}",
        project_id=PROJECT_ID,
    )


def add_daily():
    # Étape 1 : Configuration
    PROJECT_ID = "quant-dev-442615"
    DATASET_ID = "financial_data"
    MAIN_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.sp500_data"
    TEMP_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.temp_sp500_data"

    # Récupérer les tickers
    print("Téléchargement des tickers du S&P 500...")
    tickers = get_sp500_tickers()

    # Étape 2 : Télécharger les données
    print("Téléchargement des données du jour précédent...")
    sp500_data = download_previous_day_data(tickers)
    if sp500_data.empty:
        print("Aucune donnée disponible. Fin de l'exécution.")
        return  # Arrête l'exécution si aucune donnée n'est disponible
    sp500_data["Date"] = pd.to_datetime(sp500_data["Date"]).dt.date
    # Convertir la colonne Volume en entier
    sp500_data["Volume"] = sp500_data["Volume"].fillna(0).astype(int)

    # Étape 3 : Charger dans BigQuery
    client = bigquery.Client(project=PROJECT_ID)
    print("Chargement des données dans une table temporaire...")
    load_to_temp_table(client, sp500_data, TEMP_TABLE_ID)

    # Étape 4 : Fusionner avec la table principale
    print("Fusion des données avec la table principale...")
    merge_into_main_table(client, TEMP_TABLE_ID, MAIN_TABLE_ID)


if __name__ == "__main__":
    # fill_table()
    add_daily()
