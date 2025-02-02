import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from fredapi import Fred

from .config import fred_api_key


@st.cache_data
def get_stock_tickers():
    return pd.read_csv("tickers.csv", header=None)[0].tolist()


# def load_data() -> pd.DataFrame:
#     df1 = pd.read_csv("../data/sp500_data.csv")
#     df2 = pd.read_csv("../data/sp500_stock_index_data.csv")
#     combined_df = pd.concat([df1, df2]).drop_duplicates(["Date", "Ticker"])

#     return combined_df


def load_data() -> pd.DataFrame:
    """
    Load financial data from BigQuery.

    Returns:
        pd.DataFrame: Sorted financial data.
    """
    client = bigquery.Client()

    PROJECT_ID = "quant-dev-442615"
    DATASET_ID = "financial_data"

    query = f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET_ID}.sp500_data`
    WHERE Date > '2020-01-01'
    """

    df = client.query(query).to_dataframe()

    df = df.drop_duplicates(subset=["Date", "Ticker"])

    return df.sort_values(by=["Date"])


@st.cache_data
def load_prices(special_filters: list[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and transform asset price data.

    Args:
        special_filters (list[str], optional): List of tickers to filter. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Raw data and filtered price data.
    """
    df = load_data()
    prices = df.pivot(index="Date", columns="Ticker", values="Close")
    prices = prices.infer_objects().interpolate(method="linear")

    columns_to_drop = [
        "ABNB",
        "AMTM",
        "CEG",
        "GEHC",
        "GEV",
        "KVUE",
        "SOLV",
        "SW",
        "VLTO",
        "TNX",
    ]

    prices = prices.drop(
        columns=[col for col in columns_to_drop if col in prices.columns],
        errors="ignore",
    )
    prices.index = pd.to_datetime(prices.index).tz_localize("UTC")

    if special_filters is not None:
        return df, prices[special_filters]
    return df, prices.dropna()


@st.cache_data
def calculate_sp500_returns(prices):
    """
    Calculate daily returns of the S&P 500.

    Args:
        prices (pd.DataFrame): DataFrame containing price data.

    Returns:
        pd.Series: Daily returns of the S&P 500.
    """
    if "^GSPC" not in prices.columns:
        st.warning("The ticker ^GSPC is not present in the data.")
        return None
    return np.log(prices["^GSPC"] / prices["^GSPC"].shift(1))


@st.cache_data
def get_risk_free_rate() -> float:
    """
    Retrieve the latest risk-free rate from the FRED API.

    Returns:
        float: Risk-free rate.
    """
    fred = Fred(api_key=fred_api_key)
    ten_year_treasury_rate = fred.get_series_latest_release("GS10") / 100
    return ten_year_treasury_rate.iloc[-1]
