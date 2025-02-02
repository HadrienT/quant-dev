from typing import Any

import pandas as pd
import streamlit as st
from statsmodels.tsa.stattools import adfuller, coint


def adf_test(series: pd.Series) -> dict[str, Any]:
    """
    Wrapper to perform the Augmented Dickey-Fuller test to check for stationarity.

    Args:
        series (pd.Series): Time series data to test.

    Returns:
        dict[str, Any]: Results of the ADF test.
    """
    result = adfuller(series, autolag="AIC")
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "used_lag": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "ic_best": result[5],
    }


def stationarity_test(series: pd.Series) -> None:
    """
    Display stationarity test results using Streamlit.

    Args:
        series (pd.Series): Time series data to test.
    """
    st.title("Augmented Dickey-Fuller Test")
    # Perform the ADF test
    adf_results = adf_test(series)

    # Display the results
    st.write("### ADF Test Results")
    st.write(f"**ADF Statistic**: {adf_results['ADF Statistic']:.4f}")
    st.write(f"**p-value**: {adf_results['p-value']:.4f}")
    st.write("**Critical Values**:")
    for key, value in adf_results["critical_values"].items():
        st.write(f"  - {key}: {value:.4f}")

    # Check stationarity
    significance_level = 0.05
    is_stationary = False

    # Decision based on p-value
    if adf_results["p-value"] < significance_level:
        p_value_decision = True
    else:
        p_value_decision = False

    # Decision based on critical values
    critical_value_decision = (
        adf_results["ADF Statistic"] < adf_results["critical_values"]["5%"]
    )

    # Combine decisions
    if p_value_decision and critical_value_decision:
        is_stationary = True

    # Display decision
    if is_stationary:
        st.success(
            "The series is statistically stationary. "
            "Both the p-value < 0.05 and the ADF Statistic is below the 5% critical value."
        )
    elif p_value_decision:
        st.warning(
            "The series may be stationary (p-value < 0.05), "
            "but the ADF Statistic does not pass the critical value test."
        )
    elif critical_value_decision:
        st.warning(
            "The series may be stationary based on the critical values, "
            "but the p-value does not indicate stationarity."
        )
    else:
        st.error(
            "The series is NOT statistically stationary. "
            "Both the p-value and the ADF Statistic fail the stationarity tests."
        )


def calculate_coint(stock_pair: tuple[str], data: pd.DataFrame) -> dict[str, float]:
    """
    Calculate cointegration metrics for a pair of stocks.

    Args:
        stock_pair (tuple[str]): Pair of stock tickers.
        data (pd.DataFrame): DataFrame containing stock prices.

    Returns:
        dict[str, float]: Cointegration score and p-value.
    """
    stockA, stockB = stock_pair
    score, p_value, _ = coint(data[stockA], data[stockB])
    return {"score": score, "p_value": p_value}
