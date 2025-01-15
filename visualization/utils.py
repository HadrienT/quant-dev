import os
from typing import Any

from dotenv import load_dotenv
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import QuantLib as ql
from fredapi import Fred
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import streamlit as st
from google.cloud import bigquery


load_dotenv()
fred_api_key = os.getenv("FRED_APY_KEY")
calendar = ql.UnitedStates(ql.UnitedStates.NYSE)


def get_stock_performance_table(
    prices: pd.DataFrame,
    sorted_assets: list,
    sorted_weights: list,
    ql_start_date: ql.Date,
):
    # Convert QuantLib Dates to Timestamps
    start_date = pd.Timestamp(ql_start_date.to_date()).tz_localize("UTC")
    end_date = prices.index[-1]

    # Filter for retained stocks
    retained_prices = prices[sorted_assets]

    # Extract spot prices at the start and end date
    start_prices = retained_prices.loc[start_date]
    end_prices = retained_prices.loc[end_date]

    # Calculate total returns
    total_returns = ((end_prices - start_prices) / start_prices) * 100

    # Create a summary DataFrame
    summary_df = pd.DataFrame(
        {
            "Ticker": sorted_assets,
            "Start Price": start_prices.values,
            "End Price": end_prices.values,
            "Total Return (%)": total_returns.values,
            "Weight (%)": [w * 100 for w in sorted_weights],
        }
    ).set_index("Ticker")

    return summary_df


def adf_test(series: pd.Series) -> dict[str, Any]:
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
    stockA, stockB = stock_pair
    score, p_value, _ = coint(data[stockA], data[stockB])
    return {"score": score, "p_value": p_value}


# def load_data() -> pd.DataFrame:
#     df1 = pd.read_csv("../data/sp500_data.csv")
#     df2 = pd.read_csv("../data/sp500_stock_index_data.csv")
#     combined_df = pd.concat([df1, df2]).drop_duplicates(["Date", "Ticker"])

#     return combined_df

def load_data() -> pd.DataFrame:
    client = bigquery.Client()

    PROJECT_ID = "quant-dev-442615"
    DATASET_ID = "financial_data"
    
    query = f"""
    SELECT * 
    FROM `{PROJECT_ID}.{DATASET_ID}.sp500_data`
    WHERE Date > '2020-01-01'
    """
    
    combined_df = client.query(query).to_dataframe()

    combined_df = combined_df.drop_duplicates(subset=["Date", "Ticker"])

    return combined_df.sort_values(by=["Date"])

@st.cache_data
def load_prices(special_filters: list[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and transform asset price data."""
    df = load_data()
    prices = df.pivot(index="Date", columns="Ticker", values="Close")
    prices = prices.infer_objects().interpolate(method="linear")
    
    columns_to_drop = ["ABNB", "AMTM", "CEG", "GEHC", "GEV", "KVUE", "SOLV", "SW", "VLTO"]
    
    prices = prices.drop(columns=[col for col in columns_to_drop if col in prices.columns], errors="ignore")
    prices.index = pd.to_datetime(prices.index)
    
    if special_filters is not None:
        return df, prices[special_filters]
    return df, prices.dropna()



@st.cache_data
def calculate_sp500_returns(prices):
    """Calculate daily returns of the S&P 500."""
    if "^GSPC" not in prices.columns:
        st.warning("The ticker ^GSPC is not present in the data.")
        return None
    return np.log(prices["^GSPC"] / prices["^GSPC"].shift(1))


def calculate_capm_metrics(
    portfolio_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float
) -> tuple[float, float, Any]:
    portfolio_returns = portfolio_returns.dropna()
    market_returns = market_returns.dropna()
    # Adjust returns for the risk-free rate
    excess_portfolio_returns = portfolio_returns - risk_free_rate
    excess_market_returns = market_returns - risk_free_rate

    # Add a constant for the linear model
    X = sm.add_constant(excess_market_returns)
    y = excess_portfolio_returns

    # Linear regression model
    model = sm.OLS(y, X).fit()

    # Extract Alpha and Beta
    alpha = model.params["const"]  # Intercept
    beta = model.params[excess_market_returns.name]  # Slope coefficient

    return alpha, beta, model.summary()


@st.cache_data
def get_risk_free_rate() -> float:
    fred = Fred(api_key=fred_api_key)
    ten_year_treasury_rate = fred.get_series_latest_release("GS10") / 100
    return ten_year_treasury_rate.iloc[-1]


def is_business_day(date):
    ql_date = ql.Date(date.day, date.month, date.year)
    return calendar.isBusinessDay(ql_date)


def portfolio_return(weights, mean_returns):
    return np.sum(mean_returns * weights)


def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_return = portfolio_return(weights, mean_returns)
    p_volatility = portfolio_volatility(weights, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility


def get_portfolio(
    returns: pd.DataFrame,
    risk_free_rate: float,
    max_share: float,
    start_date: ql.Date = None,
    end_date: ql.Date = None,
) -> Any:
    work_returns = returns
    if start_date:
        start_date = pd.Timestamp(start_date.to_date()).tz_localize("UTC")
        work_returns = work_returns.loc[start_date:]
    if end_date:
        end_date = pd.Timestamp(end_date.to_date()).tz_localize("UTC")
        work_returns = work_returns.loc[:end_date]
    mean_returns = work_returns.mean()
    cov_matrix = work_returns.cov()

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = tuple((0, max_share) for _ in range(work_returns.shape[1]))
    num_assets = work_returns.shape[1]
    initial_weights = num_assets * [1.0 / num_assets]

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate / 252),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result


def get_returns(df: pd.DataFrame) -> pd.DataFrame:
    business_day_df = df[df.index.map(lambda x: is_business_day(x))]
    returns = np.log(business_day_df / business_day_df.shift(1))
    return returns


def filter_portfolio(
    optimal_weights: np.array, assets: list, threshold: float = 0.005
) -> tuple[list, list]:
    filtered_indices = optimal_weights >= threshold
    filtered_assets = [assets[i] for i in range(len(assets)) if filtered_indices[i]]
    filtered_weights = optimal_weights[filtered_indices]
    sorted_indices = np.argsort(filtered_weights)[::-1]
    sorted_assets = [filtered_assets[i] for i in sorted_indices]
    sorted_weights = filtered_weights[sorted_indices]
    return sorted_assets, sorted_weights


def plot_portfolio(sorted_assets: list, sorted_weights: list) -> None:
    plt.bar(sorted_assets, sorted_weights)
    plt.title("Optimal Portfolio Weights")
    plt.ylabel("Weight")
    plt.xlabel("Assets")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def get_latest_available_date(date: ql.Date, prices: pd.DataFrame) -> ql.Date:
    date = pd.Timestamp(date.to_date()).tz_localize("UTC")
    if date in prices.index:
        closest_date = date
    else:
        closest_index = prices.index.get_indexer([date], method="nearest")[0]
        closest_date = prices.index[closest_index]

    closest_ql_date = ql.Date(closest_date.day, closest_date.month, closest_date.year)
    return closest_ql_date


def track_portfolio_value(
    purchase_date: ql.Date,
    prices: pd.DataFrame,
    sorted_assets: list,
    sorted_weights: list,
    initial_capital: int,
) -> pd.Series:
    print(f"{purchase_date=}")
    print(f"{sorted_assets=}")
    print(f"{sorted_weights=}")
    print(f"{initial_capital=}")
    purchase_date = pd.Timestamp(purchase_date.to_date()).tz_localize("UTC")
    filtered_prices = prices[sorted_assets].loc[purchase_date:]
    initial_allocations = sorted_weights * initial_capital
    initial_prices = filtered_prices[sorted_assets].loc[purchase_date]
    shares = initial_allocations / initial_prices
    portfolio_values = filtered_prices.dot(shares)

    return portfolio_values


def metrics(portfolio_values: pd.Series) -> dict:
    portfolio_values = portfolio_values.dropna()
    ath_value = portfolio_values.max()
    atl_value = portfolio_values.min()
    initial_value = portfolio_values.iloc[0]
    final_value = portfolio_values.iloc[-1]
    total_return = (final_value / initial_value - 1) * 100
    num_days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
    annualized_return = ((final_value / initial_value) ** (365.0 / num_days) - 1) * 100
    running_max = portfolio_values.cummax()
    drawdowns = portfolio_values / running_max - 1
    max_drawdown = drawdowns.min()
    return {
        "ATH": ath_value,
        "ATL": atl_value,
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Max Drawdown": max_drawdown,
    }


def calculate_implied_volatility(
    option_type,
    strike,
    market_price,
    spot_price,
    risk_free_rate,
    dividend_yield,
    expiry,
):
    if market_price <= 0 or strike <= 0 or expiry <= 0:
        return np.nan

    payoff = ql.PlainVanillaPayoff(option_type, strike)
    exercise = ql.EuropeanExercise(ql.Date.todaysDate() + int(expiry * 365))

    european_option = ql.VanillaOption(payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(
            0,
            ql.NullCalendar(),
            ql.QuoteHandle(ql.SimpleQuote(risk_free_rate)),
            ql.Actual360(),
        )
    )
    dividend_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(
            0,
            ql.NullCalendar(),
            ql.QuoteHandle(ql.SimpleQuote(dividend_yield)),
            ql.Actual360(),
        )
    )
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(
            0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(0.2)), ql.Actual360()
        )
    )

    black_scholes_process = ql.BlackScholesMertonProcess(
        spot_handle, dividend_handle, rate_handle, vol_handle
    )

    try:
        implied_vol = european_option.impliedVolatility(
            market_price, black_scholes_process
        )
    except RuntimeError:
        implied_vol = np.nan

    return implied_vol
