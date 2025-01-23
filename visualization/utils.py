import os
from typing import Any
import datetime

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
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/service-account-key.json"

calendar = ql.UnitedStates(ql.UnitedStates.NYSE)


def get_stock_performance_table(
    prices: pd.DataFrame,
    sorted_assets: list,
    sorted_weights: list,
    ql_start_date: ql.Date,
):
    """
    Create a table summarizing stock performance based on sorted assets and weights.

    Args:
        prices (pd.DataFrame): DataFrame containing stock prices.
        sorted_assets (list): List of sorted assets.
        sorted_weights (list): List of corresponding weights for the assets.
        ql_start_date (ql.Date): Start date for performance calculation.

    Returns:
        pd.DataFrame: Summary DataFrame with performance metrics.
    """
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
    
    columns_to_drop = ["ABNB", "AMTM", "CEG", "GEHC", "GEV", "KVUE", "SOLV", "SW", "VLTO", "TNX"]
    
    prices = prices.drop(columns=[col for col in columns_to_drop if col in prices.columns], errors="ignore")
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


def calculate_capm_metrics(
    portfolio_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float
) -> tuple[float, float, Any]:
    """
    Calculate CAPM metrics (Alpha and Beta).

    Args:
        portfolio_returns (pd.Series): Portfolio returns.
        market_returns (pd.Series): Market returns.
        risk_free_rate (float): Risk-free rate.

    Returns:
        tuple[float, float, Any]: Alpha, Beta, and model summary.
    """
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
    """
    Retrieve the latest risk-free rate from the FRED API.

    Returns:
        float: Risk-free rate.
    """
    fred = Fred(api_key=fred_api_key)
    ten_year_treasury_rate = fred.get_series_latest_release("GS10") / 100
    return ten_year_treasury_rate.iloc[-1]


def is_business_day(date) -> bool:
    """
    Check if a given date is a business day.

    Args:
        date (pd.Timestamp): Date to check.

    Returns:
        bool: True if business day, False otherwise.
    """
    ql_date = ql.Date(date.day, date.month, date.year)
    return calendar.isBusinessDay(ql_date)


def portfolio_return(weights:np.array, mean_returns:np.ndarray) -> float:
    """
    Calculate the expected return of a portfolio.

    Args:
        weights (np.array): Asset weights.
        mean_returns (np.ndarray): Mean returns of assets.

    Returns:
        float: Portfolio return.
    """
    return np.sum(mean_returns * weights)


def portfolio_volatility(weights:np.array, cov_matrix:np.ndarray) -> float:
    """
    Calculate portfolio volatility.

    Args:
        weights (np.array): Asset weights.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.

    Returns:
        float: Portfolio volatility.
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def negative_sharpe_ratio(weights:np.array, mean_returns:np.ndarray, cov_matrix:np.ndarray, risk_free_rate:float) -> float:
    """
    Calculate the negative Sharpe ratio of a portfolio.

    Args:
        weights (np.array): Asset weights.
        mean_returns (np.ndarray): Mean returns of assets.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        risk_free_rate (float): Risk-free rate.

    Returns:
        float: Negative Sharpe ratio.
    """
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
    """
    Optimize a portfolio using Sharpe ratio maximization.

    Args:
        returns (pd.DataFrame): Asset returns.
        risk_free_rate (float): Risk-free rate.
        max_share (float): Maximum allowable weight for any asset.
        start_date (ql.Date, optional): Start date for returns. Defaults to None.
        end_date (ql.Date, optional): End date for returns. Defaults to None.

    Returns:
        Any: Optimization result.
    """
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
    """
    Calculate daily returns for a DataFrame of prices.

    Args:
        df (pd.DataFrame): DataFrame containing asset prices.

    Returns:
        pd.DataFrame: DataFrame of daily returns.
    """
    business_day_df = df[df.index.map(lambda x: is_business_day(x))]
    returns = np.log(business_day_df / business_day_df.shift(1))
    return returns


def filter_portfolio(
    optimal_weights: np.array, assets: list, threshold: float = 0.005
) -> tuple[list, list]:
    """
    Filter portfolio assets based on a weight threshold.

    Args:
        optimal_weights (np.array): Array of asset weights.
        assets (list): List of asset names.
        threshold (float, optional): Minimum weight to retain asset. Defaults to 0.005.

    Returns:
        tuple[list, list]: Filtered assets and corresponding weights.
    """
    filtered_indices = optimal_weights >= threshold
    filtered_assets = [assets[i] for i in range(len(assets)) if filtered_indices[i]]
    filtered_weights = optimal_weights[filtered_indices]
    sorted_indices = np.argsort(filtered_weights)[::-1]
    sorted_assets = [filtered_assets[i] for i in sorted_indices]
    sorted_weights = filtered_weights[sorted_indices]
    return sorted_assets, sorted_weights


def plot_portfolio(sorted_assets: list, sorted_weights: list) -> None:
    """
    Plot the weights of an optimized portfolio.

    Args:
        sorted_assets (list): List of asset names.
        sorted_weights (list): List of corresponding weights.
    """
    plt.bar(sorted_assets, sorted_weights)
    plt.title("Optimal Portfolio Weights")
    plt.ylabel("Weight")
    plt.xlabel("Assets")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def get_latest_available_date(date: ql.Date, prices: pd.DataFrame) -> ql.Date:
    """
    Find the latest available date for price data relative to a given date.

    Args:
        date (ql.Date): Target date.
        prices (pd.DataFrame): DataFrame containing price data.

    Returns:
        ql.Date: Closest available date.
    """
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
    selling_date: ql.Date = None,
) -> pd.Series:
    """
    Track the value of a portfolio over time.

    Args:
        purchase_date (ql.Date): Date of purchase.
        prices (pd.DataFrame): DataFrame of asset prices.
        sorted_assets (list): List of asset names.
        sorted_weights (list): List of corresponding weights.
        initial_capital (int): Initial capital allocated.
        selling_date (ql.Date, optional): Date of sale. Defaults to None.
    Returns:
        pd.Series: Time series of portfolio value.
    """
    purchase_date = pd.Timestamp(purchase_date.to_date()).tz_localize("UTC")
    filtered_prices = prices[sorted_assets].loc[purchase_date:]
    if selling_date is not None:
        selling_date = pd.Timestamp(selling_date.to_date()).tz_localize("UTC")
        filtered_prices = filtered_prices.loc[:selling_date]
        
    initial_allocations = sorted_weights * initial_capital
    initial_prices = filtered_prices[sorted_assets].loc[purchase_date]
    shares = initial_allocations / initial_prices
    portfolio_values = filtered_prices.dot(shares)
    return portfolio_values


def sharpe_ratio_portfolio(
    purchase_date: ql.Date,
    prices: pd.DataFrame,
    sorted_assets: list,
    sorted_weights: list,
    risk_free_rate: float,
    selling_date: ql.Date = None,
) -> float:
    """
    Calculate the Sharpe ratio for an optimized portfolio.

    Args:
        purchase_date (ql.Date): Date of purchase.
        prices (pd.DataFrame): DataFrame of asset prices.
        sorted_assets (list): List of asset names.
        sorted_weights (list): List of corresponding weights.
        risk_free_rate (float): Risk-free rate.
        selling_date (ql.Date, optional): Date of sale. Defaults to None.
    Returns:
        float: Sharpe ratio.
    """
    purchase_date = pd.Timestamp(purchase_date.to_date()).tz_localize("UTC")
    filtered_prices = prices[sorted_assets].loc[purchase_date:]
    if selling_date is not None:
        selling_date = pd.Timestamp(selling_date.to_date()).tz_localize("UTC")
        filtered_prices = filtered_prices.loc[:selling_date]
    
    returns = get_returns(filtered_prices)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    sharpe_ratio = -negative_sharpe_ratio(
        weights=np.array(sorted_weights),
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_rate / 252
    )
    return sharpe_ratio


def metrics(portfolio_values: pd.Series) -> dict:
    """
    Calculate portfolio performance metrics.

    Args:
        portfolio_values (pd.Series): Time series of portfolio values.

    Returns:
        dict: Dictionary containing performance metrics.
    """
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
    option_type: Any,
    strike: float,
    market_price: float,
    spot_price: float,
    risk_free_rate: float,
    dividend_yield: float,
    expiry: float,
) -> float:
    """
    Calculate the implied volatility of a European option using QuantLib.

    Parameters:
    -----------
    option_type : ql.Option.Type
        The type of the option (e.g., ql.Option.Call or ql.Option.Put).
    strike : float
        The strike price of the option.
    market_price : float
        The observed market price of the option.
    spot_price : float
        The current price of the underlying asset.
    risk_free_rate : float
        The continuously compounded risk-free interest rate (annualized).
    dividend_yield : float
        The continuously compounded dividend yield (annualized).
    expiry : float
        Time to expiration in years.

    Returns:
    --------
    float
        The implied volatility of the option. If the input parameters are invalid 
        (e.g., non-positive market price, strike, or expiry), or if the implied 
        volatility cannot be calculated, returns np.nan.

    Notes:
    ------
    This function uses QuantLib to model the European option and employs the 
    Black-Scholes-Merton framework for calculating implied volatility.
    """
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


def rebalance_portfolio(
    prices: pd.DataFrame,
    selected_tickers: list,
    initial_capital: float,
    rebalance_freq: int,
    risk_free_rate: float,
    max_share: float,
    start_date: ql.Date,
    end_date: ql.Date = None,  # End date is optional
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Rebalances the portfolio every rebalance_freq business days and calculates Sharpe Ratios dynamically.

    Args:
        prices (pd.DataFrame): Asset price data.
        selected_tickers (list): List of selected assets.
        initial_capital (float): Initial allocated capital.
        rebalance_freq (int): Rebalancing frequency in business days. If 0, no rebalancing is performed.
        risk_free_rate (float): Risk-free rate.
        max_share (float): Maximum allocation to a single asset.
        start_date (datetime.date): Start date of the investment.
        end_date (datetime.date, optional): End date of the investment. Defaults to the last date in the data.

    Returns:
        tuple[pd.Series, pd.DataFrame, pd.Series]: 
        - Time series of portfolio values.
        - Table of allocations over time.
        - Time series of Sharpe Ratios.
    """
    ql_start_date = start_date
    ql_end_date = end_date
    start_date = pd.Timestamp(start_date.to_date()).tz_localize("UTC")
    if end_date is None:
        end_date = prices.index.max()  # Use the last available date in the data
    else:
        end_date = pd.Timestamp(end_date.to_date()).tz_localize("UTC")

    # Filter prices for dates and selected assets
    filtered_prices = prices[selected_tickers].loc[start_date:end_date]
    
    returns_to_date = get_returns(prices[selected_tickers]).loc[start_date:end_date]
    # Handle the no-rebalancing case
    if rebalance_freq == 0:
        # Calculate initial optimal weights
        result = get_portfolio(
            returns=returns_to_date,
            risk_free_rate=risk_free_rate,
            max_share=max_share,
            start_date=ql_start_date,
            end_date=ql_end_date,
        )
        optimal_weights = result.x
        print(f"Optimal Weights: {optimal_weights}")
        
        sorted_assets, sorted_weights = filter_portfolio(
                optimal_weights, selected_tickers, threshold=0.005
            )
        print(f"Sorted Assets: {sorted_assets}")
        # Track the portfolio value over time
        portfolio_values = track_portfolio_value(
            purchase_date=ql_end_date,
            prices=prices,
            sorted_assets=sorted_assets,
            sorted_weights=sorted_weights,
            initial_capital=initial_capital,
            selling_date=None,
        )

        # Allocation history
        allocation_history = [{
            "Date": start_date,
            "Portfolio Value": initial_capital,
            **{ticker: weight for ticker, weight in zip(selected_tickers, optimal_weights)}
        }]

        sharpe_ratio = -result.fun * np.sqrt(252)
        sharpe_ratios = pd.Series({start_date: sharpe_ratio})

        return portfolio_values, pd.DataFrame(allocation_history).set_index("Date"), sharpe_ratios

    # Find all business days
    business_days = [date for date in returns_to_date.index if is_business_day(date)]
    rebalance_dates = business_days[::rebalance_freq]  # Select every rebalance_freq-th business day

    # Initialization
    portfolio_values = [initial_capital]  # Initial capital as the first portfolio value
    portfolio_dates = [filtered_prices.index[0]]  # Store the first date
    allocation_history = []
    sharpe_ratios = pd.Series(dtype=float)
    last_rebalance_date = start_date  # Initialize to the start date

    for i, current_date in enumerate(filtered_prices.index):
        # Rebalance at the specified frequency
        if current_date in rebalance_dates or i == 0:
            returns_to_date = get_returns(prices[selected_tickers]).loc[start_date:current_date]

            if returns_to_date.empty:
                print("WARNING: Returns to date is empty. Skipping rebalancing.")
                continue

            result = get_portfolio(
                returns=returns_to_date,
                risk_free_rate=risk_free_rate,
                max_share=max_share,
                start_date=ql.Date(last_rebalance_date.day, last_rebalance_date.month, last_rebalance_date.year),
                end_date=ql.Date(current_date.day, current_date.month, current_date.year),
            )
            optimal_weights = result.x
            allocation_history.append({
                "Date": current_date,
                "Portfolio Value": portfolio_values[-1],
                **{ticker: weight for ticker, weight in zip(selected_tickers, optimal_weights)}
            })
            sharpe_ratios.loc[current_date] = -result.fun * np.sqrt(252)

            sorted_assets, sorted_weights = filter_portfolio(
                optimal_weights, selected_tickers, threshold=0.005
            )
            # Update the portfolio value using `track_portfolio_value`
            ql_last_rebalance_date = ql.Date(last_rebalance_date.day, last_rebalance_date.month, last_rebalance_date.year)
            ql_current_date = ql.Date(current_date.day, current_date.month, current_date.year)

            tracked_values = track_portfolio_value(
                purchase_date=ql_last_rebalance_date,
                prices=prices,
                sorted_assets=sorted_assets,
                sorted_weights=sorted_weights,
                initial_capital=portfolio_values[-1],  # Use the latest portfolio value as the starting capital
                selling_date=ql_current_date
            )

            # Append the tracked values and their corresponding dates
            portfolio_values.extend(tracked_values[1:])
            portfolio_dates.extend(filtered_prices.loc[last_rebalance_date:current_date].index[1:])

            # Update the last rebalance date
            last_rebalance_date = current_date

    # Time series of portfolio values with their corresponding dates
    portfolio_values_series = pd.Series(portfolio_values, index=portfolio_dates)

    # Allocation history
    allocation_history_df = pd.DataFrame(allocation_history).set_index("Date")

    # Filter out NaN values from Sharpe Ratios
    sharpe_ratios = sharpe_ratios.dropna()

    return portfolio_values_series, allocation_history_df, sharpe_ratios
