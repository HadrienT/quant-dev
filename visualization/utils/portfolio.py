import numpy as np
import pandas as pd
import QuantLib as ql

import utils.market_calendar
import utils.optimization


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

    # Create a summary DataFrame with rounded values
    summary_df = pd.DataFrame(
        {
            "Ticker": sorted_assets,
            "Start Price": start_prices.values.round(2),
            "End Price": end_prices.values.round(2),
            "Total Return (%)": total_returns.values.round(2),
            "Weight (%)": [round(w * 100, 2) for w in sorted_weights],
        }
    ).set_index("Ticker")

    return summary_df


def portfolio_return(weights: np.array, mean_returns: np.ndarray) -> float:
    """
    Calculate the expected return of a portfolio.

    Args:
        weights (np.array): Asset weights.
        mean_returns (np.ndarray): Mean returns of assets.

    Returns:
        float: Portfolio return.
    """
    return np.sum(mean_returns * weights)


def portfolio_volatility(weights: np.array, cov_matrix: np.ndarray) -> float:
    """
    Calculate portfolio volatility.

    Args:
        weights (np.array): Asset weights.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.

    Returns:
        float: Portfolio volatility.
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


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

    sharpe_ratio = -utils.negative_sharpe_ratio(
        weights=np.array(sorted_weights),
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_rate / 252,
    )
    return sharpe_ratio


def get_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns for a DataFrame of prices.

    Args:
        df (pd.DataFrame): DataFrame containing asset prices.

    Returns:
        pd.DataFrame: DataFrame of daily returns.
    """
    business_day_df = df[df.index.map(lambda x: utils.is_business_day(x))]
    returns = np.log(business_day_df / business_day_df.shift(1))
    return returns
