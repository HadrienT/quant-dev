from typing import Any

import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.optimize import minimize

import utils.portfolio
import utils.market_calendar


def negative_sharpe_ratio(
    weights: np.array,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> float:
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
    p_return = utils.portfolio_return(weights, mean_returns)
    p_volatility = utils.portfolio_volatility(weights, cov_matrix)
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

    returns_to_date = utils.get_returns(prices[selected_tickers]).loc[
        start_date:end_date
    ]
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
        portfolio_values = utils.track_portfolio_value(
            purchase_date=ql_end_date,
            prices=prices,
            sorted_assets=sorted_assets,
            sorted_weights=sorted_weights,
            initial_capital=initial_capital,
            selling_date=None,
        )

        # Allocation history
        allocation_history = [
            {
                "Date": start_date,
                "Portfolio Value": initial_capital,
                **{
                    ticker: weight
                    for ticker, weight in zip(selected_tickers, optimal_weights)
                },
            }
        ]

        sharpe_ratio = -result.fun * np.sqrt(252)
        sharpe_ratios = pd.Series({start_date: sharpe_ratio})

        return (
            portfolio_values,
            pd.DataFrame(allocation_history).set_index("Date"),
            sharpe_ratios,
        )

    # Find all business days
    business_days = [
        date for date in returns_to_date.index if utils.is_business_day(date)
    ]
    rebalance_dates = business_days[
        ::rebalance_freq
    ]  # Select every rebalance_freq-th business day

    # Initialization
    portfolio_values = [initial_capital]  # Initial capital as the first portfolio value
    portfolio_dates = [filtered_prices.index[0]]  # Store the first date
    allocation_history = []
    sharpe_ratios = pd.Series(dtype=float)
    last_rebalance_date = start_date  # Initialize to the start date

    for i, current_date in enumerate(filtered_prices.index):
        # Rebalance at the specified frequency
        if current_date in rebalance_dates or i == 0:
            returns_to_date = utils.get_returns(prices[selected_tickers]).loc[
                start_date:current_date
            ]

            if returns_to_date.empty:
                print("WARNING: Returns to date is empty. Skipping rebalancing.")
                continue

            result = get_portfolio(
                returns=returns_to_date,
                risk_free_rate=risk_free_rate,
                max_share=max_share,
                start_date=ql.Date(
                    last_rebalance_date.day,
                    last_rebalance_date.month,
                    last_rebalance_date.year,
                ),
                end_date=ql.Date(
                    current_date.day, current_date.month, current_date.year
                ),
            )
            optimal_weights = result.x
            allocation_history.append(
                {
                    "Date": current_date,
                    "Portfolio Value": portfolio_values[-1],
                    **{
                        ticker: weight
                        for ticker, weight in zip(selected_tickers, optimal_weights)
                    },
                }
            )
            sharpe_ratios.loc[current_date] = -result.fun * np.sqrt(252)

            sorted_assets, sorted_weights = filter_portfolio(
                optimal_weights, selected_tickers, threshold=0.005
            )
            # Update the portfolio value using `track_portfolio_value`
            ql_last_rebalance_date = ql.Date(
                last_rebalance_date.day,
                last_rebalance_date.month,
                last_rebalance_date.year,
            )
            ql_current_date = ql.Date(
                current_date.day, current_date.month, current_date.year
            )

            tracked_values = utils.track_portfolio_value(
                purchase_date=ql_last_rebalance_date,
                prices=prices,
                sorted_assets=sorted_assets,
                sorted_weights=sorted_weights,
                initial_capital=portfolio_values[
                    -1
                ],  # Use the latest portfolio value as the starting capital
                selling_date=ql_current_date,
            )

            # Append the tracked values and their corresponding dates
            portfolio_values.extend(tracked_values[1:])
            portfolio_dates.extend(
                filtered_prices.loc[last_rebalance_date:current_date].index[1:]
            )

            # Update the last rebalance date
            last_rebalance_date = current_date

    # Time series of portfolio values with their corresponding dates
    portfolio_values_series = pd.Series(portfolio_values, index=portfolio_dates)

    # Allocation history
    allocation_history_df = pd.DataFrame(allocation_history).set_index("Date")

    # Filter out NaN values from Sharpe Ratios
    sharpe_ratios = sharpe_ratios.dropna()

    return portfolio_values_series, allocation_history_df, sharpe_ratios
