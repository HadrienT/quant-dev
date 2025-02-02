from typing import Any

import pandas as pd
import statsmodels.api as sm


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
