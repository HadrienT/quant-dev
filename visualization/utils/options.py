from typing import Any

import numpy as np
import QuantLib as ql


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
