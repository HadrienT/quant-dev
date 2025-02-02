import pandas as pd
import QuantLib as ql

from .config import calendar


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
