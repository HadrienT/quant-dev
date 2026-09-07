"""Market data retrieval from yfinance.

This module knows nothing about where the data ends up; it only produces a
DataFrame with the columns listed in config.COLUMNS.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

from config import (
    CHUNK_SIZE,
    COLUMNS,
    MARKET_CALENDAR,
    MARKET_TZ,
    MAX_RETRIES,
    RETRY_BACKOFF_SECONDS,
    TICKERS_PATH,
)

logger = logging.getLogger("quant-dev.ingestion")


def get_sp500_tickers() -> List[str]:
    return pd.read_csv(TICKERS_PATH, header=None)[0].dropna().astype(str).tolist()


def _chunked(items: Iterable[str], size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _last_trading_day(reference_dt: Optional[datetime] = None, calendar_name: str = MARKET_CALENDAR) -> datetime:
    now = reference_dt or datetime.now(ZoneInfo(MARKET_TZ))
    start = (now - timedelta(days=10)).date()
    end = now.date()
    try:
        calendar = mcal.get_calendar(calendar_name)
        schedule = calendar.schedule(start_date=start, end_date=end)
        if schedule.empty:
            raise ValueError("Empty market schedule")
        last_session = schedule.index[-1].to_pydatetime().date()
        return datetime.combine(last_session, datetime.min.time(), tzinfo=ZoneInfo(MARKET_TZ))
    except Exception as exc:
        logger.warning("Falling back to weekday logic: %s", exc)
        current_date = now - timedelta(days=1)
        while current_date.isoweekday() not in range(1, 6):
            current_date -= timedelta(days=1)
        return current_date


def _download_yfinance_chunk(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = yf.download(
                tickers=" ".join(tickers),
                start=start_date,
                end=end_date,
                interval="1d",
                group_by="ticker",
                threads=True,
            )
            if not data.empty:
                return data
        except Exception as exc:
            last_error = exc
            logger.warning("Download attempt %s failed: %s", attempt, exc)
        time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    if last_error:
        logger.error("Download failed after %s attempts: %s", MAX_RETRIES, last_error)
    return pd.DataFrame()


def _normalize_yfinance_data(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        data = df.stack(level=0, future_stack=True).reset_index()
    else:
        data = df.reset_index().copy()
        ticker = tickers[0] if tickers else "UNKNOWN"
        data["Ticker"] = ticker

    data = data[[col for col in COLUMNS if col in data.columns]]
    data["Date"] = pd.to_datetime(data["Date"]).dt.date
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    if "Volume" in data.columns:
        data["Volume"] = data["Volume"].fillna(0).astype("int64")
    return data


def _download_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for batch in _chunked(tickers, CHUNK_SIZE):
        raw = _download_yfinance_chunk(batch, start_date, end_date)
        normalized = _normalize_yfinance_data(raw, batch)
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["Date", "Ticker"])
    data = data.drop_duplicates(subset=["Date", "Ticker"], keep="last")
    return data


def download_previous_day_data(tickers: List[str], reference_dt: Optional[datetime] = None) -> pd.DataFrame:
    """Downloads daily data for the tickers from the previous trading day."""
    last_session = _last_trading_day(reference_dt)
    start_date = last_session.strftime("%Y-%m-%d")
    end_date = (last_session + timedelta(days=1)).strftime("%Y-%m-%d")
    data = _download_prices(tickers, start_date, end_date)
    if data.empty:
        logger.info("No data available for %s", start_date)
    return data


def download_sp500_data(tickers: List[str], start_date: str = "2000-01-01", end_date: Optional[str] = None) -> pd.DataFrame:
    """Downloads daily data for all S&P 500 tickers."""
    end_date = end_date or datetime.now(ZoneInfo(MARKET_TZ)).strftime("%Y-%m-%d")
    return _download_prices(tickers, start_date, end_date)
