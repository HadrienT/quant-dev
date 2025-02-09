import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))

from main import get_sp500_tickers


def test_get_sp500_tickers():
    tickers = get_sp500_tickers()
    assert len(tickers) == 519
