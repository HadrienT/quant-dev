import pandas as pd
import yfinance as yf


def create_sample_multiindex_df():
    """Creates a sample DataFrame with a MultiIndex matching the expected structure."""
    data = {
        ("AAPL", "Open"): [130.788116],
        ("AAPL", "High"): [132.382025],
        ("AAPL", "Low"): [128.030558],
        ("AAPL", "Close"): [131.169479],
        ("AAPL", "Volume"): [106239800],
        ("MSFT", "Open"): [227.213221],
        ("MSFT", "High"): [234.404861],
        ("MSFT", "Low"): [224.671011],
        ("MSFT", "Close"): [231.649994],
        ("MSFT", "Volume"): [33314200],
    }

    index = pd.to_datetime(["2021-02-01"])  # Sample date
    df = pd.DataFrame(data, index=index)

    # Convert to MultiIndex
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Ticker", "Price"])
    df.index.name = "Date"  # Name the index correctly
    return df


def test_multiindex_structure():
    """Tests if the DataFrame has the correct MultiIndex structure and data values."""

    # Create a reference DataFrame with expected data
    df_ref = create_sample_multiindex_df()

    # Download actual data using yfinance
    df = yf.download(
        tickers=" ".join(["AAPL", "MSFT"]),
        start="2021-02-01",
        end="2021-02-02",
        interval="1d",
        group_by="ticker",
        threads=True,
    )

    # Ensure the DataFrame columns use a MultiIndex
    assert isinstance(df.columns, pd.MultiIndex), "Columns are not using a MultiIndex."

    # Check the levels of the MultiIndex (ignoring order)
    expected_levels = [
        ["AAPL", "MSFT"],
        sorted(["Open", "High", "Low", "Close", "Volume"]),
    ]
    actual_levels = [
        sorted(df.columns.get_level_values(i).unique().tolist()) for i in range(2)
    ]
    print(f"Expected levels: {expected_levels}")
    print(f"Actual levels: {actual_levels}")
    assert (
        actual_levels == expected_levels
    ), f"MultiIndex levels are incorrect.\nExpected: {expected_levels}\nGot: {actual_levels}"

    # Verify the names of the MultiIndex levels
    assert df.columns.names == [
        "Ticker",
        "Price",
    ], "MultiIndex level names are incorrect."

    # Check if the columns are structured as expected (ignoring order)
    expected_columns = {
        ("AAPL", "Open"),
        ("AAPL", "High"),
        ("AAPL", "Low"),
        ("AAPL", "Close"),
        ("AAPL", "Volume"),
        ("MSFT", "Open"),
        ("MSFT", "High"),
        ("MSFT", "Low"),
        ("MSFT", "Close"),
        ("MSFT", "Volume"),
    }
    actual_columns = set(df.columns.to_list())
    assert (
        actual_columns == expected_columns
    ), f"Columns do not match the expected structure.\nExpected: {expected_columns}\nGot: {actual_columns}"

    # Ensure the index is correctly named and is a DatetimeIndex
    assert df.index.name == "Date", "Index is not named 'Date'."
    assert isinstance(df.index, pd.DatetimeIndex), "Index is not a DatetimeIndex."

    # Convert both DataFrames to have the same structure for comparison
    df_sorted = df[sorted(expected_columns)].sort_index(axis=0).round(3)
    df_ref_sorted = df_ref[sorted(expected_columns)].sort_index(axis=0).round(3)

    # Ensure all values in the dataset match
    assert df_sorted.equals(
        df_ref_sorted
    ), "Data values do not match the expected reference DataFrame."
