import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import utils
import QuantLib as ql
import numpy as np
import datetime
import random

init = True


def dynamic_initialization():
    if "rebalance_freq" not in st.session_state:
        st.session_state["rebalance_freq"] = 0
    if "max_share" not in st.session_state:
        st.session_state["max_share"] = 0.3
    if "min_share" not in st.session_state:
        st.session_state["min_share"] = 0.05
    if "selected_tickers" not in st.session_state:
        st.session_state["selected_tickers"] = []
    if "start_date" not in st.session_state:
        st.session_state["start_date"] = datetime.date(2023, 1, 1)
    if "end_date" not in st.session_state:
        st.session_state["end_date"] = datetime.date(2024, 1, 1)
    if "stocks_selection" not in st.session_state:
        st.session_state["stocks_selection"] = []
    if "indexes_selection" not in st.session_state:
        st.session_state["indexes_selection"] = []
    if "minerals_selection" not in st.session_state:
        st.session_state["minerals_selection"] = []
    if "etf_selection" not in st.session_state:
        st.session_state["etf_selection"] = []
    if "crypto_selection" not in st.session_state:
        st.session_state["crypto_selection"] = []
    if "bonds_selection" not in st.session_state:
        st.session_state["bonds_selection"] = []
    if "example_selection" not in st.session_state:
        st.session_state["example_selection"] = None
    if "last_selected_example" not in st.session_state:
        st.session_state["last_selected_example"] = None
    return False


def randomize_selection(stocks, index, minerals, etf, crypto, bond):
    st.session_state["example_selection"] = None
    random_tickers = (
        random.sample(stocks, k=random.randint(0, min(len(stocks) // 2, 5)))
        + random.sample(index, k=random.randint(0, min(len(index) // 2, 5)))
        + random.sample(minerals, k=random.randint(0, min(len(minerals) // 2, 5)))
        + random.sample(etf, k=random.randint(0, min(len(etf) // 2, 5)))
        + random.sample(crypto, k=random.randint(0, min(len(crypto) // 2, 5)))
        + random.sample(bond, k=random.randint(0, min(len(bond) // 2, 5)))
    )

    random_start_date = datetime.date(
        random.randint(2021, 2023), random.randint(1, 12), random.randint(1, 28)
    )
    random_end_date = random_start_date + datetime.timedelta(
        days=random.randint(30, 365)
    )

    st.session_state.update(
        {
            "selected_tickers": random_tickers,
            "start_date": random_start_date,
            "end_date": random_end_date,
            "stocks_selection": [
                ticker for ticker in random_tickers if ticker in stocks
            ],
            "indexes_selection": [
                ticker for ticker in random_tickers if ticker in index
            ],
            "minerals_selection": [
                ticker for ticker in random_tickers if ticker in minerals
            ],
            "etf_selection": [ticker for ticker in random_tickers if ticker in etf],
            "crypto_selection": [
                ticker for ticker in random_tickers if ticker in crypto
            ],
            "bonds_selection": [ticker for ticker in random_tickers if ticker in bond],
            "rebalance_freq": random.randint(0, 10),
        }
    )


def reset_session_state():
    st.session_state.update(
        {
            "selected_tickers": [],
            "start_date": datetime.date(2023, 1, 1),
            "end_date": datetime.date(2024, 1, 1),
            "stocks_selection": [],
            "indexes_selection": [],
            "minerals_selection": [],
            "etf_selection": [],
            "crypto_selection": [],
            "bonds_selection": [],
            "rebalance_freq": 0,
            "max_share": 0.3,
            "min_share": 0.05,
        }
    )


def page_dynamicPortfolio():
    global init
    st.text(
        "Create a portfolio on the fly using selected tickers from various asset classes. The date range determines the period for optimization. The investment is considered to begin at the end of this period."
    )
    _, prices = utils.load_prices()
    risk_free_rate = utils.get_risk_free_rate()
    sp500_returns = utils.calculate_sp500_returns(prices)
    st.title("Make Your Portfolio")

    # --- List of tickers ---
    index = sorted(
        ["^GSPC", "^DJI", "^IXIC", "^FTSE", "^GDAXI", "^FCHI", "^N225", "URTH"]
    )
    minerals = sorted(["GC=F", "SI=F", "CL=F", "BZ=F", "HG=F"])
    etf = sorted(["SPY", "QQQ", "VTI", "EEM", "IWM", "GLD", "TLT"])
    crypto = sorted(["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD"])
    bond = sorted(["^TYX"])

    top_stocks = sorted(
        ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BRK-B", "NVDA", "META"]
    )
    all_stocks = utils.get_stock_tickers()
    stocks = top_stocks + [
        stock
        for stock in all_stocks
        if stock not in top_stocks + minerals + etf + crypto + bond + index
    ]

    st.markdown("### Example Portfolios")
    example_portfolios = {
        "Example 1": {
            "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "max_share": 0.4,
            "start_date": datetime.date(2023, 1, 1),
            "end_date": datetime.date(2024, 1, 1),
        },
        "Example 2": {
            "tickers": ["SPY", "QQQ", "GLD", "TLT", "BTC-USD"],
            "max_share": 0.5,
            "start_date": datetime.date(2021, 6, 1),
            "end_date": datetime.date(2023, 1, 1),
        },
        "Example 3": {
            "tickers": ["NVDA", "META", "ETH-USD", "BNB-USD", "GC=F"],
            "max_share": 0.6,
            "start_date": datetime.date(2023, 12, 1),
            "end_date": datetime.date(2024, 9, 1),
        },
        "Example 4": {
            "tickers": ["BRK-B", "^DJI", "TLT", "GLD", "HG=F"],
            "max_share": 0.3,
            "start_date": datetime.date(2021, 1, 1),
            "end_date": datetime.date(2022, 1, 1),
        },
    }

    # Initialize session state for dynamic updates
    if init:
        init = dynamic_initialization()

    selected_example = st.radio(
        "Select a predefined portfolio (optional):",
        options=[None] + list(example_portfolios.keys()),
        horizontal=True,
    )

    if "last_selected_example" not in st.session_state:
        st.session_state["last_selected_example"] = None

    # Update session state if a new example is selected otherwise the last selected example will be used
    if (
        selected_example
        and selected_example != st.session_state["last_selected_example"]
    ):
        portfolio = example_portfolios[selected_example]
        st.session_state.update(
            selected_tickers=portfolio["tickers"],
            max_share=portfolio["max_share"],
            start_date=portfolio["start_date"],
            end_date=portfolio["end_date"],
            stocks_selection=[
                ticker for ticker in portfolio["tickers"] if ticker in stocks
            ],
            indexes_selection=[
                ticker for ticker in portfolio["tickers"] if ticker in index
            ],
            minerals_selection=[
                ticker for ticker in portfolio["tickers"] if ticker in minerals
            ],
            etf_selection=[ticker for ticker in portfolio["tickers"] if ticker in etf],
            crypto_selection=[
                ticker for ticker in portfolio["tickers"] if ticker in crypto
            ],
            bonds_selection=[
                ticker for ticker in portfolio["tickers"] if ticker in bond
            ],
        )
    st.session_state["last_selected_example"] = selected_example

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Randomize Selection"):
            randomize_selection(stocks, index, minerals, etf, crypto, bond)

    with col2:
        if st.button("Reset All"):
            reset_session_state()

    # --- User Interface ---
    st.markdown("**Choose your tickers from the categories below:**")

    # Input for initial capital
    initial_capital = st.number_input(
        label="Enter your initial capital:", min_value=0.0, value=10000.0, step=100.0
    )

    # Input for max share, using session state value
    max_share = st.number_input(
        label="Enter max share of an asset:",
        min_value=0.1,
        value=st.session_state["max_share"],
        step=0.05,
        max_value=1.0,
    )

    min_share = st.number_input(
        label="Enter min share of an asset:",
        min_value=0.0,
        value=st.session_state["min_share"],
        step=0.05,
        max_value=max_share,
    )

    rebalance_freq = st.number_input(
        label="Specify rebalance frequency (in days): Set to 0 for a static portfolio",
        min_value=0,
        value=st.session_state["rebalance_freq"],
        step=1,
    )

    # Date inputs, using session state values
    start_date = st.date_input("Start Date:", format="DD/MM/YYYY", key="start_date")
    end_date = st.date_input("End Date:", format="DD/MM/YYYY", key="end_date")

    # Multiple selection for each category
    stock_tickers = st.multiselect(
        label="Stocks",
        options=stocks,
        key="stocks_selection",
    )
    index_tickers = st.multiselect(
        label="Indexes",
        options=index,
        key="indexes_selection",
    )
    mineral_tickers = st.multiselect(
        label="Minerals",
        options=minerals,
        key="minerals_selection",
    )
    etf_tickers = st.multiselect(
        label="ETFs",
        options=etf,
        key="etf_selection",
    )
    crypto_tickers = st.multiselect(
        label="Cryptocurrency",
        options=crypto,
        key="crypto_selection",
    )
    bond_tickers = st.multiselect(
        label="Bonds",
        options=bond,
        key="bonds_selection",
    )

    # Merge all selected tickers
    selected_tickers = (
        stock_tickers
        + index_tickers
        + mineral_tickers
        + etf_tickers
        + crypto_tickers
        + bond_tickers
    )
    st.session_state["selected_tickers"] = list(set(selected_tickers))

    # Button to validate the selection
    if st.button("Validate Selection"):
        # Check if the user has made a selection
        if st.session_state["selected_tickers"]:
            st.success(f"You have selected: {', '.join(selected_tickers)}")
            work_df = prices[selected_tickers]
            returns = utils.get_returns(work_df)
            ql_start_date = ql.Date(start_date.day, start_date.month, start_date.year)
            ql_end_date = ql.Date(end_date.day, end_date.month, end_date.year)

            if (
                pd.Timestamp(ql_start_date.to_date()).tz_localize("UTC")
                not in prices.index
            ):
                ql_start_date = utils.get_latest_available_date(ql_start_date, prices)
                st.warning(
                    f"The selected start date is not a business day or data is missing. The nearest date is used. {ql_start_date=}"
                )

            if (
                pd.Timestamp(ql_end_date.to_date()).tz_localize("UTC")
                not in prices.index
            ):
                ql_end_date = utils.get_latest_available_date(ql_end_date, prices)
                st.warning(
                    f"The selected end date is not a business day or data is missing. The nearest date is used. {ql_end_date=}"
                )

            result = utils.get_portfolio(
                returns, risk_free_rate, max_share, ql_start_date, ql_end_date
            )
            optimal_weights = result.x
            optimal_sharpe_ratio = -result.fun * np.sqrt(252)
            sorted_assets, sorted_weights = utils.filter_portfolio(
                optimal_weights, selected_tickers, threshold=min_share
            )
            portfolio_sharpe_ratio = utils.sharpe_ratio_portfolio(
                ql_end_date, prices, sorted_assets, sorted_weights, risk_free_rate
            ) * np.sqrt(252)
            portfolio_values = utils.track_portfolio_value(
                ql_end_date,
                prices,
                sorted_assets,
                sorted_weights,
                initial_capital=initial_capital,
            )

            portfolio_rebalanced_values, allocation_history, sharpe_ratios = (
                utils.rebalance_portfolio(
                    prices=prices,
                    selected_tickers=sorted_assets,
                    initial_capital=initial_capital,
                    rebalance_freq=rebalance_freq,
                    risk_free_rate=risk_free_rate,
                    max_share=max_share,
                    start_date=ql_start_date,
                    end_date=ql_end_date,
                )
            )
            portfolio_values = portfolio_rebalanced_values
            summary_df = utils.get_stock_performance_table(
                prices, sorted_assets, sorted_weights, ql_end_date
            )
            # Display the summary table
            st.write("### Stock Performance Summary")
            st.dataframe(summary_df)

            metrics = utils.metrics(portfolio_values)
            portfolio_values_filtered = portfolio_values.dropna()
            portfolio_returns = np.log(
                portfolio_values_filtered / portfolio_values_filtered.shift(1)
            ).dropna()
            sp500_returns_filtered = sp500_returns.dropna()
            common_dates = portfolio_returns.index.intersection(
                sp500_returns_filtered.index
            )
            alpha, beta, summary = utils.calculate_capm_metrics(
                portfolio_returns.loc[common_dates],
                sp500_returns_filtered.loc[common_dates],
                risk_free_rate,
            )

            # Line plot of portfolio value over time with continuous dates
            portfolio_values = portfolio_values.reindex(
                pd.date_range(
                    start=portfolio_values.index.min(),
                    end=portfolio_values.index.max(),
                    freq="D",
                ),
                method="pad",
            )
            # Save portfolio values to CSV
            portfolio_csv = portfolio_values.to_csv(index=True)

            # Add a download button
            st.download_button(
                label="Download Portfolio Data as CSV",
                data=portfolio_csv,
                file_name="portfolio_values.csv",
                mime="text/csv",
            )
            # Bar plot of asset allocation
            allocation_fig = go.Figure()
            allocation_fig.add_trace(
                go.Bar(
                    x=sorted_assets,
                    y=sorted_weights,
                    name="Asset Allocation",
                )
            )
            allocation_fig.update_layout(
                title="Optimal Portfolio Allocation",
                xaxis_title="Assets",
                yaxis_title="Weights",
                template="plotly_white",
            )
            st.plotly_chart(allocation_fig, use_container_width=True)

            # Line plot of portfolio value over time
            value_fig = go.Figure()
            value_fig.add_trace(
                go.Scatter(
                    x=portfolio_values.index,
                    y=portfolio_values.values,
                    mode="lines",
                    name="Portfolio Value",
                )
            )
            value_fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                template="plotly_white",
                hovermode="x unified",
                yaxis=dict(tickformat=",.2f"),
            )
            st.plotly_chart(value_fig, use_container_width=True)

            # Display portfolio metrics
            st.markdown("### Portfolio Metrics")
            st.write(
                f"**Sharpe Ratio (Optimal ratio at purchase):** {optimal_sharpe_ratio:.3f}"
            )
            st.write(f"**Sharpe Ratio (Current):** {portfolio_sharpe_ratio:.3f}")
            st.write(f"**All-Time High (ATH):** ${metrics['ATH']:.2f}")
            st.write(f"**All-Time Low (ATL):** ${metrics['ATL']:.2f}")
            st.write(f"**Max Drawdown:** {metrics['Max Drawdown']:.2%}")
            st.write(f"**Total Return:** {metrics['Total Return']:.2f}%")
            st.write(f"**Annualized Return:** {metrics['Annualized Return']:.2f}%")
            # Calcul des métriques CAPM

            # Display the results
            st.write("### CAPM Model Results")
            st.write(f"- **Alpha (Risk-adjusted performance)**: {alpha:.4f}")
            st.write(f"- **Beta (Market sensitivity)**: {beta:.4f}")
            st.write("#### Implications:")

            # Beta interpretation
            if beta > 1:
                st.write(
                    f"The portfolio is more volatile than the market (beta = {beta:.2f}). "
                    f"It is therefore considered riskier compared to the market."
                )
            elif beta == 1:
                st.write(
                    "The portfolio has the same volatility as the market (Beta = 1)."
                )
            else:
                st.write(
                    f"The portfolio is less volatile than the market (beta = {beta:.2f}). "
                    f"It is therefore considered defensive."
                )

            # Alpha interpretation
            if alpha > 0:
                st.write(
                    f"The portfolio outperforms the market after adjusting for risk "
                    f"(Alpha = {alpha:.4f}). This may indicate effective active management."
                )
            elif alpha == 0:
                st.write(
                    "The portfolio is perfectly aligned with the expected performance according to the CAPM (Alpha = 0)."
                )
            else:
                st.write(
                    f"The portfolio underperforms the market after adjusting for risk "
                    f"(Alpha = {alpha:.4f}). This may indicate ineffective management."
                )

            # Display the model summary
            with st.expander("Show Model Summary"):
                st.text(summary)

        else:
            st.warning("Please select at least one ticker before validating.")

    st.header("Notes on Portfolio Optimization")
    st.write(
        """
    **Working Assumptions:**
    - We assume no slippage, no taxes, no transaction fees, no liquidity constraints, no short selling, and no leverage.
    - These assumptions simplify the modeling process for a theoretical approach and does not fully reflect real-world scenarios.
    """
    )
    st.write(
        """
    **Rebalancing Approach:**
    - During rebalancing, we assume that all assets are completely sold and then repurchased based on the new weights.
    - This is a simplification, as real-world portfolio adjustments would incur transaction fees and be influenced by liquidity constraints.
    """
    )
    st.write(
        """
**Dynamic Portfolio Observations:**
- In dynamic portfolios, asset weights often exhibit abrupt shifts between zero and the maximum cap.
- This phenomenon arises due to the high sensitivity of the optimization process.
- Notably, removing just a single data point from a large dataset can result in significant changes, with some assets losing nearly all their allocation.
- A potential improvement could involve using a Kalman filter to update the weights more gradually, providing smoother transitions and reducing abrupt changes in allocations.
"""
    )


# Launch the page
if __name__ == "__main__":
    page_dynamicPortfolio()
