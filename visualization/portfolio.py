import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import utils
import QuantLib as ql
import numpy as np
import datetime


def page_portfolio():
    st.text(
        "Make a portfolio on the fly using the selected tickers from various asset classes. The date range is to select the period for the optimization. We consider a one time investment at the end of this period."
    )
    _, prices = utils.load_prices()
    risk_free_rate = utils.get_risk_free_rate()
    sp500_returns = utils.calculate_sp500_returns(prices)
    st.title("Make Your Portfolio")

    # --- List of tickers ---
    stocks = sorted(["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BRK-B", "NVDA", "META"])
    index = sorted(
        ["^GSPC", "^DJI", "^IXIC", "^FTSE", "^GDAXI", "^FCHI", "^N225", "URTH"]
    )
    minerals = sorted(["GC=F", "SI=F", "CL=F", "BZ=F", "HG=F"])
    etf = sorted(["SPY", "QQQ", "VTI", "EEM", "IWM", "GLD", "TLT"])
    crypto = sorted(["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD"])
    bond = sorted(["^TYX"])

    # --- User Interface ---
    st.markdown("**Choose your tickers from the categories below:**")
    # Input for initial capital
    initial_capital = st.number_input(
        label="Enter your initial capital:", min_value=0.0, value=10000.0, step=100.0
    )
    max_share = st.number_input(
        label="Enter max share of an asset:",
        min_value=0.1,
        value=0.5,
        step=0.05,
        max_value=1.0,
    )
    start_date = st.date_input("Start Date:", min_value=datetime.date(2021, 1, 1))
    end_date = st.date_input("End Date:", min_value=start_date)
    # Multiple selection for each category
    stock_tickers = st.multiselect(label="Stocks", options=stocks, default=None)
    index_tickers = st.multiselect(label="Indexes", options=index, default=None)
    mineral_tickers = st.multiselect(label="Minerals", options=minerals, default=None)
    etf_tickers = st.multiselect(label="ETFs", options=etf, default=None)
    crypto_tickers = st.multiselect(
        label="Cryptocurrency", options=crypto, default=None
    )
    bond_tickers = st.multiselect(label="Bonds", options=bond, default=None)

    # Merge all selected tickers
    selected_tickers = (
        stock_tickers
        + index_tickers
        + mineral_tickers
        + etf_tickers
        + crypto_tickers
        + bond_tickers
    )

    # Button to validate the selection
    if st.button("Validate Selection"):
        # Check if the user has made a selection
        if selected_tickers:
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
                optimal_weights, selected_tickers, threshold=0.005
            )
            portfolio_values = utils.track_portfolio_value(
                ql_end_date,
                prices,
                sorted_assets,
                sorted_weights,
                initial_capital=initial_capital,
            )
            print(f"{portfolio_values=}")
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
            portfolio_returns.to_csv("portfolio_returns.csv")
            sp500_returns_filtered.to_csv("sp500_returns.csv")
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
            st.write(f"**Sharpe Ratio:** {optimal_sharpe_ratio:.3f}")
            st.write(f"**All-Time High (ATH):** ${metrics['ATH']:.2f}")
            st.write(f"**All-Time Low (ATL):** ${metrics['ATL']:.2f}")
            st.write(f"**Max Drawdown:** {metrics['Max Drawdown']:.2%}")
            st.write(f"**Total Return:** {metrics['Total Return']:.2f}%")
            st.write(f"**Annualized Return:** {metrics['Annualized Return']:.2f}%")
            # Calcul des métriques CAPM

            # Display the results
            st.write(f"### CAPM Model Results")
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


# Launch the page
if __name__ == "__main__":
    page_portfolio()
