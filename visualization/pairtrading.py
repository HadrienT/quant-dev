import streamlit as st
import utils
import plotly.graph_objects as go
import statsmodels.api as sm


def page_pairtrading():
    special_filters = [
        "APD",
        "SHW",
        "AMZN",
        "HD",
        "COST",
        "WMT",
        "KO",
        "PEP",
        "MRK",
        "UNH",
        "HON",
        "UNP",
        "AAPL",
        "MSFT",
    ]
    _, prices = utils.load_prices(special_filters)

    sectors = [
        "Basic Materials",
        "Consumer Cyclical",
        "Consumer Defensive",
        "Energy",
        "Financial Services",
        "Healthcare",
        "Industrials",
        "Real Estate",
        "Technology",
        "Utilities",
    ]

    known_pairs = {
        "Basic Materials": ["APD-SHW"],
        "Consumer Cyclical": ["AMZN-HD"],
        "Consumer Defensive": ["COST-WMT", "KO-PEP"],
        "Healthcare": ["MRK-UNH"],
        "Industrials": ["HON-UNP"],
        "Technology": ["AAPL-MSFT"],
    }
    sector_selector = st.selectbox(label="Sector", options=sectors, index=0)
    if sector_selector:
        selected_pairs = known_pairs.get(sector_selector, [])
        if selected_pairs:
            pair_selector = st.selectbox(label="Pair", options=selected_pairs, index=0)
            if pair_selector:
                # Calculate cointegration
                results = utils.calculate_coint(pair_selector.split("-"), prices)
                st.write(
                    f"For pair {pair_selector}, cointegration score is {results['score']:.2f} "
                    f"and p-value is {results['p_value']:.2f}"
                )

                # Plot stock prices
                stock_a, stock_b = pair_selector.split("-")
                st.write(f"### Stock Prices for {stock_a} and {stock_b}")

                # Create the plot using Plotly
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=prices.index,
                        y=prices[stock_a],
                        mode="lines",
                        name=stock_a,
                        line=dict(width=2),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=prices.index,
                        y=prices[stock_b],
                        mode="lines",
                        name=stock_b,
                        line=dict(width=2),
                    )
                )

                # Customize the layout
                fig.update_layout(
                    title=f"Historical Prices for {stock_a} and {stock_b}",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1),
                    template="plotly_white",
                    hovermode="x unified",
                )

                # Display the plot
                st.plotly_chart(fig, use_container_width=True)

                price_a = prices[stock_a]
                price_b = prices[stock_b]
                X = sm.add_constant(price_b)
                model = sm.OLS(price_a, X).fit()
                hedge_ratio = model.params[stock_b]
                spread = price_a - hedge_ratio * price_b
                st.write(f"### Testing spread between {stock_a} and {stock_b}")
                utils.stationarity_test(spread)
                spread_mean = spread.mean()
                spread_std = spread.std()
                z_score = (spread - spread_mean) / spread_std

                threshold = 2
                st.write("### Trading Strategy")
                st.write(
                    f"Buy {stock_a} and sell {stock_b} when z-score is below -{threshold}"
                )
                st.write(
                    f"Sell {stock_a} and buy {stock_b} when z-score is above {threshold}"
                )
                st.write("### Spread Z-Score")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=prices.index,
                        y=z_score,
                        mode="lines",
                        name="Z-Score",
                        line=dict(width=2),
                    )
                )
                fig.add_hline(y=threshold, line_dash="dash", line_color="red")
                fig.add_hline(y=-threshold, line_dash="dash", line_color="green")
                fig.update_layout(
                    title=f"Spread Z-Score for {stock_a} and {stock_b}",
                    xaxis_title="Date",
                    yaxis_title="Z-Score",
                    legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1),
                    template="plotly_white",
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No known pairs for this sector.")


# Launch the page
if __name__ == "__main__":
    page_pairtrading()
