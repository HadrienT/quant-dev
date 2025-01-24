from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import QuantLib as ql
import numpy as np
from scipy.interpolate import griddata
import utils

# Get the risk-free rate
risk_free_rate = utils.get_risk_free_rate()


def page_IV():
    st.title("Implied Volatility")

    # List of tickers
    top_stocks = sorted(
        ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BRK-B", "NVDA", "META"]
    )
    all_stocks = utils.get_stock_tickers()
    stocks = top_stocks + [stock for stock in all_stocks if stock not in top_stocks]

    # User interface
    st.markdown("**Choose your tickers and expiration date below:**")
    option_type = st.selectbox(label="Type", options=["Call", "Put"], index=0)
    stock_ticker = st.selectbox(label="Stock", options=stocks, index=0)
    display_type = st.radio(
        "Choose the display type", options=["Curve", "Surface"], index=0
    )

    # Initialize necessary keys
    st.session_state.setdefault("stock", None)
    st.session_state.setdefault("expirations", [])
    st.session_state.setdefault("option_chain", None)
    st.session_state.setdefault("selected_expiration", None)

    # Validate the stock to display expiration dates
    if st.button("Validate stock"):
        stock = yf.Ticker(stock_ticker)

        # Check the availability of options
        expirations = stock.options
        if not expirations:
            st.error(f"No options data available for {stock_ticker}.")
            return

        # Save expirations in st.session_state
        st.session_state["stock"] = stock
        st.session_state["expirations"] = expirations
        (
            st.success("Stock validated. Choose an expiration date.")
            if display_type == "Curve"
            else st.success("Stock validated. Wait for the surface to load.")
        )

    # Display the curve or surface based on the choice
    if display_type == "Curve":
        # If expirations are available, display a dropdown list
        if st.session_state["expirations"]:
            expiration_date = st.selectbox(
                "Select an expiration date",
                st.session_state["expirations"],
            )

            # Load options data for a selected date
            if st.button("Load Options Data"):
                stock = st.session_state["stock"]
                option_chain = stock.option_chain(expiration_date)
                st.session_state["option_chain"] = option_chain
                st.session_state["selected_expiration"] = expiration_date
                st.success(f"Options data loaded for {expiration_date}.")

        # Display the curve for a specific date
        if st.session_state["option_chain"]:
            expiration_date = st.session_state["selected_expiration"]
            option_chain = st.session_state["option_chain"]
            calls = option_chain.calls
            puts = option_chain.puts

            spot_data = st.session_state["stock"].history(period="1d")
            if spot_data.empty:
                st.error(f"No historical data available for {stock_ticker}.")
                return

            spot_price = spot_data["Close"].iloc[-1]
            work_option = calls if option_type == "Call" else puts
            ql_option_type = ql.Option.Call if option_type == "Call" else ql.Option.Put
            strikes = work_option["strike"]
            option_prices = work_option["lastPrice"]

            volatilities = []
            for i, strike in enumerate(strikes):
                try:
                    implied_vol = utils.calculate_implied_volatility(
                        ql_option_type,
                        strike,
                        option_prices[i],
                        spot_price,
                        risk_free_rate,
                        0.0,
                        (
                            ql.DateParser.parseFormatted(expiration_date, "%Y-%m-%d")
                            - ql.Date.todaysDate()
                        )
                        / 365.0,
                    )
                    volatilities.append(implied_vol)
                except Exception as e:
                    print(e)
                    volatilities.append(np.nan)

            valid_strikes = [
                s for s, v in zip(strikes, volatilities) if not np.isnan(v)
            ]
            valid_volatilities = [v for v in volatilities if not np.isnan(v)]

            if valid_strikes:
                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=valid_strikes,
                            y=valid_volatilities,
                            mode="lines",
                            name="IV Curve",
                        )
                    ]
                )

                # Add a vertical line for the spot price with a cursor
                fig.add_shape(
                    type="line",
                    x0=spot_price,
                    y0=0,
                    x1=spot_price,
                    y1=max(valid_volatilities),
                    line=dict(color="red", width=2, dash="dash"),
                    name="Spot Price",
                )

                # Add an annotation to display a cursor above the line
                fig.add_trace(
                    go.Scatter(
                        x=[spot_price],
                        y=[
                            max(valid_volatilities) / 2
                        ],  # Position of the text on the Y-axis
                        mode="text",
                        text=["Spot Price"],
                        textposition="top right",
                        hoverinfo="text",
                        showlegend=False,
                    )
                )

                # Update the layout
                fig.update_layout(
                    title=f"Implied Volatility Curve for {expiration_date}",
                    xaxis_title="Strike",
                    yaxis_title="Implied Volatility",
                    yaxis=dict(range=[0, None]),
                )
                st.plotly_chart(fig)
            else:
                st.error("No valid data for this expiration date.")

    elif display_type == "Surface":
        # Build a surface for all expirations
        stock = st.session_state.get("stock")
        expirations = st.session_state.get("expirations", [])
        if stock and expirations:
            spot_price = stock.history(period="1d")["Close"].iloc[-1]
            vol_data = []

            for expiration_date in expirations:
                option_chain = stock.option_chain(expiration_date)
                calls = option_chain.calls
                puts = option_chain.puts
                work_option = calls if option_type == "Call" else puts
                strikes = work_option["strike"]
                option_prices = work_option["lastPrice"]
                expiry = (
                    ql.DateParser.parseFormatted(expiration_date, "%Y-%m-%d")
                    - ql.Date.todaysDate()
                ) / 365.0

                vol_row = []
                for i, strike in enumerate(strikes):
                    try:
                        implied_vol = utils.calculate_implied_volatility(
                            ql.Option.Call,
                            strike,
                            option_prices[i],
                            spot_price,
                            risk_free_rate,
                            0.0,
                            expiry,
                        )
                        vol_row.append(implied_vol)
                    except Exception:
                        vol_row.append(np.nan)
                vol_data.append((strikes, vol_row, expiration_date))

            # Create a 3D surface
            fig = go.Figure()
            fig.update_layout(
                width=1000,  # Chart width
                height=800,  # Chart height
            )

            if vol_data:  # Check that vol_data is not empty
                all_strikes = np.concatenate([strikes for strikes, _, _ in vol_data])
                all_expirations = np.concatenate(
                    [
                        [
                            (
                                datetime.strptime(exp_date, "%Y-%m-%d") - datetime.now()
                            ).days
                            / 365.0
                        ]
                        * len(strikes)
                        for strikes, _, exp_date in vol_data
                    ]
                )
                all_vols = np.concatenate([vols for _, vols, _ in vol_data])

                # Remove points with NaN values
                mask = ~np.isnan(all_vols)
                all_strikes = all_strikes[mask]
                all_expirations = all_expirations[mask]
                all_vols = all_vols[mask]

                # Create a regular grid for interpolation
                grid_x, grid_y = np.meshgrid(
                    np.linspace(min(all_strikes), max(all_strikes), 50),
                    np.linspace(min(all_expirations), max(all_expirations), 50),
                )

                # Interpolate the data
                grid_z = griddata(
                    (all_strikes, all_expirations),
                    all_vols,
                    (grid_x, grid_y),
                    method="linear",
                )

                fig.add_trace(
                    go.Surface(
                        x=grid_x,
                        y=grid_y,
                        z=grid_z,
                        colorscale="Reds",
                        showscale=True,
                    )
                )

            fig.update_layout(
                title="Implied Volatility Surface",
                scene=dict(
                    xaxis_title="Strike",
                    yaxis_title="Expiration",
                    zaxis_title="Volatility",
                ),
            )
            st.plotly_chart(fig)


# Run the application
if __name__ == "__main__":
    page_IV()
