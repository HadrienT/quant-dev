import streamlit as st


def page_home():
    st.title("📊 Welcome to the Financial Data Experimentation Hub")
    st.markdown(
        """
        This platform allows you to explore and experiment with **financial data** through a variety of tools and visualizations.
        Whether you're optimizing portfolios, analyzing volatility, or backtesting strategies, you'll find something interesting here!
        """
    )
    st.header("🛠️ Available Tools")
    st.markdown(
        """
        Below is an overview of the tools currently available on this platform:

        1. **💼 Make a Portfolio**
           - Create an optimized portfolio by selecting a bundle of assets.
           - Automatically compute the **best allocation of capital** based on **Sharpe ratio optimization** between your selected dates.
           - Choose between a **static portfolio** or set up a **dynamic rebalancing** strategy that adjusts based on Sharpe ratio over time.

        2. **📈 Implied Volatility**
           - Visualize the **implied volatility smile** of a stock.
           - Dive deeper with a **volatility surface** visualization to better understand market expectations.

        3. **🔗 Pair Trading**
           - Backtest pair trading strategies on **preselected pairs of cointegrated stocks** (They were determined using data from 2000 to now).
           - **Development in progress**

        4. **🚀 Coming Soon: Options in Portfolio Optimization**
           - Explore how to optimize a portfolio with **options**
        """
    )

    st.header("✨ How to Get Started")
    st.markdown(
        """
        - Navigate through the sidebar to explore the tools available.
        - Input your data or choose from preloaded configurations to get started quickly.
        - Experiment with the tools and see how financial analysis can inform decision-making.
        """
    )

    st.info(
        """
    **Pro Tip**: The risk-free rate used in every model is retrieved from the Federal Reserve API using data from the 10-year U.S. Treasury bond.
    """
    )

    st.markdown("---")
    st.subheader("📬 Feedback & Suggestions")
    st.markdown(
        """
        If you have any feedback, feature requests, or just want to say hi, feel free to reach out!
        You can find my information in the **About** tab.
        """
    )
