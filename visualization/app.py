import streamlit as st

st.set_page_config(page_title="Multi-Page Application", layout="wide")
from about import page_about
from portfolio import page_portfolio
from home_page import page_home
from impliedVolatility import page_IV
from pairtrading import page_pairtrading
from dynamicPortfolio import page_dynamicPortfolio


# --- Navigation bar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home Page", "Make a Portfolio", "Implied Volatility", "Pair Trading", "About", "Dynamic Portfolio"],
)

# --- Display the corresponding page ---
if page == "Home Page":
    page_home()
elif page == "Make a Portfolio":
    page_portfolio()

elif page == "Implied Volatility":
    page_IV()

elif page == "About":
    page_about()

elif page == "Pair Trading":
    page_pairtrading()
    
elif page == "Dynamic Portfolio":
    page_dynamicPortfolio()
