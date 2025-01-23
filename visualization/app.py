import streamlit as st

st.set_page_config(page_title="Multi-Page Application", layout="wide")
from about import page_about
from home_page import page_home
from impliedVolatility import page_IV
from pairtrading import page_pairtrading
from dynamicPortfolio import page_dynamicPortfolio

# --- Navigation bar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home Page", "Make a Portfolio", "Implied Volatility", "Pair Trading", "About"],
)

# --- Display the corresponding page ---
if page == "Home Page":
    page_home()
elif page == "Make a Portfolio":
    page_dynamicPortfolio()

elif page == "Implied Volatility":
    page_IV()

elif page == "About":
    page_about()

elif page == "Pair Trading":
    page_pairtrading()