import streamlit as st

st.set_page_config(page_title="Financial Data Experimentation Hub", layout="wide")
from about import page_about
from home_page import page_home
from impliedVolatility import page_IV
from pairtrading import page_pairtrading
from dynamicPortfolio import page_dynamicPortfolio
from architecture import page_architecture

# --- Navigation bar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Home Page",
        "Make a Portfolio",
        "Implied Volatility",
        "Pair Trading",
        "Architecture",
        "About",
    ],
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

elif page == "Architecture":
    page_architecture()
