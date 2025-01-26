import streamlit as st
from fastapi import FastAPI
from fastapi.responses import FileResponse
from uvicorn import Config, Server
import threading

st.set_page_config(page_title="Financial Data Experimentation Hub", layout="wide")

app = FastAPI()


@app.get("/robots.txt")
async def robots_txt():
    return FileResponse("robots.txt")


def run_fastapi():
    config = Config(app=app, host="0.0.0.0", port=8000, log_level="info")
    server = Server(config)
    server.run()


thread = threading.Thread(target=run_fastapi, daemon=True)
thread.start()

from about import page_about
from home_page import page_home
from impliedVolatility import page_IV
from pairtrading import page_pairtrading
from dynamicPortfolio import page_dynamicPortfolio
from architecture import page_architecture

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
