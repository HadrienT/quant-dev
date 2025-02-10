# Quantitative Finance Project

This repository is a personal project aimed at exploring the field of quantitative finance and demonstrating my development skills with the goal of securing a position as a quantitative developer. The project encompasses automated data ingestion, cloud infrastructure management, and interactive financial visualizations.

## Repository Structure

- **Cloud**
  Contains all the Google Cloud Platform (GCP) infrastructure code. This folder sets up and manages the cloud resources required for the project.

- **GetData**
  A subproject dedicated to automating the daily retrieval of market data from [Yfinance](https://pypi.org/project/yfinance/) into Google BigQuery. This ensures that the data is up-to-date for analysis and visualization.

- **Visualization**
  The core of the project where the financial analysis happens. This folder includes a [Streamlit](https://streamlit.io/) web application that can be accessed at [https://tramonihadrien-portfolio.com](https://tramonihadrien-portfolio.com). Key features include:
  - **Portfolio Construction**: Create portfolios with various asset classes including stocks, bonds, ETFs, and cryptocurrencies.
  - **Static & Dynamic Portfolios**: Build static portfolios or dynamically manage capital allocation by optimizing the Sharpe ratio.
  - **Implied Volatility Visualization**: Visualize implied volatility curves or surfaces.
  - **Pair Trading Module**: Currently under development, this module only includes asset pairs that have been pre-validated as cointegrated using historical data from 2000 to the present, and I am actively developing the buy-sell strategy.

## Cloud Architecture

Below is the cloud architecture diagram for this project:

![Cloud Architecture](visualization/assets/quant-dev%203.svg)

## Getting Started

### Prerequisites

- Python 3.11 or later
- [Streamlit](https://streamlit.io/)
- Google Cloud SDK (for managing GCP resources)
- Other dependencies as listed in the respective `requirements.txt` files.

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/quant-dev.git
cd quant-dev
```

## Running the Streamlit Application

To run the visualization web app:

Navigate to the visualization directory:

```bash
cd visualization
```

Start the Streamlit app:

```bash
python3 server.py
```

Open your browser and go to https://tramonihadrien-portfolio.com (or the local URL provided by Streamlit).

### Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with improvements, bug fixes, or new features.

### License

This project is licensed under the MIT License. See the LICENSE file for details.

### Contact

For any questions or feedback, please contact me at tramonihadrien@gmail.com

This project is continuously evolving as I delve deeper into quantitative finance and develop more sophisticated trading and visualization tools.
