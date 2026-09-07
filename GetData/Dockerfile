FROM python:3.11-slim

WORKDIR /app

# Dependencies first, so code edits do not invalidate the wheel layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py main.py market_data.py storage.py tickers.csv ./

ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "daily"]
