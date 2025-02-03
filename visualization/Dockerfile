FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/.streamlit && \
    cp ./.streamlit/config.toml /app/.streamlit/config.toml

ENV STREAMLIT_CONFIG_FILE=/app/.streamlit/config.toml
EXPOSE 8080

# CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
CMD ["python3", "server.py"]
