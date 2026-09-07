FROM python:3.11-slim

WORKDIR /app

# Dependencies first, so code edits do not invalidate the wheel layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

ENV STREAMLIT_CONFIG_FILE=/app/.streamlit/config.toml
EXPOSE 8080

CMD ["python3", "server.py"]
