FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/.streamlit

COPY ./.streamlit/config.toml /root/.streamlit/config.toml

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
