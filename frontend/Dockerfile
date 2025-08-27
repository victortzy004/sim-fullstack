FROM python:3.11-slim

WORKDIR /app

# optional but helpful for wheels
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV API_BASE_URL=http://api:8000
EXPOSE 8505

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8505", "--server.address=0.0.0.0"]