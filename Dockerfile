# api container for src/api/main.py
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# system deps for psycopg2 & pptx/pdf parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . /app

# default port
EXPOSE 8000

# health and uvicorn
CMD ["python","-m","uvicorn","src.api.main:app","--host","0.0.0.0","--port","8000"]
