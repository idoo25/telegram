FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY dashboard.py .
COPY ai_search.py .
COPY algorithms.py .
COPY data_structures.py .
COPY indexer.py .
COPY search.py .
COPY semantic_search.py .
COPY hybrid_search.py .
COPY gemini_client.py .
COPY stylometry.py .
COPY schema.sql .
COPY static/ static/
COPY templates/ templates/

# DB is downloaded from HF Dataset repo on startup (see ensure_db_exists in dashboard.py)

# HF Spaces uses port 7860
ENV PORT=7860
ENV HOST=0.0.0.0
ENV DB_PATH=telegram.db

EXPOSE 7860

CMD ["gunicorn", "dashboard:app", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120"]
