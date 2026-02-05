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
COPY schema.sql .
COPY static/ static/
COPY templates/ templates/

# Copy database (must be present at build time or mounted as volume)
COPY telegram.db .

# Environment
ENV PORT=8080
ENV HOST=0.0.0.0
ENV DB_PATH=telegram.db

EXPOSE 8080

CMD ["gunicorn", "dashboard:app", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120"]
