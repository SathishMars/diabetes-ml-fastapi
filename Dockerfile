FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app including model
COPY app/ ./app/

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.fastapi_diabetes_app:app", "--host", "0.0.0.0", "--port", "8000"]
