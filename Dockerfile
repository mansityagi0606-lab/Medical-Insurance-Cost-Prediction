FROM python:3.10-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . /app

# 🔥 THIS is what fixes your error
ENV PYTHONPATH="/app/src"

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
