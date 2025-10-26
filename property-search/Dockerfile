FROM python:3.11-slim

WORKDIR /app

# Install system dependencies


# Copy and install dependencies safely
COPY requirements.txt .

# ✅ Fix: upgrade pip + set timeout + optional mirror
RUN python -m pip install --upgrade pip \
    && pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

EXPOSE 8010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
