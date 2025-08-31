# Multi-stage to keep image smaller
FROM python:3.11-slim AS base

# Install system deps incl. tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy files
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

# Default Streamlit configuration for container
EXPOSE 8501
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Streamlit uses ~/.streamlit; create it to avoid warnings
RUN mkdir -p /root/.streamlit
RUN printf "[server]\nheadless = true\nenableCORS = true\nenableXsrfProtection = true\naddress = \"0.0.0.0\"\nport = 8501\n" > /root/.streamlit/config.toml

# Command to run the app
CMD ["bash", "-lc", "streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
