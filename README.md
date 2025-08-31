# Social Media Content Analyzer

An app that uploads PDFs/images, extracts text (PDF parse + OCR with Tesseract), and analyzes social posts to suggest engagement improvements.

## Features
- Upload PDFs and images (PNG/JPG/JPEG/WEBP)
- PDF parsing via PyMuPDF (preserves basic layout line breaks)
- OCR for images via Tesseract (pytesseract) with light preprocessing
- Per-post analysis (split by blank lines or markers `---` / `###`)
  - Sentiment (VADER)
  - Word/char counts, hashtag/mention/url/emoji counts, question count
  - Heuristic suggestions (CTA, hashtags, tone, length, etc.)
- Download extracted text and CSV

## Local Setup
1. Install Python 3.10+
2. Install Tesseract OCR:
   - **Windows**: https://github.com/UB-Mannheim/tesseract/wiki
     - If not in PATH, set env var `TESSERACT_PATH` to `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`
   - **macOS (Homebrew)**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`
3. Create & activate a virtual environment (recommended)
4. Install deps: `pip install -r requirements.txt`
5. Run: `streamlit run app.py`

## Usage
- Upload files in the sidebar, click **Process Files**
- Review the extracted text, optionally edit
- Click **Run Analysis** to get metrics + suggestions
- Download **extracted_text.txt** and/or **analysis.csv**

## Deployment (Free-friendly Options)

### Option A: Docker (works on Render, Railway, etc.)
Create a repo with this project, then deploy the container to a PaaS.

1. Build locally
   ```bash
   docker build -t smca .
   docker run -p 8501:8501 smca
   ```
2. Push to your registry and deploy on Render/Railway/Your VM.
   - Expose port `8501`
   - Set root command to:
     ```
     streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
     ```

### Option B: VM (Ubuntu)
```bash
sudo apt-get update && sudo apt-get install -y python3-venv tesseract-ocr
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

> **Note**: Streamlit Cloud does not provide apt packages, so installing `tesseract-ocr` there is non-trivial. Prefer Docker/VM PaaS for this app.

## Tech Choices
- **Streamlit** for fast, clean UI and hosting ease.
- **PyMuPDF** for robust, layout-aware PDF text extraction.
- **pytesseract + Tesseract** for local OCR (free, good quality).
- **VADER** for lightweight sentiment analysis (social text friendly).

## Deliverables Check
- Working application URL: deploy via Docker to Render/Railway and share the URL.
- GitHub repo: include these files and a short description.
- Brief write-up (≤200 words): see below.

## Brief Approach (≤200 words)
We prioritized a simple, reliable pipeline. PDFs are parsed with PyMuPDF, which preserves readable line breaks; images go through a light preprocessing step (grayscale, autocontrast, sharpen) before Tesseract OCR via `pytesseract`. The UI is built with Streamlit to minimize boilerplate and provide fast iteration; it includes spinners, progress indicators, and clear error messages (e.g., when Tesseract isn’t found). Posts are segmented using blank lines or simple markers (`---`, `###`). For analysis, VADER provides a sentiment score that works well on social text. Heuristics add actionable suggestions (CTA presence, hashtag counts, link context, emojis, length, and a hook). Outputs are downloadable (raw text and CSV). The result is production-quality, dependency-light code that’s easy to run locally and deploy via Docker to free-tier platforms.

## License
MIT

--------------------------------
=== FILE: Dockerfile ===
--------------------------------
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

# Streamlit uses ~/.streamlit; create it to avoid warnings (optional)
RUN mkdir -p /root/.streamlit && \
    bash -lc 'cat > /root/.streamlit/config.toml <<EOF\n\
[server]\n\
headless = true\n\
enableCORS = true\n\
enableXsrfProtection = true\n\
address = "0.0.0.0"\n\
port = 8501\n\
EOF'

CMD ["bash", "-lc", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]

