# import io
# import os
# import re
# import time
# import base64
# from typing import List, Tuple, Dict


# import streamlit as st
# import fitz  # PyMuPDF
# from PIL import Image, ImageFilter, ImageOps
# import pytesseract
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# # ---------------- OCR & Sentiment Setup ----------------
# # Set Tesseract path (Windows)
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# # Test Tesseract (optional)
# try:
#     version = pytesseract.get_tesseract_version()
#     print("Tesseract is working! Version:", version)
# except Exception as e:
#     print("Error detecting Tesseract:", e)

# # Initialize sentiment analyzer
# analyzer = SentimentIntensityAnalyzer()

# # ----------------------------- 
# # App configuration
# APP_TITLE = "Social Media Content Analyzer"
# APP_DESC = (
#     "Upload PDFs or images to extract text. Then analyze posts and get suggestions "
#     "to improve engagement. Split posts with a blank line, '---', or '###'."
# )

# # ----------------------------- 
# # Utils
# from pdf2image import convert_from_bytes
# from PIL import Image
# import io

# def extract_text_from_pdf(pdf_bytes: bytes) -> str:
#     """
#     Extract text from PDF. If a page has no selectable text, use OCR on that page.
#     Works for both text-based and image-based PDFs.
#     """
#     import fitz  # PyMuPDF
#     text_chunks = []

#     with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
#         for page in doc:
#             # Try to extract text directly
#             text = page.get_text("text").strip()

#             # If no text found, convert page to image and run OCR
#             if not text:
#                 # Convert page to image
#                 pix = page.get_pixmap(dpi=300)  # high resolution for OCR
#                 img_bytes = pix.tobytes("png")
#                 text = extract_text_from_image(img_bytes)  # use your existing OCR function

#             text_chunks.append(text)

#     return "\n".join(text_chunks).strip()




# def preprocess_image(img: Image.Image) -> Image.Image:
#     gray = ImageOps.grayscale(img)
#     gray = ImageOps.autocontrast(gray)
#     gray = gray.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
#     return gray


# def extract_text_from_image(image_bytes: bytes, lang: str = "eng") -> str:
#     img = Image.open(io.BytesIO(image_bytes))
#     img = preprocess_image(img)
#     custom_oem_psm_config = r"--oem 1 --psm 3"
#     try:
#         text = pytesseract.image_to_string(img, lang=lang, config=custom_oem_psm_config)
#     except pytesseract.TesseractNotFoundError:
#         raise RuntimeError(
#             "Tesseract is not installed or not in PATH. "
#             "Install it and/or set TESSERACT_PATH env var."
#         )
#     return text.strip()


# def split_into_posts(text: str) -> List[str]:
#     text = text.replace("\r\n", "\n").replace("\r", "\n")
#     parts = re.split(r"(?:\n\s*\n)|(?:\n?---+\n?)|(?:\n?###\n?)", text)
#     return [p.strip() for p in parts if p and p.strip()]


# # Regex patterns
# EMOJI_PATTERN = re.compile(
#     "[" "\U0001F300-\U0001F5FF"
#     "\U0001F600-\U0001F64F"
#     "\U0001F680-\U0001F6FF"
#     "\U0001F700-\U0001F77F"
#     "\U0001F780-\U0001F7FF"
#     "\U0001F800-\U0001F8FF"
#     "\U0001F900-\U0001F9FF"
#     "\U0001FA00-\U0001FA6F"
#     "\U0001FA70-\U0001FAFF"
#     "\u2600-\u26FF\u2700-\u27BF" "]+")

# URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
# HASHTAG_PATTERN = re.compile(r"(?<!\w)#([A-Za-z0-9_]+)")
# MENTION_PATTERN = re.compile(r"(?<!\w)@([A-Za-z0-9_]+)")


# def analyze_post(text: str, analyzer: SentimentIntensityAnalyzer) -> Dict:
#     words = re.findall(r"\b\w+\b", text)
#     word_count = len(words)
#     char_count = len(text)
#     hashtags = HASHTAG_PATTERN.findall(text)
#     mentions = MENTION_PATTERN.findall(text)
#     urls = URL_PATTERN.findall(text)
#     emojis = EMOJI_PATTERN.findall(text)
#     questions = text.count("?")
#     sentiment = analyzer.polarity_scores(text)["compound"]

#     suggestions = []

#     if word_count < 15:
#         suggestions.append("Add a bit more context—very short posts may underperform.")
#     elif word_count > 220:
#         suggestions.append("Trim the length or front-load key info to keep attention.")

#     if len(hashtags) < 2:
#         suggestions.append("Include 2–5 relevant #hashtags for discovery.")
#     elif len(hashtags) > 8:
#         suggestions.append("Reduce hashtag count; too many can look spammy.")

#     if questions == 0:
#         suggestions.append("Add a question or prompt to spark comments.")
#     if not re.search(r"\b(like|comment|share|save|follow|subscribe|dm|link in bio)\b", text, re.I):
#         suggestions.append("Add a clear call-to-action (e.g., 'Comment your thoughts').")

#     if len(emojis) == 0:
#         suggestions.append("A tasteful emoji can add tone and scannability.")
#     elif len(emojis) > 10:
#         suggestions.append("Use fewer emojis for readability.")

#     if sentiment < -0.4:
#         suggestions.append("Tone feels negative—consider reframing benefits or solutions.")
#     if urls and not re.search(r"\b(why|what|how|learn|check)\b", text, re.I):
#         suggestions.append("Explain why the link is worth clicking.")

#     if not re.search(r"[A-Z].{0,40}:", text):
#         suggestions.append("Consider a short hook/heading at the start to grab attention.")

#     return {
#         "words": word_count,
#         "chars": char_count,
#         "hashtags": len(hashtags),
#         "mentions": len(mentions),
#         "urls": len(urls),
#         "emojis": sum(len(e) for e in emojis),
#         "questions": questions,
#         "sentiment": sentiment,
#         "suggestions": suggestions,
#     }


# def make_downloadable_bytes(content: str, filename: str) -> Tuple[str, bytes]:
#     return filename, content.encode("utf-8")


# def results_table_to_csv(results: List[Dict]) -> str:
#     import csv
#     import io as _io

#     fieldnames = ["post_index", "words", "chars", "hashtags", "mentions", "urls", "emojis", "questions", "sentiment", "suggestions"]
#     buf = _io.StringIO()
#     writer = csv.DictWriter(buf, fieldnames=fieldnames)
#     writer.writeheader()
#     for idx, row in enumerate(results, start=1):
#         writer.writerow({
#             "post_index": idx,
#             **{k: row[k] for k in fieldnames if k not in {"post_index", "suggestions"}},
#             "suggestions": " | ".join(row["suggestions"]),
#         })
#     return buf.getvalue()


# # ----------------------------- 
# # Streamlit UI
# st.set_page_config(page_title=APP_TITLE, page_icon="📝", layout="wide")
# st.title(APP_TITLE)
# st.caption(APP_DESC)

# with st.sidebar:
#     st.header("Upload")
#     files = st.file_uploader(
#         "Upload PDFs and/or images (PNG/JPG/JPEG/WEBP):",
#         type=["pdf", "png", "jpg", "jpeg", "webp"],
#         accept_multiple_files=True,
#     )
#     st.info("Tip: Separate multiple posts with a blank line, '---', or '###'.")

# process_clicked = st.button("Process Files", type="primary", use_container_width=True)

# if "extracted_text" not in st.session_state:
#     st.session_state["extracted_text"] = ""

# if process_clicked:
#     if not files:
#         st.warning("Please upload at least one PDF or image.")
#     else:
#         combined_text = []
#         progress = st.progress(0)
#         status = st.empty()
#         for i, f in enumerate(files, start=1):
#             ext = (f.name.split(".")[-1] or "").lower()
#             status.write(f"Processing **{f.name}** ...")
#             try:
#                 if ext == "pdf":
#                     text = extract_text_from_pdf(f.read())
#                 else:
#                     text = extract_text_from_image(f.read())
#                 if text:
#                     combined_text.append(f"### {f.name}\n{text.strip()}")
#                 else:
#                     combined_text.append(f"### {f.name}\n[No text detected]")
#             except Exception as e:
#                 st.error(f"Error processing {f.name}: {e}")
#             progress.progress(i / len(files))
#         status.write("Done.")
#         st.session_state["extracted_text"] = "\n\n---\n\n".join(combined_text)

# extracted = st.session_state.get("extracted_text", "")

# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Extracted Text")
#     st.write("You can edit before analysis.")
#     text_input = st.text_area("Text to analyze", value=extracted, height=350, label_visibility="collapsed")
#     st.session_state["extracted_text"] = text_input
#     if text_input:
#         fname, fbytes = make_downloadable_bytes(text_input, "extracted_text.txt")
#         st.download_button("Download Extracted Text", fbytes, file_name=fname, mime="text/plain")

# with col2:
#     st.subheader("Analysis")
#     if st.button("Run Analysis", use_container_width=True):
#         if not st.session_state["extracted_text"].strip():
#             st.warning("No text to analyze.")
#         else:
#             with st.spinner("Analyzing posts..."):
#                 analyzer = SentimentIntensityAnalyzer()
#                 posts = split_into_posts(st.session_state["extracted_text"])
#                 results = [analyze_post(p, analyzer) for p in posts]
#                 st.session_state["analysis_results"] = (posts, results)
#                 time.sleep(0.2)

#     if "analysis_results" in st.session_state:
#         posts, results = st.session_state["analysis_results"]
#         for i, (p, r) in enumerate(zip(posts, results), start=1):
#             with st.expander(f"Post {i} — words: {r['words']} | sentiment: {r['sentiment']:.2f}"):
#                 st.write(p)
#                 st.markdown("**Suggestions:**")
#                 if r["suggestions"]:
#                     for s in r["suggestions"]:
#                         st.write(f"- {s}")
#                 else:
#                     st.write("- Looks good! 👍")
#         csv_text = results_table_to_csv(results)
#         st.download_button(
#             "Download Analysis CSV",
#             csv_text.encode("utf-8"),
#             file_name="analysis.csv",
#             mime="text/csv",
#         )

# st.markdown("---")
# with st.expander("Troubleshooting"):
#     st.write(
#         """
# **Common issues**
# - *Tesseract not found*: Install it and set `TESSERACT_PATH` env var (Windows) or ensure it's in PATH.
# - *Garbled OCR*: Try higher-resolution images, good contrast, or scan in B/W. The app also tries light sharpening.
# - *PDFs with images only*: Those may need OCR per page (convert pages to images first). PyMuPDF text mode works best with real text PDFs.
#         """
#     )
import io
import os
import re
import time
from typing import List, Tuple, Dict

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageFilter, ImageOps
import pytesseract


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# ---------------- OCR & Sentiment Setup ----------------
# Set Tesseract path (Streamlit Cloud / Linux)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Test Tesseract (optional)
try:
    version = pytesseract.get_tesseract_version()
    print("Tesseract is working! Version:", version)
except Exception as e:
    print("Error detecting Tesseract:", e)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# ----------------------------- 
# App configuration
APP_TITLE = "Social Media Content Analyzer"
APP_DESC = (
    "Upload PDFs or images to extract text. Then analyze posts and get suggestions "
    "to improve engagement. Split posts with a blank line, '---', or '###'."
)

# ----------------------------- 
# Utils
from pdf2image import convert_from_bytes

# ---------- Updated Preprocessing ----------
def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Preprocess image to improve OCR accuracy:
    - Grayscale
    - Auto-contrast
    - Adaptive thresholding
    - Sharpen
    """
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    np_img = np.array(gray)
    np_img = 255 * (np_img > np_img.mean()).astype(np.uint8)
    bin_img = Image.fromarray(np_img)
    bin_img = bin_img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    return bin_img

def extract_text_from_image(image_bytes: bytes, lang: str = "eng") -> str:
    img = Image.open(io.BytesIO(image_bytes))
    img = preprocess_image(img)
    custom_oem_psm_config = r"--oem 1 --psm 3"
    try:
        text = pytesseract.image_to_string(img, lang=lang, config=custom_oem_psm_config)
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract is not installed or not in PATH. "
            "Install it and/or set TESSERACT_PATH env var."
        )
    return text.strip()

# ---------- Updated PDF Extraction ----------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text_chunks = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                st.write(f"Page {page_number}: Using OCR")
                pix = page.get_pixmap(dpi=400)
                img_bytes = pix.tobytes("png")
                text = extract_text_from_image(img_bytes)
            text_chunks.append(text)
    return "\n".join(text_chunks).strip()


# ---------- Other Utilities ----------
def split_into_posts(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(?:\n\s*\n)|(?:\n?---+\n?)|(?:\n?###\n?)", text)
    return [p.strip() for p in parts if p and p.strip()]

# Regex patterns
EMOJI_PATTERN = re.compile(
    "[" "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF\u2700-\u27BF" "]+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#([A-Za-z0-9_]+)")
MENTION_PATTERN = re.compile(r"(?<!\w)@([A-Za-z0-9_]+)")

def analyze_post(text: str, analyzer: SentimentIntensityAnalyzer) -> Dict:
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    char_count = len(text)
    hashtags = HASHTAG_PATTERN.findall(text)
    mentions = MENTION_PATTERN.findall(text)
    urls = URL_PATTERN.findall(text)
    emojis = EMOJI_PATTERN.findall(text)
    questions = text.count("?")
    sentiment = analyzer.polarity_scores(text)["compound"]

    suggestions = []
    if word_count < 15:
        suggestions.append("Add a bit more context—very short posts may underperform.")
    elif word_count > 220:
        suggestions.append("Trim the length or front-load key info to keep attention.")
    if len(hashtags) < 2:
        suggestions.append("Include 2–5 relevant #hashtags for discovery.")
    elif len(hashtags) > 8:
        suggestions.append("Reduce hashtag count; too many can look spammy.")
    if questions == 0:
        suggestions.append("Add a question or prompt to spark comments.")
    if not re.search(r"\b(like|comment|share|save|follow|subscribe|dm|link in bio)\b", text, re.I):
        suggestions.append("Add a clear call-to-action (e.g., 'Comment your thoughts').")
    if len(emojis) == 0:
        suggestions.append("A tasteful emoji can add tone and scannability.")
    elif len(emojis) > 10:
        suggestions.append("Use fewer emojis for readability.")
    if sentiment < -0.4:
        suggestions.append("Tone feels negative—consider reframing benefits or solutions.")
    if urls and not re.search(r"\b(why|what|how|learn|check)\b", text, re.I):
        suggestions.append("Explain why the link is worth clicking.")
    if not re.search(r"[A-Z].{0,40}:", text):
        suggestions.append("Consider a short hook/heading at the start to grab attention.")

    return {
        "words": word_count,
        "chars": char_count,
        "hashtags": len(hashtags),
        "mentions": len(mentions),
        "urls": len(urls),
        "emojis": sum(len(e) for e in emojis),
        "questions": questions,
        "sentiment": sentiment,
        "suggestions": suggestions,
    }

def make_downloadable_bytes(content: str, filename: str) -> Tuple[str, bytes]:
    return filename, content.encode("utf-8")

def results_table_to_csv(results: List[Dict]) -> str:
    import csv
    import io as _io
    fieldnames = ["post_index", "words", "chars", "hashtags", "mentions", "urls", "emojis", "questions", "sentiment", "suggestions"]
    buf = _io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for idx, row in enumerate(results, start=1):
        writer.writerow({
            "post_index": idx,
            **{k: row[k] for k in fieldnames if k not in {"post_index", "suggestions"}},
            "suggestions": " | ".join(row["suggestions"]),
        })
    return buf.getvalue()

# ----------------------------- 
# Streamlit UI
st.set_page_config(page_title=APP_TITLE, page_icon="📝", layout="wide")
st.title(APP_TITLE)
st.caption(APP_DESC)

with st.sidebar:
    st.header("Upload")
    files = st.file_uploader(
        "Upload PDFs and/or images (PNG/JPG/JPEG/WEBP):",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    st.info("Tip: Separate multiple posts with a blank line, '---', or '###'.")

process_clicked = st.button("Process Files", type="primary", use_container_width=True)

if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = ""

if process_clicked:
    if not files:
        st.warning("Please upload at least one PDF or image.")
    else:
        combined_text = []
        progress = st.progress(0)
        status = st.empty()
        for i, f in enumerate(files, start=1):
            ext = (f.name.split(".")[-1] or "").lower()
            status.write(f"Processing **{f.name}** ...")
            try:
                if ext == "pdf":
                    text = extract_text_from_pdf(f.read())
                else:
                    text = extract_text_from_image(f.read())
                if text:
                    combined_text.append(f"### {f.name}\n{text.strip()}")
                else:
                    combined_text.append(f"### {f.name}\n[No text detected]")
            except Exception as e:
                st.error(f"Error processing {f.name}: {e}")
            progress.progress(i / len(files))
        status.write("Done.")
        st.session_state["extracted_text"] = "\n\n---\n\n".join(combined_text)

extracted = st.session_state.get("extracted_text", "")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Extracted Text")
    st.write("You can edit before analysis.")
    text_input = st.text_area("Text to analyze", value=extracted, height=350, label_visibility="collapsed")
    st.session_state["extracted_text"] = text_input
    if text_input:
        fname, fbytes = make_downloadable_bytes(text_input, "extracted_text.txt")
        st.download_button("Download Extracted Text", fbytes, file_name=fname, mime="text/plain")

with col2:
    st.subheader("Analysis")
    if st.button("Run Analysis", use_container_width=True):
        if not st.session_state["extracted_text"].strip():
            st.warning("No text to analyze.")
        else:
            with st.spinner("Analyzing posts..."):
                analyzer = SentimentIntensityAnalyzer()
                posts = split_into_posts(st.session_state["extracted_text"])
                results = [analyze_post(p, analyzer) for p in posts]
                st.session_state["analysis_results"] = (posts, results)
                time.sleep(0.2)

    if "analysis_results" in st.session_state:
        posts, results = st.session_state["analysis_results"]
        for i, (p, r) in enumerate(zip(posts, results), start=1):
            with st.expander(f"Post {i} — words: {r['words']} | sentiment: {r['sentiment']:.2f}"):
                st.write(p)
                st.markdown("**Suggestions:**")
                if r["suggestions"]:
                    for s in r["suggestions"]:
                        st.write(f"- {s}")
                else:
                    st.write("- Looks good! 👍")
        csv_text = results_table_to_csv(results)
        st.download_button(
            "Download Analysis CSV",
            csv_text.encode("utf-8"),
            file_name="analysis.csv",
            mime="text/csv",
        )

st.markdown("---")
with st.expander("Troubleshooting"):
    st.write(
        """
**Common issues**
- *Tesseract not found*: Install it and set `TESSERACT_PATH` env var (Windows) or ensure it's in PATH.
- *Garbled OCR*: Try higher-resolution images, good contrast, or scan in B/W. The app also tries light sharpening.
- *PDFs with images only*: Those may need OCR per page (convert pages to images first). PyMuPDF text mode works best with real text PDFs.
        """
    )

