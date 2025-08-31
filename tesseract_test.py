from PIL import Image
import pytesseract

# Set tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Open your test image
img = Image.open("test_image.png")

# OCR
text = pytesseract.image_to_string(img)

print("Extracted text:", text)
