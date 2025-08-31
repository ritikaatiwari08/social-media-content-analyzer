from PIL import Image, ImageDraw, ImageFont

# Create a white image
img = Image.new('RGB', (300, 100), color='white')

# Initialize drawing
d = ImageDraw.Draw(img)

# Optional: load a default font
try:
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()

# Add text
d.text((10, 30), "Hello World!", fill='black', font=font)

# Save the image
img.save("test_image.png")

print("test_image.png created successfully!")
