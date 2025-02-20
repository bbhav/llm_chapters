import cv2
import pytesseract
import re
import numpy as np


def preprocess_image(image_path):
    """Load the image, convert to grayscale, apply thresholding for better OCR"""
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to remove noise
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return processed


def extract_text(image_path):
    """Apply OCR to extract text from the PAN card image"""
    processed_image = preprocess_image(image_path)

    # Use optimized OCR settings for better text extraction
    custom_config = r"--oem 3 --psm 6"  # OCR Engine Mode 3, Page Segmentation Mode 6
    text = pytesseract.image_to_string(processed_image, config=custom_config)

    return text


def extract_pan_card_details(text):
    """Extract Name, Father's Name, Date of Birth, and PAN Number using regex"""

    # Extract Name
    name_match = re.search(r"Name\s*\n([A-Z\s]+)", text)
    name = name_match.group(1).strip() if name_match else "Not Found"

    # Extract Father's Name
    father_name_match = re.search(r"Father’s Name\s*\n([A-Z\s]+)", text)
    father_name = (
        father_name_match.group(1).strip() if father_name_match else "Not Found"
    )

    # Extract Date of Birth (DOB)
    dob_match = re.search(r"Date of Birth\s*\n(\d{2}/\d{2}/\d{4})", text)
    dob = dob_match.group(1).strip() if dob_match else "Not Found"

    # Extract PAN Number (10-character alphanumeric format)
    pan_match = re.search(r"([A-Z]{5}[0-9]{4}[A-Z]{1})", text)
    pan_number = pan_match.group(1).strip() if pan_match else "Not Found"

    return {
        "Name": name,
        "Father's Name": father_name,
        "Date of Birth": dob,
        "PAN Number": pan_number,
    }


# Example usage
image_path = "a.jpg"  # Change this to your PAN card image file
extracted_text = extract_text(image_path)
details = extract_pan_card_details(extracted_text)

# Display extracted details
print("\n🔍 Extracted PAN Card Details:")
for key, value in details.items():
    print(f"{key}: {value}")

import pytesseract
from PIL import Image

# Test OCR on the image
image_path = "a.jpg"  # Replace with your image path
text = pytesseract.image_to_string(Image.open(image_path))

# Print extracted text
print("\n🔍 OCR Extracted Text:\n", text)


def preprocess_image(image_path):
    """Improve text visibility for better OCR accuracy"""
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (binarization) to enhance contrast
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Denoise the image to remove small artifacts
    denoised = cv2.fastNlMeansDenoising(thresh, h=30)

    # Sharpen the image (useful for low-quality text)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    return sharpened


# Save the preprocessed image (for debugging)
image_path = "a.jpg"
processed_image = preprocess_image(image_path)
cv2.imwrite("processed_pan.jpg", processed_image)

# Perform OCR with better settings
text = pytesseract.image_to_string(processed_image, config="--oem 3 --psm 6")

print("\n🔍 Improved OCR Extracted Text:\n", text)


def crop_text_area(image_path):
    """Automatically detect text region and crop the image"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use edge detection to find text contours
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be text area)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Crop only the text area
    cropped_image = image[y : y + h, x : x + w]

    return cropped_image


cropped = crop_text_area("a.jpg")
cv2.imwrite("cropped_pan.jpg", cropped)

text = pytesseract.image_to_string(cropped, config="--oem 3 --psm 7")
print("\n🔍 Improved OCR After Cropping:\n", text)


def extract_pan_card_details(text):
    """Extract Name, Father's Name, Date of Birth, and PAN Number using regex"""

    # Extract Name
    name_match = re.search(r"Name\s*\n([A-Z\s]+)", text, re.IGNORECASE)
    name = name_match.group(1).strip() if name_match else "Not Found"

    # Extract Father's Name
    father_name_match = re.search(
        r"Father[’']?s Name\s*\n([A-Z\s]+)", text, re.IGNORECASE
    )
    father_name = (
        father_name_match.group(1).strip() if father_name_match else "Not Found"
    )

    # Extract Date of Birth (DOB)
    dob_match = re.search(
        r"Date of Birth\s*\n?(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE
    )
    dob = dob_match.group(1).strip() if dob_match else "Not Found"

    # Extract PAN Number (10-character alphanumeric format)
    pan_match = re.search(r"([A-Z]{5}[0-9]{4}[A-Z])", text)
    pan_number = pan_match.group(1).strip() if pan_match else "Not Found"

    return {
        "Name": name,
        "Father's Name": father_name,
        "Date of Birth": dob,
        "PAN Number": pan_number,
    }


# Extract details again after fixing OCR
details = extract_pan_card_details(text)

print("\n🔍 Extracted PAN Card Details:")
for key, value in details.items():
    print(f"{key}: {value}")
