# config.py
# All project-wide constants in one place.
# Change anything here and it flows through the whole pipeline.

import os

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

SCANNED_IMAGE_PATH = os.path.join(OUTPUT_DIR, "ScannedImage.jpg")
PROCESSED_IMAGE_PATH = os.path.join(OUTPUT_DIR, "ProcessedImage.jpg")
EXTRACTED_TEXT_PATH  = os.path.join(OUTPUT_DIR, "extracted_text.txt")

# ── tesseract ──────────────────────────────────────────────────────────────
# Windows default path — change this if Tesseract is installed elsewhere
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Supported languages (install Tesseract language packs for each)
# eng = English (default, always installed)
# hin = Hindi, tel = Telugu, tam = Tamil, fra = French, deu = German
SUPPORTED_LANGUAGES = {
    "English":  "eng",
    "Hindi":    "hin",
    "Telugu":   "tel",
    "Tamil":    "tam",
    "French":   "fra",
    "German":   "deu",
    "Spanish":  "spa",
}
DEFAULT_LANGUAGE = "eng"

# ── camera ─────────────────────────────────────────────────────────────────
# Camera index: 0 = built-in webcam, 1 = external webcam
# Change to 0 if your laptop camera is not detected on index 1
CAMERA_INDEX = 0

# ── preprocessing ──────────────────────────────────────────────────────────
# Morphological kernel size for noise removal
MORPH_KERNEL_SIZE = (1, 1)

# Median blur kernel size (must be odd number)
MEDIAN_BLUR_KERNEL = 3

# Otsu threshold value (128 is standard)
THRESH_VALUE = 128

# ── CNN model ──────────────────────────────────────────────────────────────
CNN_MODEL_PATH  = os.path.join(BASE_DIR, "outputs", "cnn_ocr_model.h5")
CNN_IMG_HEIGHT  = 32
CNN_IMG_WIDTH   = 128
CNN_BATCH_SIZE  = 32
CNN_EPOCHS      = 10

# Characters the CNN model recognises
CNN_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?-"
