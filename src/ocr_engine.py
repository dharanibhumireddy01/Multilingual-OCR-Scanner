# ocr_engine.py
# Text extraction using Tesseract OCR with multilingual support.
#
# Originally this was a single pytesseract.image_to_string() call.
# This version adds:
#   - Configurable language selection (English, Hindi, Telugu, Tamil, etc.)
#   - Confidence scoring per detected word
#   - Structured output (bounding boxes + text)
#   - Auto language detection hint
#   - Clean text post-processing

import os
import pytesseract
import numpy as np
from typing import Dict, List

from src.config import (
    TESSERACT_CMD,
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    EXTRACTED_TEXT_PATH,
    OUTPUT_DIR,
)

# Set Tesseract executable path (Windows)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def extract_text(
    image: np.ndarray,
    language: str = DEFAULT_LANGUAGE,
    save_result: bool = True,
) -> str:
    """
    Extract text from a preprocessed image using Tesseract OCR.

    Args:
        image    : preprocessed grayscale/binary numpy image array
        language : Tesseract language code e.g. 'eng', 'hin', 'tel'
        save_result : whether to save extracted text to outputs/

    Returns:
        Extracted text as a clean string.
    """
    # Validate language code
    if language not in SUPPORTED_LANGUAGES.values():
        print(f"Warning: '{language}' not in supported list. Falling back to English.")
        language = DEFAULT_LANGUAGE

    try:
        text = pytesseract.image_to_string(image, lang=language)
        text = _clean_text(text)
    except pytesseract.TesseractError as e:
        print(f"Tesseract error: {e}")
        print("Check that Tesseract is installed and the language pack is available.")
        text = ""

    if save_result and text:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(EXTRACTED_TEXT_PATH, "w", encoding="utf-8") as f:
            f.write(text)

    return text


def extract_with_confidence(
    image: np.ndarray,
    language: str = DEFAULT_LANGUAGE,
) -> List[Dict]:
    """
    Extract text with per-word confidence scores and bounding box locations.
    Returns a list of dicts, each containing:
        text       : the detected word
        confidence : Tesseract confidence score (0-100)
        left, top, width, height : bounding box in pixels

    Useful for filtering low-confidence detections and building
    structured output for downstream NLP workflows.
    """
    try:
        data = pytesseract.image_to_data(
            image,
            lang=language,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractError as e:
        print(f"Tesseract error: {e}")
        return []

    results = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = int(data["conf"][i])

        if word and conf > 0:
            results.append({
                "text":       word,
                "confidence": conf,
                "left":       data["left"][i],
                "top":        data["top"][i],
                "width":      data["width"][i],
                "height":     data["height"][i],
            })

    return results


def extract_multilingual(image: np.ndarray, languages: List[str]) -> str:
    """
    Try extraction in multiple languages and return the result
    with the highest average word confidence.

    This is useful when you don't know the document language in advance.
    Tesseract also supports combined language codes like 'eng+hin' which
    can detect mixed-language documents in a single pass.

    Args:
        image     : preprocessed image array
        languages : list of Tesseract language codes to try

    Returns:
        Best extracted text string.
    """
    best_text  = ""
    best_score = -1

    for lang in languages:
        words = extract_with_confidence(image, language=lang)
        if not words:
            continue

        avg_conf = sum(w["confidence"] for w in words) / len(words)
        text     = " ".join(w["text"] for w in words)

        if avg_conf > best_score:
            best_score = avg_conf
            best_text  = text

    return _clean_text(best_text)


def get_available_languages() -> List[str]:
    """
    Returns a list of language codes currently installed in Tesseract.
    If Tesseract is not installed, returns an empty list.
    """
    try:
        langs = pytesseract.get_languages(config="")
        return langs
    except Exception:
        return []


def _clean_text(text: str) -> str:
    """
    Post-process extracted text:
      - Strip leading/trailing whitespace
      - Collapse multiple blank lines into one
      - Remove non-printable characters
    """
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines = []
    blank_count = 0

    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)
            blank_count = 0
        else:
            blank_count += 1
            if blank_count == 1:
                cleaned_lines.append("")

    return "\n".join(cleaned_lines).strip()
