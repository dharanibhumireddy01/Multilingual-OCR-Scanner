# tests/test_ocr_engine.py
# Unit tests for the OCR engine module.
# Run with: pytest tests/ -v
# Note: These tests mock Tesseract so they run without Tesseract installed.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.ocr_engine import _clean_text, get_available_languages


# ── text cleaning ──────────────────────────────────────────────────────────
def test_clean_text_strips_whitespace():
    result = _clean_text("  hello world  ")
    assert result == "hello world"


def test_clean_text_collapses_blank_lines():
    text = "line one\n\n\n\nline two"
    result = _clean_text(text)
    assert "\n\n\n" not in result


def test_clean_text_empty_string():
    assert _clean_text("") == ""


def test_clean_text_none_equivalent():
    assert _clean_text("   \n   \n   ") == ""


def test_clean_text_preserves_content():
    text = "Hello World\nThis is a test"
    result = _clean_text(text)
    assert "Hello World" in result
    assert "This is a test" in result


def test_clean_text_handles_single_line():
    result = _clean_text("Single line of text")
    assert result == "Single line of text"


# ── language detection ─────────────────────────────────────────────────────
def test_get_available_languages_returns_list():
    langs = get_available_languages()
    assert isinstance(langs, list)


# ── extract text with mock ────────────────────────────────────────────────
@patch("src.ocr_engine.pytesseract.image_to_string")
def test_extract_text_returns_string(mock_tesseract):
    mock_tesseract.return_value = "Hello World"
    from src.ocr_engine import extract_text
    dummy_image = np.ones((100, 200), dtype=np.uint8) * 255
    result = extract_text(dummy_image, save_result=False)
    assert isinstance(result, str)


@patch("src.ocr_engine.pytesseract.image_to_string")
def test_extract_text_cleans_output(mock_tesseract):
    mock_tesseract.return_value = "  Hello World  \n\n\n"
    from src.ocr_engine import extract_text
    dummy_image = np.ones((100, 200), dtype=np.uint8) * 255
    result = extract_text(dummy_image, save_result=False)
    assert "Hello World" in result
    assert result == result.strip()


@patch("src.ocr_engine.pytesseract.image_to_string")
def test_extract_text_fallback_on_bad_language(mock_tesseract):
    mock_tesseract.return_value = "some text"
    from src.ocr_engine import extract_text
    dummy_image = np.ones((50, 50), dtype=np.uint8) * 200
    # Should not raise even with an invalid language code
    result = extract_text(dummy_image, language="xyz", save_result=False)
    assert isinstance(result, str)


@patch("src.ocr_engine.pytesseract.image_to_data")
def test_extract_with_confidence_returns_list(mock_data):
    mock_data.return_value = {
        "text":   ["Hello", "World", ""],
        "conf":   [95, 87, -1],
        "left":   [10, 60, 0],
        "top":    [10, 10, 0],
        "width":  [40, 40, 0],
        "height": [20, 20, 0],
    }
    from src.ocr_engine import extract_with_confidence
    dummy_image = np.ones((100, 200), dtype=np.uint8) * 255
    result = extract_with_confidence(dummy_image)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["text"] == "Hello"
    assert result[0]["confidence"] == 95
