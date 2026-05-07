# tests/test_preprocessor.py
# Unit tests for the preprocessing pipeline.
# Run with: pytest tests/ -v

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import pytest

from src.preprocessor import (
    rotate, invert_colors, to_grayscale,
    binarize, remove_noise, smooth, crop_borders,
    preprocess,
)


# ── shared fixture ─────────────────────────────────────────────────────────
@pytest.fixture
def sample_color_image():
    """A small realistic BGR image with white background and black text region."""
    img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (20, 20), (180, 80), (0, 0, 0), -1)
    cv2.putText(img, "TEST", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img


@pytest.fixture
def sample_gray_image():
    """A simple grayscale image for testing single-channel operations."""
    img = np.ones((100, 200), dtype=np.uint8) * 200
    cv2.rectangle(img, (10, 10), (190, 90), 50, -1)
    return img


# ── rotation ───────────────────────────────────────────────────────────────
def test_rotate_changes_dimensions(sample_color_image):
    h, w = sample_color_image.shape[:2]
    rotated = rotate(sample_color_image)
    assert rotated.shape[0] == w
    assert rotated.shape[1] == h


def test_rotate_preserves_channels(sample_color_image):
    rotated = rotate(sample_color_image)
    assert rotated.shape[2] == 3


def test_rotate_not_same_as_original(sample_color_image):
    rotated = rotate(sample_color_image)
    assert not np.array_equal(rotated, sample_color_image)


# ── color inversion ────────────────────────────────────────────────────────
def test_invert_colors_reverses_values(sample_color_image):
    inverted = invert_colors(sample_color_image)
    # White (255) becomes black (0), black (0) becomes white (255)
    assert inverted.max() == 255
    assert inverted.min() == 0


def test_invert_twice_equals_original(sample_color_image):
    double_inverted = invert_colors(invert_colors(sample_color_image))
    assert np.array_equal(double_inverted, sample_color_image)


# ── grayscale ──────────────────────────────────────────────────────────────
def test_grayscale_reduces_to_one_channel(sample_color_image):
    gray = to_grayscale(sample_color_image)
    assert len(gray.shape) == 2


def test_grayscale_preserves_height_width(sample_color_image):
    h, w = sample_color_image.shape[:2]
    gray = to_grayscale(sample_color_image)
    assert gray.shape == (h, w)


# ── binarization ───────────────────────────────────────────────────────────
def test_binarize_produces_binary_image(sample_gray_image):
    binary = binarize(sample_gray_image)
    unique_vals = np.unique(binary)
    # Binary image should only contain 0 and 255
    assert set(unique_vals).issubset({0, 255})


def test_binarize_same_shape(sample_gray_image):
    binary = binarize(sample_gray_image)
    assert binary.shape == sample_gray_image.shape


# ── noise removal ──────────────────────────────────────────────────────────
def test_remove_noise_same_shape(sample_gray_image):
    binary  = binarize(sample_gray_image)
    cleaned = remove_noise(binary)
    assert cleaned.shape == binary.shape


def test_remove_noise_stays_binary(sample_gray_image):
    binary  = binarize(sample_gray_image)
    cleaned = remove_noise(binary)
    unique_vals = np.unique(cleaned)
    assert set(unique_vals).issubset({0, 255})


# ── smoothing ──────────────────────────────────────────────────────────────
def test_smooth_same_shape(sample_gray_image):
    binary   = binarize(sample_gray_image)
    smoothed = smooth(binary)
    assert smoothed.shape == binary.shape


# ── border crop ────────────────────────────────────────────────────────────
def test_crop_borders_returns_array(sample_gray_image):
    binary  = binarize(sample_gray_image)
    cropped = crop_borders(binary)
    assert isinstance(cropped, np.ndarray)


def test_crop_borders_not_larger_than_input(sample_gray_image):
    binary  = binarize(sample_gray_image)
    cropped = crop_borders(binary)
    assert cropped.shape[0] <= binary.shape[0]
    assert cropped.shape[1] <= binary.shape[1]


# ── full pipeline ──────────────────────────────────────────────────────────
def test_preprocess_returns_2d_array(sample_color_image):
    result = preprocess(sample_color_image, save_result=False)
    assert len(result.shape) == 2


def test_preprocess_output_is_binary(sample_color_image):
    result = preprocess(sample_color_image, save_result=False)
    unique_vals = np.unique(result)
    assert set(unique_vals).issubset({0, 255})


def test_preprocess_no_nan(sample_color_image):
    result = preprocess(sample_color_image, save_result=False)
    assert not np.any(np.isnan(result.astype(float)))
