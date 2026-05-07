# preprocessor.py
# The 7-stage image preprocessing pipeline.
#
# This was originally written in 2023 as part of my B.Tech final year project
# at SCSVMV University. The pipeline was designed to clean and enhance document
# images before passing them to Tesseract OCR, significantly improving
# text extraction accuracy on real-world document photos.
#
# The 7 stages in order:
#   1. Rotation correction
#   2. Color inversion (contrast enhancement)
#   3. Grayscale conversion
#   4. Otsu's adaptive binarization
#   5. Morphological noise removal (dilation, erosion, closing)
#   6. Median blur smoothing
#   7. Contour-based border detection and cropping

import cv2
import numpy as np
import os

from src.config import (
    MORPH_KERNEL_SIZE,
    MEDIAN_BLUR_KERNEL,
    THRESH_VALUE,
    PROCESSED_IMAGE_PATH,
    OUTPUT_DIR,
)


def rotate(image: np.ndarray) -> np.ndarray:
    """
    Stage 1: Rotate the image 90 degrees clockwise.
    Most document photos taken with a phone come in landscape orientation.
    This corrects the orientation so text runs left-to-right for Tesseract.
    """
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def invert_colors(image: np.ndarray) -> np.ndarray:
    """
    Stage 2: Invert image colors using bitwise NOT.
    Dark text on white background becomes white text on dark background.
    This improves contrast for documents with low-quality printing or
    faded text, making the subsequent binarization step more accurate.
    """
    return cv2.bitwise_not(image)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Stage 3: Convert BGR image to grayscale.
    Reduces the image from 3 channels (Blue, Green, Red) to 1 channel.
    Required before thresholding — Otsu's method only works on single-channel images.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize(image: np.ndarray) -> np.ndarray:
    """
    Stage 4: Otsu's adaptive binarization.
    Converts the grayscale image to pure black and white.

    Why Otsu's method instead of a fixed threshold?
    Otsu's automatically calculates the optimal threshold value by
    analyzing the histogram of pixel intensities. This works much
    better on documents with uneven lighting or background variation —
    a fixed threshold of 128 would fail on a photo taken under a lamp
    that creates shadows on one side of the document.
    """
    _, binary = cv2.threshold(
        image,
        THRESH_VALUE,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return binary


def remove_noise(image: np.ndarray) -> np.ndarray:
    """
    Stage 5: Morphological noise removal — dilation, erosion, closing.

    These three operations remove small black dots and artifacts
    that appear in the binary image due to paper texture, dust,
    or compression artifacts in the photo.

    Dilation:  expands white regions (fills small gaps in text strokes)
    Erosion:   shrinks white regions (removes thin noise pixels)
    Closing:   dilation followed by erosion (closes small holes inside letters)

    The kernel size (1,1) is intentionally small — we only want to remove
    tiny noise pixels without distorting the actual text characters.
    """
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    dilated  = cv2.dilate(image, kernel, iterations=1)
    eroded   = cv2.erode(dilated, kernel, iterations=1)
    closed   = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
    return closed


def smooth(image: np.ndarray) -> np.ndarray:
    """
    Stage 6: Median blur smoothing.
    Removes any remaining salt-and-pepper noise (isolated white/black pixels)
    while preserving the edges of text characters.

    Median blur replaces each pixel with the median of its neighborhood —
    isolated noise pixels (which are extreme values) get replaced by the
    surrounding normal values, while text edges (which are consistent
    across multiple pixels) are preserved.
    """
    return cv2.medianBlur(image, MEDIAN_BLUR_KERNEL)


def crop_borders(image: np.ndarray) -> np.ndarray:
    """
    Stage 7: Contour-based border detection and cropping.
    Finds the largest contour in the image (which should be the document border)
    and crops to its bounding rectangle.

    This removes dark borders, table edges, and frame artifacts that
    confuse Tesseract into trying to read non-text regions.
    Falls back to the original image if contour detection fails.
    """
    contours, _ = cv2.findContours(
        image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    # Sort by area, take the smallest contour (the tightest text boundary)
    sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    cnt = sorted_contours[-1]
    x, y, w, h = cv2.boundingRect(cnt)

    # Safety check — don't crop to a tiny region
    if w < 10 or h < 10:
        return image

    return image[y:y + h, x:x + w]


def preprocess(image: np.ndarray, save_result: bool = True) -> np.ndarray:
    """
    Runs all 7 preprocessing stages in order on a given image.
    Optionally saves the processed image to outputs/ProcessedImage.jpg.

    This is the main function called by the scanner and the batch processor.
    """
    processed = rotate(image)
    processed = invert_colors(processed)
    processed = to_grayscale(processed)
    processed = binarize(processed)
    processed = remove_noise(processed)
    processed = smooth(processed)
    processed = crop_borders(processed)

    if save_result:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cv2.imwrite(PROCESSED_IMAGE_PATH, processed)

    return processed


def preprocess_from_path(image_path: str, save_result: bool = True) -> np.ndarray:
    """
    Loads an image from disk and runs the full preprocessing pipeline.
    Raises FileNotFoundError if the image path does not exist.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return preprocess(image, save_result=save_result)
