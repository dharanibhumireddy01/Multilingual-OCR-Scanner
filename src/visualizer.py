# visualizer.py
# Visualizes each stage of the preprocessing pipeline side by side.
# Useful for debugging, documentation, and explaining the pipeline
# to interviewers or in project presentations.
#
# Usage:
#   python -c "from src.visualizer import show_pipeline; show_pipeline('samples/sample.jpg')"

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR
from src.preprocessor import (
    rotate, invert_colors, to_grayscale,
    binarize, remove_noise, smooth, crop_borders,
)


def show_pipeline(image_path: str, save: bool = True) -> None:
    """
    Loads an image and shows all 7 preprocessing stages side by side
    in a single matplotlib figure.

    Args:
        image_path : path to input image file
        save       : if True, saves the figure to outputs/pipeline_stages.png
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    original = cv2.imread(image_path)
    if original is None:
        print(f"Could not read image: {image_path}")
        return

    # Run each stage and collect
    stage1 = rotate(original)
    stage2 = invert_colors(stage1)
    stage3 = to_grayscale(stage2)
    stage4 = binarize(stage3)
    stage5 = remove_noise(stage4)
    stage6 = smooth(stage5)
    stage7 = crop_borders(stage6)

    stages = [
        (original, "Original"),
        (stage1,   "Stage 1: Rotation"),
        (stage2,   "Stage 2: Color Inversion"),
        (stage3,   "Stage 3: Grayscale"),
        (stage4,   "Stage 4: Otsu Binarization"),
        (stage5,   "Stage 5: Noise Removal"),
        (stage6,   "Stage 6: Median Blur"),
        (stage7,   "Stage 7: Border Crop"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        "OCR Preprocessing Pipeline — 7 Stages",
        fontsize=14,
        fontweight="bold"
    )

    for ax, (img, title) in zip(axes.flatten(), stages):
        # Convert BGR to RGB for matplotlib, handle grayscale
        if len(img.shape) == 3:
            display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(display)
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "pipeline_stages.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Pipeline visualization saved: {save_path}")

    plt.show()
    plt.close(fig)


def compare_before_after(image_path: str, language: str = "eng") -> None:
    """
    Shows original image vs final processed image side by side,
    with extracted text printed below.
    """
    from src.preprocessor import preprocess_from_path
    from src.ocr_engine   import extract_text

    original  = cv2.imread(image_path)
    processed = preprocess_from_path(image_path, save_result=False)
    text      = extract_text(processed, language=language)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Before vs After Preprocessing", fontsize=13, fontweight="bold")

    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(processed, cmap="gray")
    axes[1].set_title("After 7-Stage Preprocessing", fontweight="bold")
    axes[1].axis("off")

    plt.figtext(
        0.5, 0.02,
        f"Extracted text: {text[:200]}..." if len(text) > 200 else f"Extracted text: {text}",
        ha="center", fontsize=9, wrap=True
    )

    plt.tight_layout()
    plt.show()
    plt.close(fig)
