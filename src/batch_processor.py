# batch_processor.py
# Process multiple document images in one go without needing a webcam.
# Place your images in the samples/ folder and run this script.
#
# Usage:
#   python batch_processor.py
#   python batch_processor.py --lang tel
#   python batch_processor.py --folder my_documents/
#   python batch_processor.py --multilingual

import os
import argparse
import cv2
from typing import List

from src.config import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    SAMPLES_DIR,
    OUTPUT_DIR,
)
from src.preprocessor import preprocess_from_path
from src.ocr_engine   import extract_text, extract_multilingual


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch OCR Processor")
    parser.add_argument("--folder",        default=SAMPLES_DIR,
                        help="Folder containing images to process")
    parser.add_argument("--lang",          default=DEFAULT_LANGUAGE,
                        help="Tesseract language code")
    parser.add_argument("--multilingual",  action="store_true",
                        help="Try all supported languages and pick best result")
    return parser.parse_args()


def get_image_files(folder: str) -> List[str]:
    """Returns all supported image file paths in the given folder."""
    if not os.path.exists(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]


def process_single(image_path: str,
                   language: str = DEFAULT_LANGUAGE,
                   multilingual: bool = False) -> str:
    """
    Runs the full pipeline on one image:
        load → preprocess → OCR → return text

    If multilingual=True, tries all installed language packs and
    returns the result with the highest average confidence score.
    """
    print(f"\nProcessing: {os.path.basename(image_path)}")

    try:
        processed = preprocess_from_path(image_path, save_result=False)
    except (FileNotFoundError, ValueError) as e:
        print(f"  Error: {e}")
        return ""

    if multilingual:
        langs = list(SUPPORTED_LANGUAGES.values())
        text  = extract_multilingual(processed, languages=langs)
        print(f"  Multilingual extraction complete")
    else:
        text = extract_text(processed, language=language, save_result=False)
        print(f"  Language: {language}")

    word_count = len(text.split()) if text else 0
    print(f"  Words extracted: {word_count}")

    return text


def run_batch(folder: str = SAMPLES_DIR,
              language: str = DEFAULT_LANGUAGE,
              multilingual: bool = False) -> None:
    """
    Processes all images in the folder and saves each result as a .txt file
    in outputs/ with the same base filename as the image.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = get_image_files(folder)

    if not image_files:
        print(f"No images found in: {folder}")
        print(f"Supported formats: {SUPPORTED_EXTENSIONS}")
        print(f"Place your images in the '{folder}/' folder and run again.")
        return

    print("=" * 55)
    print(f"  Batch OCR Processor")
    print(f"  Images found : {len(image_files)}")
    print(f"  Language     : {'auto-detect' if multilingual else language}")
    print(f"  Output folder: {OUTPUT_DIR}")
    print("=" * 55)

    results = {}
    for image_path in image_files:
        text = process_single(image_path, language=language, multilingual=multilingual)
        results[image_path] = text

        # Save individual result
        base_name    = os.path.splitext(os.path.basename(image_path))[0]
        output_path  = os.path.join(OUTPUT_DIR, f"{base_name}_extracted.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  Saved: {output_path}")

    # Save combined results
    combined_path = os.path.join(OUTPUT_DIR, "all_extracted_text.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        for path, text in results.items():
            f.write(f"=== {os.path.basename(path)} ===\n")
            f.write(text)
            f.write("\n\n")

    print("\n" + "=" * 55)
    print(f"  Batch complete — {len(image_files)} images processed")
    print(f"  Combined output: {combined_path}")
    print("=" * 55)


if __name__ == "__main__":
    args = parse_args()
    run_batch(
        folder=args.folder,
        language=args.lang,
        multilingual=args.multilingual,
    )
