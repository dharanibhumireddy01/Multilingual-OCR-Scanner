# scanner.py
# Real-time webcam document scanner with OCR text extraction.
#
# This is the original script written in 2023 for my B.Tech final year project.
# It opens a live webcam feed, lets you capture a document photo,
# runs the 7-stage preprocessing pipeline, and extracts text using Tesseract.
#
# Controls:
#   s — capture image and extract text
#   q — quit without capturing
#
# Usage:
#   python scanner.py
#   python scanner.py --lang hin    (Hindi)
#   python scanner.py --lang tel    (Telugu)
#   python scanner.py --camera 0   (use built-in webcam instead of external)

import cv2
import os
import argparse

from src.config import (
    CAMERA_INDEX,
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    SCANNED_IMAGE_PATH,
    OUTPUT_DIR,
)
from src.preprocessor import preprocess
from src.ocr_engine   import extract_text


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam Document Scanner with OCR")
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANGUAGE,
        help=f"Tesseract language code. Options: {list(SUPPORTED_LANGUAGES.values())}"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help="Camera index (0 = built-in, 1 = external USB webcam)"
    )
    return parser.parse_args()


def run_scanner(camera_index: int = CAMERA_INDEX, language: str = DEFAULT_LANGUAGE):
    """
    Opens the webcam feed and waits for the user to capture an image.
    On capture (press 's'), runs OCR and prints extracted text.
    On quit (press 'q'), closes cleanly.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 50)
    print("  OCR Document Scanner")
    print("  Press 's' to scan | Press 'q' to quit")
    print(f"  Language: {language}")
    print("=" * 50)

    cam = cv2.VideoCapture(camera_index)

    if not cam.isOpened():
        print(f"Error: Could not open camera at index {camera_index}.")
        print("Try changing CAMERA_INDEX in src/config.py (0 for built-in, 1 for USB).")
        return

    try:
        while True:
            check, frame = cam.read()

            if not check or frame is None:
                print("Warning: Could not read frame from camera.")
                continue

            cv2.imshow("OCR Scanner — Press 's' to scan, 'q' to quit", frame)
            key = cv2.waitKey(1)

            if key == ord("s"):
                # Save the captured frame
                cv2.imwrite(SCANNED_IMAGE_PATH, frame)
                print(f"\nImage captured: {SCANNED_IMAGE_PATH}")

                cam.release()
                cv2.destroyAllWindows()

                # Run the preprocessing pipeline
                print("Running preprocessing pipeline...")
                processed = preprocess(frame, save_result=True)
                print("Preprocessing complete.")

                # Extract text
                print(f"Extracting text (language: {language})...")
                text = extract_text(processed, language=language, save_result=True)

                print("\n" + "=" * 50)
                print("  EXTRACTED TEXT")
                print("=" * 50)
                print(text if text else "(No text detected)")
                print("=" * 50)
                print(f"\nText saved to: outputs/extracted_text.txt")
                break

            elif key == ord("q"):
                print("Scanner closed.")
                cam.release()
                cv2.destroyAllWindows()
                break

    except KeyboardInterrupt:
        print("\nScanner interrupted.")
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    run_scanner(camera_index=args.camera, language=args.lang)
