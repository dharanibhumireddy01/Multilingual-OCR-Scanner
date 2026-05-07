# main.py
# Entry point for the OCR Document Scanner.
#
# Modes:
#   scan        — open webcam and scan a document (default)
#   batch       — process all images in samples/ folder
#   image       — process a single image file
#   visualize   — show preprocessing pipeline stages for an image
#   languages   — list all installed Tesseract language packs
#
# Usage:
#   python main.py                          webcam scanner, English
#   python main.py --mode scan --lang hin   webcam scanner, Hindi
#   python main.py --mode batch             batch process samples/ folder
#   python main.py --mode image --file samples/doc.jpg
#   python main.py --mode visualize --file samples/doc.jpg
#   python main.py --mode languages         list installed languages

import argparse
import os
import sys

from src.config import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES, SAMPLES_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multilingual OCR Document Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              open webcam scanner
  python main.py --mode batch                 batch process samples/
  python main.py --mode image --file doc.jpg  process single image
  python main.py --mode visualize --file doc.jpg  show pipeline stages
  python main.py --mode languages             list installed languages
        """
    )
    parser.add_argument(
        "--mode",
        choices=["scan", "batch", "image", "visualize", "languages"],
        default="scan",
        help="Operating mode (default: scan)"
    )
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANGUAGE,
        help=f"Tesseract language code (default: {DEFAULT_LANGUAGE})"
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Path to image file (required for --mode image and --mode visualize)"
    )
    parser.add_argument(
        "--folder",
        default=SAMPLES_DIR,
        help=f"Folder of images for batch mode (default: {SAMPLES_DIR})"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for webcam mode (default: 0)"
    )
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Try all language packs and return best result (batch mode)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  Multilingual OCR Document Scanner")
    print("  Author: Dharani Bhumireddy")
    print("  B.Tech ECE (2023) | MS Data Science (2024-2026)")
    print("=" * 55)

    if args.mode == "scan":
        from src.scanner import run_scanner
        run_scanner(camera_index=args.camera, language=args.lang)

    elif args.mode == "batch":
        from src.batch_processor import run_batch
        run_batch(
            folder=args.folder,
            language=args.lang,
            multilingual=args.multilingual,
        )

    elif args.mode == "image":
        if not args.file:
            print("Error: --file is required for --mode image")
            print("Example: python main.py --mode image --file samples/doc.jpg")
            sys.exit(1)
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)

        from src.preprocessor import preprocess_from_path
        from src.ocr_engine   import extract_text

        print(f"Processing: {args.file}")
        processed = preprocess_from_path(args.file, save_result=True)
        text = extract_text(processed, language=args.lang, save_result=True)

        print("\n" + "=" * 55)
        print("  EXTRACTED TEXT")
        print("=" * 55)
        print(text if text else "(No text detected)")
        print("=" * 55)

    elif args.mode == "visualize":
        if not args.file:
            print("Error: --file is required for --mode visualize")
            print("Example: python main.py --mode visualize --file samples/doc.jpg")
            sys.exit(1)

        from src.visualizer import show_pipeline
        show_pipeline(args.file)

    elif args.mode == "languages":
        from src.ocr_engine import get_available_languages
        langs = get_available_languages()
        print("\nInstalled Tesseract language packs:")
        if langs:
            for lang in langs:
                name = next(
                    (k for k, v in SUPPORTED_LANGUAGES.items() if v == lang),
                    lang
                )
                print(f"  {lang:10s}  {name}")
        else:
            print("  Could not detect languages — check Tesseract installation.")
        print(f"\nTo add languages: download .traineddata files from")
        print(f"https://github.com/tesseract-ocr/tessdata")
        print(f"and place them in your Tesseract tessdata/ folder.")


if __name__ == "__main__":
    main()
