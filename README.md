# Multilingual OCR Document Scanner

![Tests](https://github.com/dharanibhumireddy01/Multilingual-OCR-Scanner/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![Accuracy](https://img.shields.io/badge/accuracy-89%25-orange)

**Author:** Dharani Bhumireddy
**B.Tech ECE** — SCSVMV University, Kanchipuram · GPA 3.8 · Graduated 2023
**MS Data Science** — University at Albany, SUNY · GPA 3.7
**Contact:** dharanibhumireddy.ds@gmail.com
**LinkedIn:** [linkedin.com/in/dharani-bhumireddy](https://linkedin.com/in/dharani-bhumireddy)

---

## Table of Contents

1. [Project Origin](#1-project-origin)
2. [What This Project Does](#2-what-this-project-does)
3. [How to Use It](#3-how-to-use-it)
4. [The 7-Stage Preprocessing Pipeline](#4-the-7-stage-preprocessing-pipeline)
5. [CNN Architecture](#5-cnn-architecture)
6. [Multilingual Support](#6-multilingual-support)
7. [Results](#7-results)
8. [Project Structure](#8-project-structure)
9. [Setup and Installation](#9-setup-and-installation)
10. [Running the Tests](#10-running-the-tests)
11. [Configuration](#11-configuration)
12. [Real-World Applications](#12-real-world-applications)
13. [Technical Stack](#13-technical-stack)

---

## 1. Project Origin

This project was originally built in **2023** as part of my B.Tech final year project in Electronics and Communication Engineering at SCSVMV University, Kanchipuram. The core problem came from a real observation — documents in India are often multilingual, photographed under poor lighting conditions, and scanned using mobile phones rather than flatbed scanners. Standard OCR tools performed poorly on these inputs without preprocessing.

The original 2023 version was a single Python script (`scanner.py`) that opened a webcam, captured an image, ran a preprocessing pipeline, and extracted text using Tesseract. That original code is preserved exactly in `src/scanner.py`.

This repository extends the original work with:
- A proper modular package structure
- Batch processing for multiple images
- CNN-based character recognition layer (TensorFlow/PyTorch)
- Multilingual auto-detection across 7 languages
- Unit tests and CI pipeline
- Full documentation

---

## 2. What This Project Does

A real-time document scanner that:

1. Opens a live webcam feed (or accepts a saved image)
2. Applies a 7-stage image preprocessing pipeline to clean and enhance the image
3. Passes the cleaned image to Tesseract OCR for text extraction
4. Optionally runs a CNN model for character-level confidence scoring
5. Supports English, Hindi, Telugu, Tamil, French, German, and Spanish
6. Saves extracted text to a structured output file

```
Webcam / Image File
        |
        v
7-Stage Preprocessing Pipeline
rotation → inversion → grayscale → Otsu binarization
→ morphological noise removal → median blur → border crop
        |
        v
Tesseract OCR Engine
(with language selection and confidence scoring)
        |
        v
CNN Character Recognition (optional layer)
(89% accuracy via data augmentation + hyperparameter tuning)
        |
        v
Extracted Text → outputs/extracted_text.txt
```

---

## 3. How to Use It

### Webcam Scanner (original mode — built 2023)
```bash
python main.py
```
Press `s` to capture and scan. Press `q` to quit.

### Webcam Scanner in a different language
```bash
python main.py --mode scan --lang hin    # Hindi
python main.py --mode scan --lang tel    # Telugu
python main.py --mode scan --lang tam    # Tamil
```

### Process a single saved image
```bash
python main.py --mode image --file samples/document.jpg
```

### Process all images in a folder
```bash
python main.py --mode batch
python main.py --mode batch --lang tel
python main.py --mode batch --multilingual   # auto-detect language per image
```

### Visualize the 7 preprocessing stages
```bash
python main.py --mode visualize --file samples/document.jpg
```

### List installed Tesseract language packs
```bash
python main.py --mode languages
```

---

## 4. The 7-Stage Preprocessing Pipeline

This pipeline was the core contribution of the original 2023 project. Each stage solves a specific problem with real-world document photos.

### Stage 1 — Rotation Correction
Rotates the image 90 degrees clockwise. Most document photos taken with a phone or webcam come in landscape orientation. Tesseract expects text to run left-to-right — without this correction, it reads columns instead of rows and produces garbage output.

### Stage 2 — Color Inversion
Inverts all pixel values using bitwise NOT. Dark text on white background becomes white text on dark background. This improves contrast for documents with faded ink, low-quality printing, or yellowed paper. The subsequent binarization step works more reliably on inverted images.

### Stage 3 — Grayscale Conversion
Converts the BGR 3-channel image to a single grayscale channel. Required before thresholding — Otsu's method only operates on single-channel images. Also reduces computation for all subsequent steps.

### Stage 4 — Otsu's Adaptive Binarization
Converts the grayscale image to pure black and white using Otsu's thresholding algorithm.

Why Otsu's method instead of a fixed threshold value?

Otsu's automatically calculates the optimal threshold by analyzing the histogram of pixel intensities and finding the value that minimizes within-class variance. A fixed threshold of 128 works on clean studio photos but fails on real documents photographed under uneven lighting — a lamp casting a shadow on one side of the document would make half the text invisible with a fixed threshold.

### Stage 5 — Morphological Noise Removal
Applies three morphological operations in sequence to remove small artifacts:

Dilation expands white pixel regions, filling small gaps in broken text strokes.
Erosion shrinks white pixel regions, removing thin isolated noise pixels.
Morphological closing (dilation followed by erosion) closes small holes inside letter shapes.

The kernel size (1, 1) is intentionally minimal — large kernels would distort the actual character shapes.

### Stage 6 — Median Blur Smoothing
Applies a 3×3 median blur filter. Replaces each pixel with the median value of its neighborhood. Salt-and-pepper noise pixels (isolated extreme values from paper texture or compression artifacts) get replaced by surrounding normal values. Text edges, which are consistent across multiple pixels, are preserved.

### Stage 7 — Contour-Based Border Crop
Detects contours in the image, sorts by area, and crops to the bounding rectangle of the smallest external contour. This removes dark borders, table frame lines, and edge shadows that would otherwise cause Tesseract to attempt reading non-text regions.

---

## 5. CNN Architecture

A convolutional neural network provides a second layer of character recognition confidence on top of Tesseract.

### Architecture

```
Input: (32, 128, 1) — grayscale character crop

Conv2D(32, 3×3, relu) → BatchNorm → MaxPool(2×2)
Conv2D(64, 3×3, relu) → BatchNorm → MaxPool(2×2) → Dropout(0.3)
Conv2D(128, 3×3, relu) → BatchNorm → MaxPool(2×2) → Dropout(0.3)
Flatten
Dense(256, relu) → Dropout(0.3)
Dense(NUM_CLASSES, softmax)

Output: probability distribution over 67 characters
```

### Why This Architecture

Three convolutional blocks with increasing filter counts (32 → 64 → 128) capture progressively complex visual features:
- Block 1 detects low-level edges and stroke directions
- Block 2 detects character stroke patterns and curve shapes
- Block 3 detects higher-level structural features specific to characters

BatchNormalization after each block stabilizes training and allows higher learning rates. Dropout at 30% prevents overfitting to specific fonts in the training data.

### Data Augmentation

89% accuracy was achieved through data augmentation during training:

| Augmentation | Range | Purpose |
|---|---|---|
| Rotation | ±10 degrees | Handle slightly tilted characters |
| Width shift | ±10% | Handle horizontal positioning variation |
| Height shift | ±10% | Handle vertical positioning variation |
| Zoom | ±10% | Handle different character sizes |
| Shear | ±10% | Handle perspective distortion |
| Brightness | 0.8× to 1.2× | Handle lighting variation |

Without augmentation, the model overfit to training fonts and accuracy dropped to approximately 71% on held-out test documents.

---

## 6. Multilingual Support

The pipeline supports 7 languages using Tesseract language packs:

| Language | Code | Install Status |
|---|---|---|
| English | eng | Default — always installed |
| Hindi | hin | Requires separate install |
| Telugu | tel | Requires separate install |
| Tamil | tam | Requires separate install |
| French | fra | Requires separate install |
| German | deu | Requires separate install |
| Spanish | spa | Requires separate install |

### Installing Additional Languages (Windows)

Download `.traineddata` files from the official Tesseract repository:
```
https://github.com/tesseract-ocr/tessdata
```

Place the downloaded files in:
```
C:\Program Files\Tesseract-OCR\tessdata\
```

For example, for Hindi:
```
Download: hin.traineddata
Place in: C:\Program Files\Tesseract-OCR\tessdata\hin.traineddata
```

### Auto Language Detection

The `--multilingual` flag tries all installed language packs on each image and returns the result with the highest average word confidence score. Tesseract also supports combined language codes for mixed-language documents:

```python
# Detect English and Hindi in the same document
text = extract_text(image, language="eng+hin")
```

---

## 7. Results

| Metric | Value |
|---|---|
| OCR accuracy (English printed text) | 89% |
| Preprocessing improvement | ~40% better than no preprocessing |
| Supported languages | 7 |
| Anomaly types handled | Low light, shadow, rotation, noise, border artifacts |
| Processing time per image | ~2 seconds on standard hardware |
| Unit tests | 28 passing |

### What "89% accuracy" means here

The accuracy figure represents character-level recognition accuracy on a test set of real document photos including printed forms, book pages, and handwritten notes. The preprocessing pipeline accounts for approximately 40 percentage points of that improvement — running Tesseract directly on unprocessed photos produced accuracy in the 45-55% range on the same test set.

---

## 8. Project Structure

```
multilingual-ocr-scanner/
|
|-- src/
|   |-- __init__.py
|   |-- config.py              all constants (paths, languages, model params)
|   |-- preprocessor.py        7-stage preprocessing pipeline
|   |-- ocr_engine.py          Tesseract OCR with multilingual support
|   |-- cnn_model.py           CNN architecture and training code
|   |-- scanner.py             original 2023 webcam scanner script
|   |-- batch_processor.py     process multiple images at once
|   `-- visualizer.py          show preprocessing stages as matplotlib figure
|
|-- tests/
|   |-- __init__.py
|   |-- test_preprocessor.py   17 tests for preprocessing pipeline
|   `-- test_ocr_engine.py     11 tests for OCR engine
|
|-- samples/                   place your document images here
|   `-- README.md
|
|-- outputs/                   all generated files (gitignored)
|   |-- ScannedImage.jpg
|   |-- ProcessedImage.jpg
|   |-- extracted_text.txt
|   `-- pipeline_stages.png
|
|-- .github/workflows/tests.yml  CI: runs pytest on every push
|-- main.py                      entry point
|-- requirements.txt
|-- .gitignore
`-- README.md
```

---

## 9. Setup and Installation

### Requirements

- Python 3.9 or higher
- Tesseract OCR installed on your system
- A webcam (for scan mode only)

### Install Tesseract on Windows

Download and install from:
```
https://github.com/UB-Mannheim/tesseract/wiki
```

Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

If you install to a different location, update `TESSERACT_CMD` in `src/config.py`.

### Install Python Dependencies

```bash
git clone https://github.com/dharanibhumireddy01/Multilingual-OCR-Scanner.git
cd multilingual-ocr-scanner
pip install -r requirements.txt
```

### TensorFlow (for CNN mode — optional)

```bash
pip install tensorflow
```

If TensorFlow is not installed, the pipeline runs in Tesseract-only mode automatically. The CNN is an optional enhancement layer.

### Verify Installation

```bash
python main.py --mode languages
```

If Tesseract is installed correctly, this prints the list of available language packs.

---

## 10. Running the Tests

```bash
pytest tests/ -v
```

### Test Coverage

**test_preprocessor.py — 17 tests**

| Test | What it checks |
|---|---|
| test_rotate_changes_dimensions | Height and width are swapped after rotation |
| test_rotate_preserves_channels | Channel count unchanged after rotation |
| test_rotate_not_same_as_original | Rotated image differs from original |
| test_invert_colors_reverses_values | White becomes black, black becomes white |
| test_invert_twice_equals_original | Double inversion restores original |
| test_grayscale_reduces_to_one_channel | 3 channels reduced to 1 |
| test_grayscale_preserves_height_width | Spatial dimensions unchanged |
| test_binarize_produces_binary_image | Only 0 and 255 values in output |
| test_binarize_same_shape | Shape unchanged after binarization |
| test_remove_noise_same_shape | Shape unchanged after noise removal |
| test_remove_noise_stays_binary | Values remain 0 and 255 only |
| test_smooth_same_shape | Shape unchanged after median blur |
| test_crop_borders_returns_array | Returns numpy array |
| test_crop_borders_not_larger_than_input | Crop is same size or smaller |
| test_preprocess_returns_2d_array | Full pipeline returns 2D array |
| test_preprocess_output_is_binary | Full pipeline output is binary |
| test_preprocess_no_nan | No NaN values in output |

**test_ocr_engine.py — 11 tests**

| Test | What it checks |
|---|---|
| test_clean_text_strips_whitespace | Leading/trailing whitespace removed |
| test_clean_text_collapses_blank_lines | Multiple blank lines collapsed |
| test_clean_text_empty_string | Empty input returns empty string |
| test_clean_text_none_equivalent | Whitespace-only input returns empty |
| test_clean_text_preserves_content | Real content not removed |
| test_clean_text_handles_single_line | Single line preserved correctly |
| test_get_available_languages_returns_list | Language list is a list type |
| test_extract_text_returns_string | Extraction returns string type |
| test_extract_text_cleans_output | Output is stripped of whitespace |
| test_extract_text_fallback_on_bad_language | Invalid language does not crash |
| test_extract_with_confidence_returns_list | Confidence extraction returns list |

---

## 11. Configuration

All constants are in `src/config.py`. Key settings:

**Camera index** — if your webcam is not detected:
```python
CAMERA_INDEX = 0    # 0 = built-in laptop camera, 1 = external USB webcam
```

**Tesseract path** — if installed to a non-default location:
```python
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**Default language:**
```python
DEFAULT_LANGUAGE = "eng"    # change to "tel" for Telugu, "hin" for Hindi
```

**Preprocessing parameters:**
```python
MORPH_KERNEL_SIZE = (1, 1)   # increase slightly for very noisy images
MEDIAN_BLUR_KERNEL = 3       # must be odd number
```

---

## 12. Real-World Applications

### Document Digitization
Convert printed forms, invoices, receipts, and certificates into searchable digital text. The preprocessing pipeline handles the real-world variation in document quality that makes standard OCR fail.

### Multilingual Government Documents
India's official documents often contain both English and regional language text (Hindi, Telugu, Tamil). The multilingual support and auto-detection handle this directly.

### Healthcare Record Processing
Handwritten prescriptions and patient intake forms can be partially digitized using the preprocessing pipeline to enhance legibility before OCR extraction.

### Data Entry Automation
Batch processing mode allows organizations to process hundreds of scanned documents automatically, replacing manual data entry workflows.

### Educational Technology
Students can photograph printed study materials and convert them to searchable, copyable text for note-taking and revision.

---

## 13. Technical Stack

| Component | Technology | Role |
|---|---|---|
| Image capture | OpenCV VideoCapture | Webcam access and frame capture |
| Preprocessing | OpenCV (cv2) | All 7 pipeline stages |
| OCR engine | Tesseract + pytesseract | Text extraction |
| Deep learning | TensorFlow / Keras | CNN character recognition |
| Visualization | Matplotlib | Pipeline stage display |
| Array operations | NumPy | Image array manipulation |
| Testing | Pytest | 28 unit tests |
| CI/CD | GitHub Actions | Automated test run on push |

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*Originally developed as a B.Tech ECE final year project (2023) at SCSVMV University.*
*Extended with CNN support, multilingual capability, and production-grade structure as part of MS Data Science portfolio work at University at Albany, SUNY (2026).*

*Author: Dharani Bhumireddy*
