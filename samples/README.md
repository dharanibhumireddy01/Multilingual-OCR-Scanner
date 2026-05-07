# samples/

Place your document images here to use with batch mode or single image mode.

## Supported formats
.jpg, .jpeg, .png, .bmp, .tiff

## Usage
```bash
# Process all images in this folder
python main.py --mode batch

# Process a single image
python main.py --mode image --file samples/your_document.jpg

# Visualize preprocessing stages on an image
python main.py --mode visualize --file samples/your_document.jpg
```

## Tips for best OCR accuracy
- Use good lighting — avoid shadows on the document
- Hold the camera or phone directly above the document
- Make sure the document fills most of the frame
- Higher resolution images produce better results
- Printed text works better than handwriting
