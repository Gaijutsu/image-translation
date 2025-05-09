# Manga Translation Pipeline

A specialized Python library for extracting and replacing text in manga/comic images for translation purposes. This pipeline detects text in speech bubbles and text boxes, replaces it with translations, and renders the text back onto the original image while maintaining proper layout and styling.

## Features

- **Specialized Manga OCR**: Uses manga-specific OCR techniques to detect Japanese text
- **Speech Bubble Detection**: Automatically identifies speech bubbles and text boxes
- **Vertical Text Support**: Properly handles vertical text common in Japanese manga
- **Bubble-Aware Rendering**: Ensures text fits properly within speech bubbles
- **Translation Management**: Generate templates and load translations from files
- **Font Styling**: Uses manga-appropriate fonts with proper sizing and positioning
- **Multiple OCR Options**: Works with specialized manga-ocr or Azure Computer Vision
- **Batch Processing**: Translate entire manga chapters at once

## Requirements

- Python 3.6+
- Required Python packages:
  - Pillow (PIL)
  - OpenCV (cv2)
  - NumPy
  - python-dotenv (for configuration)
  
- Optional dependencies:
  - Azure Computer Vision SDK (for improved OCR)
  - manga-ocr (for specialized manga text detection)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/manga-translation-pipeline.git
   cd manga-translation-pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install optional dependencies for better results:
   ```bash
   pip install azure-cognitiveservices-vision-computervision manga-ocr
   ```

4. Set up your Azure Computer Vision service (recommended):
   - Create a Computer Vision resource in the Azure portal
   - Get your API key and endpoint URL
   - Create a `.env` file with your credentials

## Usage

### Basic Usage

The pipeline can be used in three main modes:

1. **Translation Mode**: Replace text with actual translations
2. **Placeholder Mode**: Replace text with placeholder characters (for template creation)
3. **Extraction Mode**: Just extract text and generate translation templates

### Command Line Interface

```bash
# Process a single manga page with translations
python manga_example_usage.py process input.jpg -o output.jpg -t translations.txt

# Create a translation template from a manga page
python manga_example_usage.py template input.jpg -o template.txt

# Process a directory of manga pages
python manga_example_usage.py batch manga_chapter/ -o translated_chapter/ -t translations.txt

# Use placeholder text for debugging/preview
python manga_example_usage.py process input.jpg -o output.jpg -p
```

### Translation Template Format

The translation template is a simple text file with the format:
```
original_japanese_text|translated_text
# Comment line with position info
another_text|another_translation
```

### Programmatic Usage

```python
from manga_translation_pipeline import MangaTranslationPipeline

# Create the pipeline
pipeline = MangaTranslationPipeline(
    azure_key="your_azure_key",
    azure_endpoint="your_azure_endpoint",
    debug=True
)

# Load translations
pipeline.load_translations("translations.txt")

# Process a manga page
pipeline.process("manga_page.jpg", "translated_page.jpg")

# Generate a translation template
pipeline.generate_translation_template("template.txt")
```

## Workflow for Manga Translation

1. **Template Creation**:
   ```bash
   python manga_example_usage.py batch manga_chapter/ --templates
   ```
   This will create template files for each page in the chapter.

2. **Translation**:
   Fill in the translations in the template files.

3. **Batch Processing**:
   ```bash
   python manga_example_usage.py batch manga_chapter/ -o translated_chapter/ -t translations.txt
   ```
   This will process all pages with your translations.

## Advanced Options

### Manga-OCR Integration

The system can use the specialized `manga-ocr` library for better detection of Japanese text:

```python
pipeline = MangaTranslationPipeline(
    manga_ocr_available=True  # Enable manga-ocr if installed
)
```

### Custom Fonts

You can specify custom fonts for horizontal and vertical text:

```python
pipeline = MangaTranslationPipeline(
    font_path="path/to/manga_font.ttf",
    vertical_font_path="path/to/vertical_font.ttf"
)
```
## Acknowledgments

- Uses Azure Computer Vision or manga-ocr for text detection
- Inspired by various manga fan translation techniques