#!/usr/bin/env python3
"""
Manga Translation Example with YOLO Speech Bubble Detection

This script demonstrates how to use the manga translation pipeline
with YOLOv8 for speech bubble detection and Azure OCR for text recognition.
"""

import os
import argparse
from dotenv import load_dotenv

from translation_pipeline import MangaTranslationPipeline, batch_translate_manga


def load_azure_credentials():
    """Load Azure credentials from environment variables"""
    load_dotenv()
    return {
        "azure_key": os.getenv("AZURE_VISION_KEY"),
        "azure_endpoint": os.getenv("AZURE_VISION_ENDPOINT")
    }


def process_manga_page(
    input_path,
    output_path=None,
    yolo_model_path=None,
    placeholder_mode=False,
    visualize_bubbles=False,
    translator="deepl",
    target_language='en',
    deepl_api_key=None,
    debug=False
):
    """
    Process a single manga page with the YOLO-enhanced pipeline.
    
    Args:
        input_path: Path to the manga image
        output_path: Path for the output image
        yolo_model_path: Path to YOLOv8 model (optional)
        placeholder_mode: Whether to use placeholder text
        visualize_bubbles: Whether to visualize detected bubbles
        translator: Translator to use (default: "deepl")
        target_language: Target language code for DeepL
        deepl_api_key: DeepL API key (if using DeepL translator)
        debug: Enable debug output
    """
    # Set default output path if not provided
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_translated{ext}"
    
    # Load Azure credentials
    azure_creds = load_azure_credentials()
    
    # Create the pipeline with YOLO
    pipeline = MangaTranslationPipeline(
        azure_key=azure_creds["azure_key"],
        azure_endpoint=azure_creds["azure_endpoint"],
        yolo_model_path=yolo_model_path,
        placeholder_mode=placeholder_mode,
        visualize_bubbles=visualize_bubbles,
        translator=translator,
        target_language=target_language,
        deepl_api_key=deepl_api_key,
        debug=debug
    )
    
    # Process the manga page
    print(f"Processing manga page: {input_path}")
    _, info = pipeline.process(input_path, output_path)
    
    # Generate translation template if in placeholder mode
    if placeholder_mode:
        template_path = f"{os.path.splitext(output_path)[0]}_template.txt"
        pipeline.generate_translation_template(template_path)
        print(f"Generated translation template: {template_path}")
    
    # Print a summary
    print(f"\nProcessing complete in {info['total_time']:.2f} seconds")
    print(f"Found {info['steps']['ocr']['detected_boxes']} text boxes")
    print(f"  - {info['steps']['ocr']['bubble_text_count']} boxes inside speech bubbles")
    
    if info['steps']['replacement']['translated_boxes'] > 0:
        print(f"Translated {info['steps']['replacement']['translated_boxes']} text boxes")
    
    print(f"Output saved to: {output_path}")
    
    return info


def main():
    """Main function for the command-line interface."""
    parser = argparse.ArgumentParser(description="Manga Translation with YOLO")
    
    parser.add_argument("input", help="Input manga image or directory")
    parser.add_argument("-o", "--output", help="Output path (image or directory)")
    parser.add_argument("-y", "--yolo", help="Path to YOLOv8 model file")
    parser.add_argument("-b", "--batch", action="store_true", help="Process a directory of images")
    parser.add_argument("-p", "--placeholder", action="store_true", help="Use placeholder text")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize detected bubbles")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output")
    parser.add_argument("-l", "--language", help="Target language code")
    parser.add_argument("-t", "--translator", help="Translator to use")
    args = parser.parse_args()
    
    # Load Azure credentials
    azure_creds = load_azure_credentials()
    
    if args.batch or os.path.isdir(args.input):
        # Process a directory of manga pages
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            return
        
        output_dir = args.output or f"{args.input}_translated"
        
        batch_translate_manga(
            input_dir=args.input,
            output_dir=output_dir,
            azure_key=azure_creds["azure_key"],
            azure_endpoint=azure_creds["azure_endpoint"],
            yolo_model_path=args.yolo,
            placeholder_mode=args.placeholder,
            visualize_bubbles=args.visualize,
            translator=args.translator,
            target_language=args.language,
            deepl_api_key=os.getenv("DEEPL_API_KEY"),
            debug=args.debug
        )
    else:
        # Process a single manga page
        if not os.path.isfile(args.input):
            print(f"Error: {args.input} is not a file")
            return
        
        process_manga_page(
            input_path=args.input,
            output_path=args.output,
            yolo_model_path=args.yolo,
            placeholder_mode=args.placeholder,
            visualize_bubbles=args.visualize,
            translator=args.translator,
            target_language=args.language,
            deepl_api_key=os.getenv("DEEPL_API_KEY"),
            debug=args.debug
        )


if __name__ == "__main__":
    main()