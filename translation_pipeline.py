from typing import List, Optional, Dict, Any, Union, Tuple
import os
import time
import json

from PIL import Image

from textbox import TextBox
from ocr_processor import MangaOCRProcessor
from text_replacer import MangaTextReplacer
from renderer import MangaTextRenderer


class MangaTranslationPipeline:
    """
    Main pipeline for manga text replacement and translation.
    
    This class orchestrates the process of detecting text in manga images,
    replacing with translations, and rendering the translated text back
    onto the original images using YOLO for bubble detection.
    """
    
    def __init__(self, 
                azure_key: Optional[str] = None,
                azure_endpoint: Optional[str] = None,
                yolo_model_path: Optional[str] = None,
                font_path: Optional[str] = None,
                placeholder_mode: bool = False,
                detect_bubbles: bool = True,
                language: str = "ja",
                debug: bool = False,
                visualize_bubbles: bool = False,
                translator: str = "deepl",
                target_language: str = 'en',
                deepl_api_key: str = None):
        """
        Initialize the manga translation pipeline.
        
        Args:
            azure_key: Azure Computer Vision API key (if available)
            azure_endpoint: Azure Computer Vision endpoint (if available)
            yolo_model_path: Path to YOLOv8 bubble detection model (optional)
            font_path: Path to font file for text rendering
            placeholder_mode: Whether to use placeholder text instead of translations
            detect_bubbles: Whether to detect speech bubbles
            language: Source language (default: "ja" for Japanese)
            debug: Enable debug mode for more verbose output
            visualize_bubbles: Whether to create visualization of detected bubbles
            translator: Translator to use (default: "deepl")
            target_language: Target language code for DeepL
            deepl_api_key: DeepL API key (if using DeepL translator)
        """
        self.debug = debug
        self.visualize_bubbles = visualize_bubbles
        
        # Initialize components
        self.ocr_processor = MangaOCRProcessor(
            azure_key=azure_key,
            azure_endpoint=azure_endpoint,
            yolo_model_path=yolo_model_path,
            detect_bubbles=detect_bubbles,
            language=language
        )
        
        self.text_replacer = MangaTextReplacer(
            placeholder_mode=placeholder_mode,
            translator=translator,
            target_language=target_language,
            deepl_api_key=deepl_api_key
        )
        
        self.renderer = MangaTextRenderer(
            font_path=font_path,
        )
        
        # Storage for processing results
        self.original_text_boxes: List[TextBox] = []
        self.modified_text_boxes: List[TextBox] = []
    
    def process(self, 
               image_path: str, 
               output_path: Optional[str] = None) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Process a manga image for translation.
        
        Args:
            image_path: Path to the input manga image
            output_path: Path to save the translated image (if None, won't save)
            
        Returns:
            Tuple containing:
            - PIL Image with translated text
            - Dict with process information (timing, statistics)
        """
        info = {
            "start_time": time.time(),
            "input_image": image_path,
            "output_image": output_path,
            "steps": {}
        }
        
        try:
            # Step 1: Extract text and bubbles using OCR with YOLO
            if self.debug:
                print(f"[1/3] Extracting text and bubbles from {image_path}...")
                
            step_start = time.time()
            self.original_text_boxes = self.ocr_processor.process_image(
                image_path, 
                visualize_bubbles=self.visualize_bubbles
            )
            
            # Count how many text boxes are in bubbles
            bubble_text_count = sum(1 for box in self.original_text_boxes 
                                  if hasattr(box, "bubble_id") and box.bubble_id)
            
            info["steps"]["ocr"] = {
                "time": time.time() - step_start,
                "detected_boxes": len(self.original_text_boxes),
                "bubble_text_count": bubble_text_count
            }
            
            if self.debug:
                print(f"      Found {len(self.original_text_boxes)} text boxes")
                print(f"      {bubble_text_count} text boxes are inside bubbles")
                
                for box in self.original_text_boxes:
                    bubble_info = getattr(box, "bubble_id", "no bubble")
                    print(f"      - '{box.original_text}' [{bubble_info}]")
            
            # Step 2: Apply text replacement/translation
            if self.debug:
                print("[2/3] Applying text replacement/translation...")
            
            # Drop all text boxes that are not in bubbles
            self.original_text_boxes = [box for box in self.original_text_boxes 
                                        if hasattr(box, "bubble_id") and box.bubble_id]
                
            step_start = time.time()
            self.modified_text_boxes = self.text_replacer.replace_text(self.original_text_boxes)
            
            info["steps"]["replacement"] = {
                "time": time.time() - step_start,
                "translated_boxes": sum(1 for i, box in enumerate(self.modified_text_boxes) 
                                      if box.text != self.original_text_boxes[i].original_text)
            }
            
            if self.debug:
                for i, box in enumerate(self.modified_text_boxes):
                    if box.text != self.original_text_boxes[i].original_text:
                        print(f"      - '{self.original_text_boxes[i].original_text}' → '{box.text}'")
            
            # Step 3: Render new text onto the original image
            if self.debug:
                print("[3/3] Rendering translated text...")
                
            step_start = time.time()
            result_image = self.renderer.render_image(
                image_path,
                self.original_text_boxes,
                self.modified_text_boxes,
                output_path
            )
            
            info["steps"]["rendering"] = {
                "time": time.time() - step_start
            }
            
            # Complete process information
            info["total_time"] = time.time() - info["start_time"]
            info["success"] = True
            
            if self.debug:
                print(f"Process completed in {info['total_time']:.2f} seconds")
                if output_path:
                    print(f"Result saved to {output_path}")
            
            return result_image, info
            
        except Exception as e:
            # Handle errors
            if self.debug:
                print(f"Error during processing: {str(e)}")
                
            info["success"] = False
            info["error"] = str(e)
            
            # Re-raise the exception
            raise
    
    def get_extracted_text(self) -> str:
        """
        Get all extracted text as a single string.
        
        Returns:
            String containing all detected text
        """
        if not self.original_text_boxes:
            return ""
            
        # Group text by bubbles
        bubble_texts = {}
        non_bubble_texts = []
        
        for box in self.original_text_boxes:
            bubble_id = getattr(box, "bubble_id", None)
            if bubble_id:
                if bubble_id in bubble_texts:
                    bubble_texts[bubble_id].append(box.original_text)
                else:
                    bubble_texts[bubble_id] = [box.original_text]
            else:
                non_bubble_texts.append(box.original_text)
        
        # Combine texts
        result = []
        
        # Add bubble texts
        for bubble_id, texts in bubble_texts.items():
            result.append(f"[Bubble {bubble_id}]")
            result.append("\n".join(texts))
            result.append("")  # Empty line
        
        # Add non-bubble texts
        if non_bubble_texts:
            result.append("[Other Text]")
            result.append("\n".join(non_bubble_texts))
        
        return "\n".join(result)
    
    def generate_translation_template(self, output_path: str) -> None:
        """
        Generate a translation template file from detected text.
        
        Args:
            output_path: Path to save the translation template
        """
        if not self.original_text_boxes:
            raise ValueError("No text boxes available. Process an image first.")
        
        self.text_replacer.generate_translation_template(self.original_text_boxes, output_path)
        
        if self.debug:
            print(f"Translation template saved to {output_path}")
    
    def load_translations(self, 
                        translations_path: Optional[str] = None, 
                        translations_dict: Optional[Dict[str, str]] = None) -> None:
        """
        Load translations from a file or dictionary.
        
        Args:
            translations_path: Path to translations file
            translations_dict: Dictionary of translations
        """
        if translations_path and os.path.exists(translations_path):
            self.text_replacer.load_translations_from_file(translations_path)
            
            if self.debug:
                print(f"Loaded translations from {translations_path}")
                print(f"  {len(self.text_replacer.translations)} translation entries")
        
        elif translations_dict:
            self.text_replacer.load_translations_from_dict(translations_dict)
            
            if self.debug:
                print(f"Loaded {len(translations_dict)} translations from dictionary")
    
    def set_placeholder_mode(self, enabled: bool, char: str = 'a') -> None:
        """
        Set whether to use placeholder text instead of translations.
        
        Args:
            enabled: Whether to enable placeholder mode
            char: Character to use for placeholder text
        """
        self.text_replacer.set_placeholder_mode(enabled, char)


def batch_translate_manga(
    input_dir: str,
    output_dir: str,
    translations_path: Optional[str] = None,
    azure_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    yolo_model_path: Optional[str] = None,
    placeholder_mode: bool = False,
    visualize_bubbles: bool = False,
    translator: str = "deepl",
    target_language: str = 'en',
    deepl_api_key: str = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Batch process multiple manga images for translation.
    
    Args:
        input_dir: Directory containing manga images
        output_dir: Directory to save translated images
        translations_path: Path to translations file
        azure_key: Azure Computer Vision API key
        azure_endpoint: Azure Computer Vision endpoint
        yolo_model_path: Path to YOLOv8 bubble detection model
        placeholder_mode: Whether to use placeholder text
        visualize_bubbles: Whether to create visualization of detected bubbles
        translator: Translator to use (default: "deepl")
        target_language: Target language code for DeepL
        deepl_api_key: DeepL API key (if using DeepL translator)
        debug: Enable debug output
        
    Returns:
        Dictionary with batch processing information
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = [
        f for f in os.listdir(input_dir) 
        if os.path.isfile(os.path.join(input_dir, f)) and 
        f.lower().endswith(image_extensions)
    ]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return {"success": False, "error": "No images found"}
    
    print(f"Found {len(image_files)} manga images to process")
    
    # Create pipeline with YOLO
    pipeline = MangaTranslationPipeline(
        azure_key=azure_key,
        azure_endpoint=azure_endpoint,
        yolo_model_path=yolo_model_path,
        placeholder_mode=placeholder_mode,
        visualize_bubbles=visualize_bubbles,
        translator=translator,
        target_language=target_language,
        deepl_api_key=deepl_api_key,
        debug=debug
    )
    
    # Load translations if available
    if translations_path and os.path.exists(translations_path):
        pipeline.load_translations(translations_path)
    
    # Process each image
    results = {
        "total": len(image_files),
        "successful": 0,
        "failed": 0,
        "images": []
    }
    
    for i, image_file in enumerate(image_files):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        
        print(f"[{i+1}/{len(image_files)}] Processing {image_file}...")
        
        try:
            # Process image
            _, info = pipeline.process(input_path, output_path)
            
            # Generate translation template if in placeholder mode
            if placeholder_mode:
                template_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_template.txt")
                pipeline.generate_translation_template(template_path)
            
            results["successful"] += 1
            results["images"].append({
                "file": image_file,
                "status": "success",
                "text_boxes": info["steps"]["ocr"]["detected_boxes"],
                "bubble_text_count": info["steps"]["ocr"]["bubble_text_count"],
                "processing_time": info["total_time"]
            })
            
            print(f"  Completed in {info['total_time']:.2f} seconds")
            print(f"  Found {info['steps']['ocr']['bubble_text_count']} text boxes in bubbles")
            
        except Exception as e:
            results["failed"] += 1
            results["images"].append({
                "file": image_file,
                "status": "failed",
                "error": str(e)
            })
            
            print(f"  Error: {str(e)}")
    
    # Save batch results
    results_path = os.path.join(output_dir, "translation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nBatch processing complete: {results['successful']} successful, {results['failed']} failed")
    print(f"Results saved to {results_path}")
    
    return results