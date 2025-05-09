import os
import uuid
from typing import List, Dict, Any, Tuple, Optional, Union
import time
from io import BytesIO
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

# Azure imports
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

from textbox import TextBox
from speech_bubble_detector import YOLOSpeechBubbleDetector


class MangaOCRProcessor:
    """
    Specialized OCR processor for manga and comic images.
    
    This class detects text in manga panels and identifies text bubbles and boxes
    using a YOLOv8 model specialized for comic speech bubbles.
    """
    
    def __init__(self, 
                azure_key: Optional[str] = None,
                azure_endpoint: Optional[str] = None,
                yolo_model_path: Optional[str] = None,
                detect_bubbles: bool = True,
                language: str = "ja",
                bubble_confidence: float = 0.25):
        """
        Initialize the Manga OCR processor.
        
        Args:
            azure_key: Azure Computer Vision subscription key (optional)
            azure_endpoint: Azure Computer Vision endpoint URL (optional)
            yolo_model_path: Path to YOLOv8 speech bubble detection model (optional)
            detect_bubbles: Whether to detect speech bubbles
            language: Primary language for OCR (default: "ja" for Japanese)
            bubble_confidence: Confidence threshold for bubble detection
        """
        self.language = language
        self.detect_bubbles = detect_bubbles
        
        # Initialize Azure Vision
        self.use_azure = bool(azure_key and azure_endpoint)
        if self.use_azure:
            try:
                self.azure_client = ComputerVisionClient(
                    endpoint=azure_endpoint,
                    credentials=CognitiveServicesCredentials(azure_key)
                )
                print("Azure Computer Vision client initialized")
            except Exception as e:
                print(f"Failed to initialize Azure client: {e}")
                self.use_azure = False
        
        # Initialize YOLO for bubble detection
        if detect_bubbles:
            try:
                self.bubble_detector = YOLOSpeechBubbleDetector(
                    model_path=yolo_model_path,
                    confidence=bubble_confidence,
                    download_model=True
                )
            except Exception as e:
                print(f"Failed to initialize YOLO bubble detector: {e}")
                self.detect_bubbles = False
    
    def process_image(self, image_path: str, visualize_bubbles: bool = False) -> List[TextBox]:
        """
        Process a manga image to extract text and bubble information.
        
        Args:
            image_path: Path to the manga image file
            visualize_bubbles: Whether to create a visualization of detected bubbles
            
        Returns:
            List of TextBox objects with detected text and bubble info
        """
        # Ensure the image path exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Load image for processing
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to RGB for processing (OpenCV loads as BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect speech bubbles and text boxes if requested
        bubbles = []
        if self.detect_bubbles:
            bubbles = self.bubble_detector.detect_bubbles(image)
            precise_bubbles = self.bubble_detector.find_all_bubble_outlines(image, bubbles)
            print(f"Detected {len(bubbles)} speech bubbles")
            
            if visualize_bubbles:
                # Create visualization output path
                vis_path = os.path.join(
                    os.path.dirname(image_path),
                    f"{os.path.splitext(os.path.basename(image_path))[0]}_bubbles.jpg"
                )
                precise_vis_path = os.path.join(
                    os.path.dirname(image_path),
                    f"{os.path.splitext(os.path.basename(image_path))[0]}_precise_bubbles.jpg"
                )
                self.bubble_detector.visualize_bubbles(image, bubbles, vis_path)
                print(f"Bubble visualization saved to {vis_path}")
                self.bubble_detector.visualize_precise_bubbles(image, precise_bubbles, precise_vis_path)
                print(f"Precise bubble visualization saved to {precise_vis_path}")
        
        # Extract text using appropriate OCR method, caching results
        pickle_path = image_path + '.ocr.pkl'
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, 'rb') as f:
                    import pickle
                    text_boxes = pickle.load(f)
                    print(f"Loaded cached OCR results from {pickle_path}")
            except Exception as e:
                print(f"Failed to load cached OCR results: {e}")
                text_boxes = None
        else:
            if self.use_azure:
                text_boxes = self._extract_with_azure(image_path, bubbles)
            else:
                # Use fallback OCR method
                text_boxes = self._extract_with_tesseract(image_rgb, bubbles, image_path)

            # Cache the results
            try:
                with open(pickle_path, 'wb') as f:
                    import pickle
                    pickle.dump(text_boxes, f)
                    print(f"Cached OCR results to {pickle_path}")
            except Exception as e:
                print(f"Failed to cache OCR results: {e}")
        
        # Add bubble information to text boxes
        for text_box in text_boxes:
            self._associate_with_bubble(text_box, bubbles)
        
        # Concatenate text boxes that belong to the same bubble
        text_boxes = self._concatenate_bubble_text(text_boxes)

        # Draw the text boxes on the image
        if visualize_bubbles:
            # Create visualization output path
            vis_path = os.path.join(
                os.path.dirname(image_path),
                f"{os.path.splitext(os.path.basename(image_path))[0]}_text_boxes.jpg"
            )
            self.bubble_detector.visualize_text_boxes(image, text_boxes, vis_path)
            print(f"Text box visualization saved to {vis_path}")
        
        return text_boxes
    
    def _extract_with_azure(self, 
                          image_path: str, 
                          bubbles: List[Dict[str, Any]]) -> List[TextBox]:
        """
        Extract text using Azure's OCR for manga/comics.
        
        Args:
            image_path: Path to the source image
            bubbles: List of detected speech bubbles/text boxes
            
        Returns:
            List of TextBox objects
        """
        text_boxes = []
        
        try:
            # Read the image as binary data
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            # Call Azure's Read API
            read_response = self.azure_client.read_in_stream(
                image=BytesIO(image_data),
                language=self.language,
                raw=True
            )
            
            # Get operation location to retrieve results
            operation_location = read_response.headers["Operation-Location"]
            operation_id = operation_location.split("/")[-1]
            
            # Poll for result
            max_retry = 10
            retry_delay = 1  # seconds
            
            result = None
            for i in range(max_retry):
                read_result = self.azure_client.get_read_result(operation_id)
                if read_result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                    result = read_result
                    break
                time.sleep(retry_delay)
            
            if not result or result.status != OperationStatusCodes.succeeded:
                print(f"OCR operation failed or timed out: {result.status if result else 'No result'}")
                return text_boxes
            
            # Process the OCR results
            for page in result.analyze_result.read_results:
                paragraph_index = 0
                
                for line_index, line in enumerate(page.lines):
                    line_text = line.text
                    line_bbox = line.bounding_box  # Format: [x1, y1, x2, y2, x3, y3, x4, y4]
                    
                    # Calculate bounding box in (x, y, width, height) format
                    x_coordinates = [line_bbox[i] for i in range(0, len(line_bbox), 2)]
                    y_coordinates = [line_bbox[i] for i in range(1, len(line_bbox), 2)]
                    
                    x = min(x_coordinates)
                    y = min(y_coordinates)
                    width = max(x_coordinates) - x
                    height = max(y_coordinates) - y
                    
                    # Create text box
                    text_box = TextBox(
                        id=f"textbox-{str(uuid.uuid4())[:8]}",
                        text=line_text,
                        original_text=line_text,
                        confidence=0.9,  # Azure doesn't provide confidence per line
                        x=int(x),
                        y=int(y),
                        width=int(width),
                        height=int(height),
                        paragraph_index=paragraph_index,
                        line_index=line_index,
                        word_index=0,
                        source_image_path=image_path
                    )
                    
                    text_boxes.append(text_box)
                
                paragraph_index += 1
        
        except Exception as e:
            print(f"Error in Azure OCR processing: {e}")
        
        return text_boxes
    
    def _extract_with_tesseract(self, 
                             image: np.ndarray, 
                             bubbles: List[Dict[str, Any]], 
                             image_path: str) -> List[TextBox]:
        """
        Extract text using Tesseract OCR as fallback.
        
        Args:
            image: OpenCV image (numpy array)
            bubbles: List of detected speech bubbles/text boxes
            image_path: Path to the source image
            
        Returns:
            List of TextBox objects
        """
        text_boxes = []
        
        try:
            # Import pytesseract here to avoid dependency if not needed
            import pytesseract
            
            # Set the OCR language based on the image language
            lang_code = "jpn" if self.language == "ja" else "eng"
            
            if not bubbles:
                # Process the whole image if no bubbles detected
                ocr_result = pytesseract.image_to_data(
                    image, 
                    lang=lang_code,
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6'  # Assume a single block of text
                )
                
                # Create text boxes from OCR results
                for i in range(len(ocr_result["text"])):
                    if not ocr_result["text"][i].strip():
                        continue
                    
                    text = ocr_result["text"][i]
                    x = ocr_result["left"][i]
                    y = ocr_result["top"][i]
                    width = ocr_result["width"][i]
                    height = ocr_result["height"][i]
                    conf = float(ocr_result["conf"][i])
                    
                    # Skip low confidence results
                    if conf < 30:
                        continue
                    
                    text_box = TextBox(
                        id=f"textbox-basic-{i}",
                        text=text,
                        original_text=text,
                        confidence=conf/100.0,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        paragraph_index=0,
                        line_index=i,
                        word_index=0,
                        source_image_path=image_path
                    )
                    text_boxes.append(text_box)
            else:
                # Process each bubble individually
                for i, bubble in enumerate(bubbles):
                    x, y, w, h = bubble["bounds"]
                    
                    # Ensure coordinates are within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, image.shape[1] - x)
                    h = min(h, image.shape[0] - y)
                    
                    if w <= 0 or h <= 0:
                        continue
                    
                    # Crop bubble region
                    bubble_img = image[y:y+h, x:x+w]
                    
                    # OCR on bubble
                    bubble_text = pytesseract.image_to_string(
                        bubble_img,
                        lang=lang_code,
                        config='--psm 6'
                    ).strip()
                    
                    if bubble_text:
                        text_box = TextBox(
                            id=f"textbox-bubble-{i}",
                            text=bubble_text,
                            original_text=bubble_text,
                            confidence=0.6,  # Default confidence
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                            paragraph_index=i,
                            line_index=0,
                            word_index=0,
                            source_image_path=image_path
                        )
                        text_boxes.append(text_box)
        
        except Exception as e:
            print(f"Error in Tesseract OCR processing: {e}")
        
        return text_boxes
    
    def _associate_with_bubble(self, 
                             text_box: TextBox, 
                             bubbles: List[Dict[str, Any]]) -> None:
        """
        Associate a text box with a speech bubble based on position.
        
        Args:
            text_box: TextBox to associate
            bubbles: List of detected speech bubbles
        """
        # Skip if no bubbles
        if not bubbles:
            return
        
        # Calculate text box center
        text_center_x = text_box.x + text_box.width / 2
        text_center_y = text_box.y + text_box.height / 2
        
        # Calculate text box area
        text_area = text_box.width * text_box.height
        
        # Find which bubble contains this text
        best_bubble = None
        best_overlap_ratio = 0
        
        for bubble in bubbles:
            bx, by, bw, bh = bubble["bounds"]
            
            # Check if text center is inside bubble
            if (bx <= text_center_x <= bx + bw and 
                by <= text_center_y <= by + bh):
                
                # Calculate overlap area
                x_overlap = min(text_box.x + text_box.width, bx + bw) - max(text_box.x, bx)
                y_overlap = min(text_box.y + text_box.height, by + bh) - max(text_box.y, by)
                
                if x_overlap > 0 and y_overlap > 0:
                    overlap_area = x_overlap * y_overlap
                    overlap_ratio = overlap_area / text_area
                    
                    # Keep track of best matching bubble (most overlap)
                    if overlap_ratio > best_overlap_ratio:
                        best_overlap_ratio = overlap_ratio
                        best_bubble = bubble
        
        # Assign the best matching bubble
        if best_bubble and best_overlap_ratio > 0.5:  # At least 50% overlap
            # Store bubble information in custom attributes
            text_box.bubble_id = best_bubble["id"]
            text_box.bubble_bounds = best_bubble["bounds"]
            text_box.bubble_confidence = best_bubble.get("confidence", 1.0)

    def _concatenate_bubble_text(self, text_boxes: List[TextBox]) -> List[TextBox]:
        """
        Concatenate all text boxes from the same bubble into a single text box.
        
        Args:
            text_boxes: List of TextBox objects with bubble associations
            
        Returns:
            List of concatenated TextBox objects, one per bubble
        """
        # Group text boxes by bubble ID
        bubble_text_boxes = defaultdict(list)
        non_bubble_text_boxes = []
        
        for text_box in text_boxes:
            if hasattr(text_box, 'bubble_id') and text_box.bubble_id:
                bubble_text_boxes[text_box.bubble_id].append(text_box)
            else:
                non_bubble_text_boxes.append(text_box)
        
        # Create new list for the concatenated results
        result_text_boxes = []
        
        # Process each bubble's text boxes
        for bubble_id, boxes in bubble_text_boxes.items():
            if not boxes:
                continue
                
            # Sort text boxes by reading order (top to bottom, left to right)
            if self.language == "ja":  # For Japanese, traditionally read top to bottom, right to left
                boxes.sort(key=lambda box: (box.x, box.y), reverse=True)  # Right to left primary, top to bottom secondary
            else:  # For most other languages (left to right, top to bottom)
                boxes.sort(key=lambda box: (box.y, box.x))  # Top to bottom primary, left to right secondary
            
            # Concatenate the text
            concatenated_text = ""
            concatenated_original_text = ""
            
            for box in boxes:
                concatenated_text += box.text
                concatenated_original_text += box.original_text
            
            # Calculate bounds that encompass all boxes in this bubble
            min_x = min(box.x for box in boxes)
            min_y = min(box.y for box in boxes)
            max_x = max(box.x + box.width for box in boxes)
            max_y = max(box.y + box.height for box in boxes)

            original_box_bounds = [(box.x, box.y, box.width, box.height) for box in boxes]
            
            # Create a new text box for the bubble
            bubble_box = TextBox(
                id=f"bubble-{bubble_id}",
                text=concatenated_text,
                original_text=concatenated_original_text,
                confidence=sum(box.confidence for box in boxes) / len(boxes),  # Average confidence
                x=min_x,
                y=min_y,
                width=max_x - min_x,
                height=max_y - min_y,
                paragraph_index=min(box.paragraph_index for box in boxes),
                line_index=0,  # Reset line index as this is now a single text block
                word_index=0,  # Reset word index
                source_image_path=boxes[0].source_image_path,
                original_box_bounds=original_box_bounds
            )
            
            # Copy bubble information
            bubble_box.bubble_id = bubble_id
            if hasattr(boxes[0], 'bubble_bounds'):
                bubble_box.bubble_bounds = boxes[0].bubble_bounds
            if hasattr(boxes[0], 'bubble_confidence'):
                bubble_box.bubble_confidence = boxes[0].bubble_confidence
            
            # Add the original boxes as children
            bubble_box.children = boxes
            
            result_text_boxes.append(bubble_box)
        
        # Add text boxes that aren't associated with any bubble
        result_text_boxes.extend(non_bubble_text_boxes)
        
        return result_text_boxes