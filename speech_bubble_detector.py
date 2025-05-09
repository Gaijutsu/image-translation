import os
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
import urllib.request
import tempfile

import numpy as np
from PIL import Image
import cv2

# Import Ultralytics YOLO
from ultralytics import YOLO

from textbox import TextBox


class YOLOSpeechBubbleDetector:
    """
    Speech bubble detector using YOLOv8 pre-trained model.
    
    This class uses a specialized YOLOv8 model trained on comic/manga speech bubbles
    to detect and extract speech bubble regions.
    """
    
    MODEL_URL = "https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m/resolve/main/comic-speech-bubble-detector.pt"
    
    def __init__(self, 
                model_path: Optional[str] = None,
                confidence: float = 0.25,
                download_model: bool = True,
                cache_dir: Optional[str] = None):
        """
        Initialize the YOLO speech bubble detector.
        
        Args:
            model_path: Path to the YOLOv8 model file (optional)
            confidence: Confidence threshold for detections
            download_model: Whether to download the model if not found
            cache_dir: Directory to cache downloaded model
        """
        self.confidence = confidence
        self.model = None
        
        # Set up model path
        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        elif download_model:
            self.model_path = self._download_model(cache_dir)
        else:
            raise ValueError("Model path not provided and download_model is False")
        
        # Initialize model
        self._load_model()
    
    def _download_model(self, cache_dir: Optional[str] = None) -> str:
        """
        Download the YOLOv8 speech bubble detection model.
        
        Args:
            cache_dir: Directory to cache the model
            
        Returns:
            Path to the downloaded model
        """
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            model_path = os.path.join(cache_dir, "comic-speech-bubble-detector.pt")
        else:
            # Use a persistent temp directory
            temp_dir = tempfile.gettempdir()
            model_path = os.path.join(temp_dir, "comic-speech-bubble-detector.pt")
        
        # Only download if it doesn't exist
        if not os.path.exists(model_path):
            print(f"Downloading speech bubble detection model to {model_path}...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, model_path)
                print("Download complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")
        else:
            print(f"Using cached model at {model_path}")
        
        return model_path
    
    def _load_model(self) -> None:
        """Load the YOLOv8 model."""
        try:
            self.model = YOLO(self.model_path)
            print(f"YOLOv8 speech bubble detection model loaded from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def detect_bubbles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect speech bubbles in an image using YOLOv8.
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            List of bubble dictionaries with bounds and IDs
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        bubbles = []
        
        try:
            # Run inference
            results = self.model(image, conf=self.confidence)
            
            # Process detections
            for i, detection in enumerate(results[0].boxes):
                # Extract box coordinates
                box = detection.xyxy[0].cpu().numpy()  # Get box in xyxy format
                x1, y1, x2, y2 = box.astype(int)
                
                # Calculate width and height
                w = x2 - x1
                h = y2 - y1
                
                # Get confidence score
                conf = detection.conf.item()
                
                # Create bubble dictionary
                bubble = {
                    "id": f"bubble-{i}",
                    "bounds": (x1, y1, w, h),
                    "confidence": conf,
                    "class": detection.cls.item()
                }
                
                bubbles.append(bubble)
            
            # Sort bubbles by position (top to bottom, left to right)
            bubbles.sort(key=lambda b: (b["bounds"][1], b["bounds"][0]))
            
            # Reassign IDs based on sorted order
            for i, bubble in enumerate(bubbles):
                bubble["id"] = f"bubble-{i}"
        
        except Exception as e:
            print(f"Error in YOLO bubble detection: {e}")
        
        return bubbles
    
    def visualize_bubbles(self, 
                         image: np.ndarray, 
                         bubbles: List[Dict[str, Any]],
                         output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detected speech bubbles on the image.
        
        Args:
            image: OpenCV image
            bubbles: List of detected bubbles
            output_path: Path to save the visualization
            
        Returns:
            Image with visualized bubbles
        """
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Draw each bubble
        for bubble in bubbles:
            x, y, w, h = bubble["bounds"]
            conf = bubble.get("confidence", 1.0)
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label with ID and confidence
            label = f"{bubble['id']} ({conf:.2f})"
            cv2.putText(vis_image, label, (x, y - 10), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save visualization if output path is provided
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return vis_image
    
    def visualize_text_boxes(self, 
                            image: np.ndarray, 
                            text_boxes: List[TextBox],
                            output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detected text boxes on the image.
        
        Args:
            image: OpenCV image
            text_boxes: List of detected text boxes
            output_path: Path to save the visualization
        """
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Draw each text box
        for box in text_boxes:
            cv2.rectangle(vis_image, (box.x, box.y), (box.x + box.width, box.y + box.height), (0, 255, 0), 2)
        
        # Save visualization if output path is provided
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return vis_image

        
    def find_bubble_outline(self, 
                           image: np.ndarray, 
                           bubble: Dict[str, Any],
                           edge_low_threshold: int = 50,
                           edge_high_threshold: int = 100,
                           blur_kernel_size: int = 5,
                           padding: int = 10) -> Dict[str, Any]:
        """
        Find the precise outline of a speech bubble using edge detection.
        
        Args:
            image: OpenCV image (numpy array)
            bubble: Bubble dictionary with bounding box
            edge_low_threshold: Lower threshold for Canny edge detection
            edge_high_threshold: Higher threshold for Canny edge detection
            blur_kernel_size: Size of Gaussian blur kernel
            padding: Padding to add around the bounding box
            
        Returns:
            Updated bubble dictionary with contour information
        """
        # Extract bubble region with padding to capture the complete outline
        x, y, w, h = bubble["bounds"]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # Extract the region of interest
        roi = image[y1:y2, x1:x2].copy()
        
        if len(roi) == 0 or len(roi[0]) == 0:
            bubble["precise_shape"] = False
            return bubble
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Apply morphological operations to remove text
        # Create a kernel
        kernel = np.ones((3, 3), np.uint8)
        
        # Apply binary threshold to separate dark text from light background
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Dilate to fill in text regions
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, edge_low_threshold, edge_high_threshold)
        
        # Dilate edges to connect broken parts
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and find the most likely bubble contour
        if contours:
            # For bubbles, we're looking for:
            # 1. Reasonably large area
            # 2. Located near the center of the ROI
            # 3. Approximately convex shape
            
            roi_center = (roi.shape[1] // 2, roi.shape[0] // 2)
            best_contour = None
            best_score = -float('inf')
            min_area = 0.1 * w * h  # Minimum area threshold
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Skip very small contours
                if area < min_area:
                    continue
                
                # Check if contour is near the border (might be a partial bubble)
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                # Calculate center of contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center_dist = np.sqrt((cx - roi_center[0])**2 + (cy - roi_center[1])**2)
                else:
                    center_dist = float('inf')
                
                # Calculate convexity - ratio of contour area to its convex hull area
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                convexity = area / hull_area if hull_area > 0 else 0
                
                # Calculate a score based on area, center distance, and convexity
                # Higher score = better bubble candidate
                score = area - center_dist * 0.5 + convexity * 100
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
            
            if best_contour is not None:
                # Adjust contour coordinates to match the original image
                adjusted_contour = best_contour.copy()
                adjusted_contour[:, :, 0] += x1
                adjusted_contour[:, :, 1] += y1
                
                # Update the bubble dictionary with contour information
                bubble["contour"] = adjusted_contour
                bubble["precise_shape"] = True
                bubble["contour_area"] = cv2.contourArea(best_contour)
                bubble["contour_perimeter"] = cv2.arcLength(best_contour, True)
                
                # Compute a simplified polygonal representation of the contour
                epsilon = 0.005 * bubble["contour_perimeter"]
                bubble["approx_contour"] = cv2.approxPolyDP(adjusted_contour, epsilon, True)
            else:
                bubble["precise_shape"] = False
        else:
            # No contour found
            bubble["precise_shape"] = False
        
        return bubble
    
    def find_all_bubble_outlines(self, 
                               image: np.ndarray, 
                               bubbles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find precise outlines for all detected bubbles.
        
        Args:
            image: OpenCV image
            bubbles: List of detected bubbles
            
        Returns:
            List of bubbles with contour information
        """
        return [self.find_bubble_outline(image, bubble) for bubble in bubbles]
    
    def visualize_precise_bubbles(self, 
                                image: np.ndarray, 
                                bubbles: List[Dict[str, Any]],
                                output_path: Optional[str] = None,
                                show_approx: bool = False) -> np.ndarray:
        """
        Visualize precise bubble outlines on the image.
        
        Args:
            image: OpenCV image
            bubbles: List of detected bubbles with contour information
            output_path: Path to save the visualization
            show_approx: Whether to show the simplified polygon approximation
            
        Returns:
            Image with visualized precise bubble outlines
        """
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        for bubble in bubbles:
            # Draw the bounding box in green
            x, y, w, h = bubble["bounds"]
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
            # If precise shape is available, draw the contour in red
            if bubble.get("precise_shape", False) and "contour" in bubble:
                cv2.drawContours(vis_image, [bubble["contour"]], -1, (0, 0, 255), 2)
                
                # If available and requested, show approximated polygon in blue
                if show_approx and "approx_contour" in bubble:
                    cv2.drawContours(vis_image, [bubble["approx_contour"]], -1, (255, 0, 0), 1)
            
            # Add label
            label = bubble["id"]
            cv2.putText(vis_image, label, (x, y - 10), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save visualization if output path is provided
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return vis_image