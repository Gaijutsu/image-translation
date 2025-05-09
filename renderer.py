import os
from typing import List, Tuple, Dict, Any, Optional

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from textbox import TextBox


class MangaTextRenderer:
    """
    Specialized renderer for manga translation text.
    
    This class handles rendering replaced text in manga speech bubbles,
    working with the standard TextBox class.
    """
    
    def __init__(self, 
                font_path: Optional[str] = None,
                padding: int = 0,
                background_color: Tuple[int, int, int] = (255, 255, 255),
                smooth_edges: bool = True):
        """
        Initialize the manga text renderer.
        
        Args:
            font_path: Path to a font file for text
            padding: Padding to add around text when covering original areas
            background_color: Default background color for text areas
            smooth_edges: Whether to smooth edges of text areas
        """
        self.padding = padding
        self.background_color = background_color
        self.smooth_edges = smooth_edges
        
        # Find appropriate manga fonts
        self.font_path = "./fonts/animeace3bb_ot/AnimeAce3BB_Regular.otf"
    
    def render_image(self, 
                    original_image_path: str, 
                    original_boxes: List[TextBox], 
                    modified_boxes: List[TextBox],
                    output_path: Optional[str] = None) -> Image.Image:
        """
        Render modified text boxes onto the manga image.
        
        Args:
            original_image_path: Path to the original manga image
            original_boxes: List of original TextBox objects
            modified_boxes: List of modified TextBox objects with translations
            output_path: Path to save the output image (if None, won't save)
            
        Returns:
            PIL Image with replaced text
        """
        # Load the original image
        original_image = Image.open(original_image_path)
        
        # First, inpaint all text boxes to remove original text
        inpainted_image = self._inpaint_all_text_boxes(original_image_path, original_boxes)
        
        # Create a drawing context
        draw = ImageDraw.Draw(inpainted_image)
        
        # Find bubbles from text boxes (if any)
        bubbles = {}
        for box in original_boxes:
            if hasattr(box, "bubble_id") and hasattr(box, "bubble_bounds"):
                bubbles[box.bubble_id] = box.bubble_bounds
        
        # Process each text box to add new text
        for orig_box, mod_box in zip(original_boxes, modified_boxes):
            # Skip if no replacement text
            if not mod_box.text.strip():
                continue
            
            # Draw the new text
            self._draw_manga_text(draw, inpainted_image, orig_box, mod_box)
        
        # Save the result if output path is provided
        if output_path:
            inpainted_image.save(output_path)
        
        return inpainted_image
    
    def _inpaint_all_text_boxes(self,
                              image_path: str,
                              text_boxes: List[TextBox]) -> Image.Image:
        """
        Apply inpainting to remove text from all text boxes at once.
        
        Args:
            image_path: Path to the original image
            text_boxes: List of TextBox objects to inpaint
            
        Returns:
            PIL Image with text removed
        """
        # Read image using CV2 (for inpainting)
        cv_img = cv2.imread(image_path)
        
        # Also read with PIL for dimensions
        pil_img = Image.open(image_path)
        
        # Create a global mask for all text areas
        mask = np.zeros((pil_img.height, pil_img.width), dtype=np.uint8)
        
        # Mark all text areas in the mask
        for box in text_boxes:
            for original_box_bounds in box.original_box_bounds:
                # Get text box coordinates
                x, y, w, h = original_box_bounds
                padding = self.padding
                
                # Add padding to the region
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(pil_img.width - x_pad, w + padding * 2)
                h_pad = min(pil_img.height - y_pad, h + padding * 2)
                
                # Skip if dimensions are invalid
                if w_pad <= 0 or h_pad <= 0:
                    continue
                
                # Mark this region in the mask (white = area to inpaint)
                mask[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad] = 255
        
        # Perform inpainting on the whole image
        inpainted_cv_img = cv2.inpaint(cv_img, mask, 7, cv2.INPAINT_NS)
        
        # Convert the final CV2 image back to PIL
        cv_img_rgb = cv2.cvtColor(inpainted_cv_img, cv2.COLOR_BGR2RGB)
        inpainted_image = Image.fromarray(cv_img_rgb)
        
        return inpainted_image
    
    def _cover_original_text(self, 
                           draw: ImageDraw.Draw, 
                           image: Image.Image, 
                           text_box: TextBox) -> None:
        """
        Cover the original text area with CV2 inpainting for better background preservation.
        
        Args:
            draw: PIL ImageDraw object
            image: PIL Image
            text_box: TextBox to cover
        """
        # Extract dimensions
        x, y, w, h = text_box.x, text_box.y, text_box.width, text_box.height
        padding = self.padding
        
        # Add padding to the region
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(image.width - x_pad, w + padding * 2)
        h_pad = min(image.height - y_pad, h + padding * 2) 
        
        # Skip if dimensions are invalid
        if w_pad <= 0 or h_pad <= 0:
            return
        
        # Create a temporary CV2 image from the PIL image for inpainting
        cv_img = np.array(image)
        # Convert to BGR
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        
        # Create a mask covering the text area
        mask = np.zeros((h_pad, w_pad), dtype="uint8")
        mask.fill(255)  # White = area to inpaint
        
        # Get the region to inpaint
        region_to_inpaint = cv_img[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
        masked = region_to_inpaint.copy()
        masked[mask == 0] = 0
        plt.imshow(masked)
        plt.show()
        # Perform inpainting
        inpainted_region = cv2.inpaint(region_to_inpaint, mask, 7, cv2.INPAINT_NS)
        plt.imshow(inpainted_region)
        plt.show()
        # Update the region in the CV2 image
        cv_img[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad] = inpainted_region
        
        # Convert back to PIL and update the original image
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        inpainted_image = Image.fromarray(cv_img_rgb)
        # inpainted_image.show()
        # Replace the original image content with inpainted content
        region = inpainted_image.crop((x_pad, y_pad, x_pad + w_pad, y_pad + h_pad))
        image.paste(region, (x_pad, y_pad))
    
    def _get_background_color(self, image: Image.Image, text_box: TextBox) -> Tuple[int, int, int]:
        """
        Get the background color around a text box.
        
        Args:
            image: PIL Image
            text_box: TextBox object
            
        Returns:
            RGB color tuple
        """
        x, y, w, h = text_box.x, text_box.y, text_box.width, text_box.height
        padding = self.padding * 3  # Sample a bit further out
        
        # Create sampling regions around the text
        regions = []
        
        # Top
        if y - padding >= 0:
            top_region = image.crop((x, max(0, y - padding), x + w, y))
            regions.append(top_region)
        
        # Bottom
        if y + h + padding < image.height:
            bottom_region = image.crop((x, y + h, x + w, min(image.height, y + h + padding)))
            regions.append(bottom_region)
        
        # Left
        if x - padding >= 0:
            left_region = image.crop((max(0, x - padding), y, x, y + h))
            regions.append(left_region)
        
        # Right
        if x + w + padding < image.width:
            right_region = image.crop((x + w, y, min(image.width, x + w + padding), y + h))
            regions.append(right_region)
        
        # Compute the most common color
        all_pixels = []
        for region in regions:
            all_pixels.extend(list(region.getdata()))
        
        if not all_pixels:
            return self.background_color
        
        # Use the most common color
        color_counts = {}
        for pixel in all_pixels:
            rgb = pixel[:3]  # Handle both RGB and RGBA
            if rgb in color_counts:
                color_counts[rgb] += 1
            else:
                color_counts[rgb] = 1
        
        if color_counts:
            most_common = max(color_counts.items(), key=lambda x: x[1])[0]
            return most_common
        
        # Fallback
        return self.background_color
    
    def _draw_manga_text(self, 
                       draw: ImageDraw.Draw, 
                       image: Image.Image,
                       original_box: TextBox, 
                       modified_box: TextBox) -> None:
        """
        Draw manga text with proper styling.
        
        Args:
            draw: PIL ImageDraw object
            image: PIL Image
            original_box: Original TextBox with position and style
            modified_box: Modified TextBox with new text
        """
        # Determine text properties
        x, y, w, h = original_box.x, original_box.y, original_box.width, original_box.height
        text = modified_box.text
        
        # Determine bubble bounds (if any)
        bubble_bounds = None
        if hasattr(original_box, "bubble_bounds"):
            bubble_bounds = original_box.bubble_bounds
        
        # Determine font size based on bubble or text box size
        if bubble_bounds:
            # For text in bubbles, size based on bubble dimensions
            bx, by, bw, bh = bubble_bounds
            
            # Calculate based on bubble size and text length
            lines = text.split('\n')
            max_line_length = max(len(line) for line in lines) if lines else 0
            font_size = int(min(bw / (max_line_length * 0.6), bh / (len(lines) * 1.5)))
        else:
            # For text outside bubbles, base on the original text box size
            lines = text.split('\n')
            max_line_length = max(len(line) for line in lines) if lines else 0
            font_size = int(min(w / (max_line_length * 0.6), h / (len(lines) * 1.5)))
        
        # Ensure minimum readable size
        font_size = max(font_size, 12)
        
        try:
            # Load font with computed size
            font = ImageFont.truetype(self.font_path, font_size) if self.font_path else None
            if not font:
                font = ImageFont.load_default()
        except Exception as e:
            print(f"Font error: {e}, falling back to default")
            font = ImageFont.load_default()
        
        # Determine text color (usually black for manga)
        text_color = (0, 0, 0)

        print(f"Drawing text: {text} in bubble: {bubble_bounds}")
        
        # Render text based on bubble or text box position
        if bubble_bounds:
            self._draw_text_in_bubble(draw, bubble_bounds, text, font, text_color)
        else:
            self._draw_text_in_box(draw, x, y, w, h, text, font, text_color)
    
    def _draw_text_in_bubble(self, 
                           draw: ImageDraw.Draw, 
                           bubble_bounds: tuple,
                           text: str, 
                           font: ImageFont.FreeTypeFont,
                           color: Tuple[int, int, int]) -> None:
        """
        Draw text centered in a speech bubble with automatic word wrapping.
        
        Args:
            draw: PIL ImageDraw object
            bubble_bounds: (x, y, width, height) of the bubble
            text: Text to draw
            font: Font to use
            color: Text color
        """
        bx, by, bw, bh = bubble_bounds
        
        # Add some padding inside the bubble
        padding = 10
        max_width = bw - (padding * 2)
        
        # Wrap text to fit the bubble width
        wrapped_lines = []
        # Split text by existing newlines first
        paragraphs = text.split('\n')
        
        for paragraph in paragraphs:
            # If paragraph is empty, add it as is
            if not paragraph:
                wrapped_lines.append("")
                continue
                
            words = paragraph.split(' ')
            current_line = words[0]
            
            for word in words[1:]:
                # Calculate width with the new word added
                test_line = current_line + " " + word
                _, _, text_width, _ = font.getbbox(test_line)
                
                if text_width <= max_width:
                    # Word fits on current line
                    current_line = test_line
                else:
                    # Word doesn't fit, start a new line
                    wrapped_lines.append(current_line)
                    current_line = word
            
            # Add the last line
            wrapped_lines.append(current_line)
        
        # Calculate total text height
        line_height = font.getbbox("Ap")[3]
        total_text_height = len(wrapped_lines) * line_height
        
        # Ensure text fits vertically
        if total_text_height > bh - (padding * 2):
            # If text is too tall, reduce font size
            # Note: A proper implementation would recalculate wrapping with new font size
            # This is a simplified approach
            scale_factor = (bh - (padding * 2)) / total_text_height
            new_font_size = int(font.size * scale_factor)
            if new_font_size >= 8:  # Don't make text too small to read
                font = ImageFont.truetype(self.font_path, new_font_size)
                # Recalculate with new font
                line_height = font.getbbox("Ap")[3]
                total_text_height = len(wrapped_lines) * line_height
        
        # Center text in bubble
        start_y = by + (bh - total_text_height) // 2
        
        # Render each line
        for i, line in enumerate(wrapped_lines):
            # Get text width for centering
            _, _, text_width, _ = font.getbbox(line)
            
            # Center horizontally in bubble
            line_x = bx + (bw - text_width) // 2
            line_y = start_y + i * line_height
            
            # Draw the text
            draw.text((line_x, line_y), line, font=font, fill=color)
    
    def _draw_text_in_box(self, 
                        draw: ImageDraw.Draw, 
                        x: int, y: int, w: int, h: int,
                        text: str, 
                        font: ImageFont.FreeTypeFont,
                        color: Tuple[int, int, int]) -> None:
        """
        Draw text within a text box (non-bubble).
        
        Args:
            draw: PIL ImageDraw object
            x, y, w, h: Text box position and dimensions
            text: Text to draw
            font: Font to use
            color: Text color
        """
        # Add padding inside the text box
        padding = 5
        max_width = w - (padding * 2)
        
        # Wrap text to fit the text box width, ignoring existing newlines
        wrapped_lines = []
        
        # Replace newlines with spaces to ignore existing line breaks
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Split into words and handle line breaks based on width
        words = text.split(' ')
        
        # Handle empty text case
        if not words:
            wrapped_lines = [""]
        else:
            current_line = words[0]
            
            for word in words[1:]:
                # Skip empty words (from multiple spaces)
                if not word:
                    continue
                    
                # Calculate width with the new word added
                test_line = current_line + " " + word
                _, _, text_width, _ = font.getbbox(test_line)
                
                if text_width <= max_width:
                    # Word fits on current line
                    current_line = test_line
                else:
                    # Word doesn't fit, start a new line
                    wrapped_lines.append(current_line)
                    current_line = word
            
            # Add the last line
            if current_line:
                wrapped_lines.append(current_line)
        
        # Calculate total text height
        line_height = font.getbbox("Ap")[3]
        total_text_height = len(wrapped_lines) * line_height
        
        # Ensure text fits vertically
        if total_text_height > h - (padding * 2):
            # If text is too tall, reduce font size
            scale_factor = (h - (padding * 2)) / total_text_height
            new_font_size = int(font.size * scale_factor)
            if new_font_size >= 8:  # Don't make text too small to read
                font = ImageFont.truetype(self.font_path, new_font_size)
                # Recalculate with new font
                line_height = font.getbbox("Ap")[3]
                total_text_height = len(wrapped_lines) * line_height
        
        # Center text in box
        start_y = y + (h - total_text_height) // 2
        
        # Render each line
        for i, line in enumerate(wrapped_lines):
            # Get text width for centering
            _, _, text_width, _ = font.getbbox(line)
            
            # Center horizontally in text box
            line_x = x + (w - text_width) // 2
            line_y = start_y + i * line_height
            
            # Draw the text
            draw.text((line_x, line_y), line, font=font, fill=color)
    
    def remove_text_with_inpainting(self, 
                                   image_path: str, 
                                   text_boxes: List[TextBox],
                                   output_path: Optional[str] = None) -> Image.Image:
        """
        Remove text from specified text boxes using CV2 inpainting.
        
        Args:
            image_path: Path to the original image
            text_boxes: List of TextBox objects defining regions to process
            output_path: Path to save the output image (if None, won't save)
            
        Returns:
            PIL Image with text removed from specified regions
        """
        # Read image using CV2 (for inpainting)
        cv_img = cv2.imread(image_path)
        
        # Also read with PIL for processing
        pil_img = Image.open(image_path)
        
        # Process each text box separately
        for box in text_boxes:
            # Get text box coordinates
            x, y, w, h = box.x, box.y, box.width, box.height
            
            # Ensure we don't go out of bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, pil_img.width - x)
            h = min(h, pil_img.height - y)
            
            # Skip if dimensions are invalid
            if w <= 0 or h <= 0:
                continue
                
            # Create a mask covering the entire text box
            mask = np.zeros((h, w), dtype="uint8")
            
            # Fill the mask (white = area to inpaint)
            mask.fill(255)
            
            # Create a region in the original CV2 image to inpaint
            region_to_inpaint = cv_img[y:y+h, x:x+w]
            
            # Perform inpainting on this region
            inpainted_region = cv2.inpaint(region_to_inpaint, mask, 7, cv2.INPAINT_NS)
            
            # Place the inpainted region back into the main image
            cv_img[y:y+h, x:x+w] = inpainted_region
        
        # Convert the final CV2 image back to PIL
        # OpenCV uses BGR order while PIL uses RGB
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(cv_img_rgb)
        
        # Save if output path provided
        if output_path:
            result_image.save(output_path)
        
        return result_image