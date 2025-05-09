from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List


@dataclass
class TextBox:
    """
    Enhanced TextBox class with additional properties for better layout handling.
    
    This class represents a text box detected in an image with position,
    ordering information, and additional attributes to support improved
    text replacement.
    """

    id: str
    text: str
    original_text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    paragraph_index: int
    line_index: int
    word_index: int
    source_image_path: str
    font_properties: Optional[Dict[str, Any]] = None
    right_spacing: float = 0.0  # Spacing to the next box on the right
    is_part_of_line: bool = True  # Whether this box is part of a text line
    line_id: Optional[str] = None  # ID of the line this box belongs to
    children: List['TextBox'] = field(default_factory=list)  # Child boxes if this is a unified box
    parent_id: Optional[str] = None  # ID of parent box if this is a child
    original_box_bounds: Optional[List[Tuple[int, int, int, int]]] = None
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Return the bounds of the text box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def box_coordinates(self) -> Tuple[int, int, int, int]:
        """Return the box coordinates as (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    @property
    def area(self) -> int:
        """Return the area of the text box in pixels."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Return the center coordinates of the box."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def aspect_ratio(self) -> float:
        """Return the aspect ratio (width / height) of the box."""
        return self.width / max(1, self.height)  # Avoid division by zero
        
    def get_ordering_key(self) -> Tuple[int, int, int]:
        """Get a tuple that can be used for sorting text boxes in reading order."""
        return (self.paragraph_index, self.line_index, self.word_index)
    
    def set_font_properties(self, font_size: int, font_family: str, font_color: Tuple[int, int, int], 
                           font_weight: str = "normal") -> None:
        """Set font properties for this text box."""
        self.font_properties = {
            "size": font_size,
            "family": font_family,
            "color": font_color,
            "weight": font_weight
        }
    
    def get_char_width(self) -> float:
        """Get the average character width for this text box."""
        if not self.text:
            return 0
        return self.width / max(1, len(self.text))
    
    def distance_to(self, other: 'TextBox') -> float:
        """Calculate the Euclidean distance to another text box."""
        c1 = self.center
        c2 = other.center
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
    
    def horizontal_distance_to(self, other: 'TextBox') -> float:
        """Calculate the horizontal distance to another text box."""
        # If self is to the left of other
        if self.x + self.width <= other.x:
            return other.x - (self.x + self.width)
        # If other is to the left of self
        elif other.x + other.width <= self.x:
            return self.x - (other.x + other.width)
        # If they overlap horizontally
        else:
            return 0
    
    def vertical_distance_to(self, other: 'TextBox') -> float:
        """Calculate the vertical distance to another text box."""
        # If self is above other
        if self.y + self.height <= other.y:
            return other.y - (self.y + self.height)
        # If other is above self
        elif other.y + other.height <= self.y:
            return self.y - (other.y + other.height)
        # If they overlap vertically
        else:
            return 0
    
    def horizontal_overlap(self, other: 'TextBox') -> float:
        """Calculate the horizontal overlap ratio with another text box."""
        # Calculate overlap
        x_overlap = max(0, min(self.x + self.width, other.x + other.width) - 
                        max(self.x, other.x))
        
        # Calculate the minimum width
        min_width = min(self.width, other.width)
        
        # Return overlap ratio
        return x_overlap / max(1, min_width)
    
    def vertical_overlap(self, other: 'TextBox') -> float:
        """Calculate the vertical overlap ratio with another text box."""
        # Calculate overlap
        y_overlap = max(0, min(self.y + self.height, other.y + other.height) - 
                        max(self.y, other.y))
        
        # Calculate the minimum height
        min_height = min(self.height, other.height)
        
        # Return overlap ratio
        return y_overlap / max(1, min_height)
    
    def is_horizontally_adjacent(self, other: 'TextBox', threshold: float = 3.0) -> bool:
        """
        Check if another text box is horizontally adjacent to this one.
        
        Args:
            other: The other TextBox to check
            threshold: Maximum horizontal distance as a factor of average character width
            
        Returns:
            True if the boxes are horizontally adjacent
        """
        # Check vertical overlap
        if self.vertical_overlap(other) < 0.5:
            return False
        
        # Check horizontal distance
        avg_char_width = (self.get_char_width() + other.get_char_width()) / 2
        horizontal_dist = self.horizontal_distance_to(other)
        
        return horizontal_dist <= threshold * avg_char_width
    
    def is_same_line(self, other: 'TextBox', threshold: float = 0.5) -> bool:
        """
        Check if another text box is on the same line as this one.
        
        Args:
            other: The other TextBox to check
            threshold: Maximum vertical distance as a factor of text height
            
        Returns:
            True if the boxes are on the same line
        """
        # Calculate vertical distance between center points
        vert_distance = abs(self.center[1] - other.center[1])
        avg_height = (self.height + other.height) / 2
        
        return vert_distance <= threshold * avg_height
    
    def merge_with(self, other: 'TextBox') -> 'TextBox':
        """
        Create a new TextBox by merging this one with another.
        
        Args:
            other: The other TextBox to merge with
            
        Returns:
            A new TextBox containing the merged information
        """
        # Calculate the new bounds
        min_x = min(self.x, other.x)
        min_y = min(self.y, other.y)
        max_x = max(self.x + self.width, other.x + other.width)
        max_y = max(self.y + self.height, other.y + other.height)
        
        # Create the new text box
        merged_box = TextBox(
            id=f"{self.id}_merged_{other.id}",
            text=self.text + other.text,
            original_text=self.original_text + other.original_text,
            confidence=(self.confidence + other.confidence) / 2,
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            paragraph_index=min(self.paragraph_index, other.paragraph_index),
            line_index=min(self.line_index, other.line_index),
            word_index=min(self.word_index, other.word_index),
            source_image_path=self.source_image_path
        )
        
        # Add the original boxes as children
        merged_box.children = [self, other]
        
        return merged_box