import random
from typing import List, Callable, Optional, Dict, Any, Tuple
import copy
import translators as ts
from textbox import TextBox
import deepl


class MangaTextReplacer:
    """
    Specialized text replacer for manga translation.
    
    This class handles replacing Japanese text with translated text,
    while working with the existing TextBox class.
    """
    
    def __init__(self,
                placeholder_mode: bool = False,
                placeholder_char: str = 'a',
                translator: str = "deepl",
                target_language: str = 'en-gb',
                deepl_api_key: str = None):
        """
        Initialize the manga text replacer.
        
        Args:
            placeholder_mode: If True, use placeholder text instead of translations
            placeholder_char: Character to use for placeholder text
            target_language: Target language code for DeepL
        """
        self.placeholder_mode = placeholder_mode
        self.placeholder_char = placeholder_char
        self.target_language = target_language
        
        if translator == "deepl":
            self.translator = deepl.Translator(deepl_api_key)
        else:
            self.translator = None
    
    def replace_text(self, text_boxes: List[TextBox]) -> List[TextBox]:
        """
        Apply text replacement to manga text boxes.
        
        Args:
            text_boxes: List of TextBox objects with manga text
            
        Returns:
            List of TextBox objects with replaced text
        """
        # Create deep copy to avoid modifying originals
        modified_boxes = copy.deepcopy(text_boxes)
        
        # Apply replacement to each text box
        for box in modified_boxes:
            original_text = box.original_text
            print(original_text)
            
            if self.placeholder_mode:
                # Use placeholder text (for debugging or privacy)
                box.text = self._create_placeholder(original_text)
            else:
                # Try to translate with DeepL API
                translated_text = self.translator.translate_text(original_text, target_lang=self.target_language).text
                box.text = translated_text
            print(box.text)
        
        return modified_boxes
    
    def _create_placeholder(self, text: str) -> str:
        """
        Create placeholder text that mimics the original text structure.
        
        Args:
            text: Original text
            
        Returns:
            Placeholder text
        """
        # Single line - just replace with placeholder characters
        placeholder = self.placeholder_char * len(text.strip())
        # Add random spaces
        placeholder_list = list(placeholder)
        for i in range(1, len(placeholder_list), 2):
            if random.random() < 0.3:  # 30% chance to add space
                placeholder_list[i] = ' '
        return ''.join(placeholder_list)
    
    def set_placeholder_mode(self, enabled: bool, char: str = 'a') -> None:
        """
        Set placeholder mode settings.
        
        Args:
            enabled: Whether to enable placeholder mode
            char: Character to use for placeholders
        """
        self.placeholder_mode = enabled
        self.placeholder_char = char
    
    def batch_translate_with_deepl(self, text_boxes: List[TextBox]) -> Dict[str, str]:
        """
        Batch translate all text using DeepL API.
        
        Args:
            text_boxes: List of TextBox objects with original text
            
        Returns:
            Dictionary of original texts to translations
        """
        if not self.translator:
            return {}
        
        batch_texts = []
        for box in text_boxes:
            if box.original_text.strip():
                batch_texts.append(box.original_text)
        
        if not batch_texts:
            return {}
            
        new_translations = {}
        try:
            # Process in smaller batches to avoid API limits
            batch_size = 50
            for i in range(0, len(batch_texts), batch_size):
                current_batch = batch_texts[i:i+batch_size]
                # DeepL API handles batch translation differently than Google
                # We need to translate one by one in this loop
                for text in current_batch:
                    result = self.translator.translate_text(
                        text,
                        target_language=self.target_language
                    )
                    new_translations[text] = result.text
            
            return new_translations
            
        except Exception as e:
            print(f"Batch translation error: {e}")
            return {}