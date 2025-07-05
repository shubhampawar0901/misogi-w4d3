"""
Text Preprocessor for Plagiarism Detection
Handles text cleaning and normalization before embedding generation.
"""

import re
from typing import List


class TextPreprocessor:
    """Handles text preprocessing for similarity analysis."""
    
    def __init__(self, options: List[str] = None):
        """
        Initialize the preprocessor with selected options.
        
        Args:
            options: List of preprocessing options to apply
        """
        self.options = options or []
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess a single text based on selected options.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        processed_text = text
        
        # Apply selected preprocessing steps
        if "Remove extra whitespace" in self.options:
            processed_text = self._remove_extra_whitespace(processed_text)
        
        if "Convert to lowercase" in self.options:
            processed_text = processed_text.lower()
        
        if "Remove special characters" in self.options:
            processed_text = self._remove_special_characters(processed_text)
        
        return processed_text.strip()
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]
    
    def _remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize spacing."""
        # Replace multiple whitespace characters with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return text.strip()
    
    def _remove_special_characters(self, text: str) -> str:
        """Remove special characters, keeping only alphanumeric and basic punctuation."""
        # Keep letters, numbers, spaces, and basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)
        return text
    
    def get_text_stats(self, text: str) -> dict:
        """
        Get statistics about the text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {
                'character_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'paragraph_count': 0
            }
        
        # Character count
        char_count = len(text)
        
        # Word count
        words = text.split()
        word_count = len(words)
        
        # Sentence count (approximate)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Paragraph count
        paragraphs = text.split('\n\n')
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        return {
            'character_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count
        }
    
    def validate_text(self, text: str, max_length: int = 10000) -> tuple[bool, str]:
        """
        Validate text input.
        
        Args:
            text: Text to validate
            max_length: Maximum allowed length
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text or not text.strip():
            return False, "Text cannot be empty"
        
        if len(text) > max_length:
            return False, f"Text exceeds maximum length of {max_length} characters"
        
        # Check for minimum meaningful content
        words = text.split()
        if len(words) < 3:
            return False, "Text must contain at least 3 words"
        
        return True, ""
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract individual sentences from text.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 3:  # At least 3 words
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def get_preprocessing_summary(self) -> str:
        """
        Get a summary of applied preprocessing options.
        
        Returns:
            String describing the preprocessing steps
        """
        if not self.options:
            return "No preprocessing applied"
        
        return f"Applied: {', '.join(self.options)}"
