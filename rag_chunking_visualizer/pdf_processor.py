"""
PDF Processor for RAG Chunking Visualizer
Handles PDF text extraction and preprocessing.
"""

import PyPDF2
import io
from typing import Optional, List
import re


class PDFProcessor:
    """Handles PDF processing and text extraction."""
    
    def __init__(self):
        """Initialize the PDF processor."""
        pass
    
    def extract_text(self, uploaded_file) -> str:
        """
        Extract text from uploaded PDF file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Extracted text as string
        """
        try:
            # Read the uploaded file
            pdf_bytes = uploaded_file.read()
            pdf_file = io.BytesIO(pdf_bytes)
            
            # Create PDF reader
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n"
            
            # Clean and preprocess the text
            cleaned_text = self._clean_text(text)
            
            return cleaned_text
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text, flags=re.IGNORECASE)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between words
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after sentences
        
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def get_pdf_metadata(self, uploaded_file) -> dict:
        """
        Extract metadata from PDF file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Dictionary with PDF metadata
        """
        try:
            pdf_bytes = uploaded_file.read()
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            metadata = {
                'num_pages': len(pdf_reader.pages),
                'file_size': len(pdf_bytes),
                'title': '',
                'author': '',
                'subject': '',
                'creator': ''
            }
            
            # Extract metadata if available
            if pdf_reader.metadata:
                metadata.update({
                    'title': pdf_reader.metadata.get('/Title', ''),
                    'author': pdf_reader.metadata.get('/Author', ''),
                    'subject': pdf_reader.metadata.get('/Subject', ''),
                    'creator': pdf_reader.metadata.get('/Creator', '')
                })
            
            return metadata
            
        except Exception as e:
            return {'error': str(e)}
    
    def extract_text_by_page(self, uploaded_file) -> List[str]:
        """
        Extract text from PDF, returning a list of pages.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            List of strings, one per page
        """
        try:
            pdf_bytes = uploaded_file.read()
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            pages = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                cleaned_text = self._clean_text(page_text)
                pages.append(cleaned_text)
            
            return pages
            
        except Exception as e:
            raise Exception(f"Error extracting text by page: {str(e)}")
    
    def validate_pdf(self, uploaded_file) -> tuple[bool, str]:
        """
        Validate uploaded PDF file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if uploaded_file is None:
                return False, "No file uploaded"
            
            if not uploaded_file.name.lower().endswith('.pdf'):
                return False, "File must be a PDF"
            
            # Check file size (limit to 50MB)
            if uploaded_file.size > 50 * 1024 * 1024:
                return False, "File size must be less than 50MB"
            
            # Try to read the PDF
            pdf_bytes = uploaded_file.read()
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Check if PDF has pages
            if len(pdf_reader.pages) == 0:
                return False, "PDF appears to be empty"
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            return True, ""
            
        except Exception as e:
            return False, f"Invalid PDF file: {str(e)}"
    
    def extract_text_with_structure(self, uploaded_file) -> dict:
        """
        Extract text with basic structure information.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Dictionary with structured text information
        """
        try:
            pages = self.extract_text_by_page(uploaded_file)
            full_text = "\n".join(pages)
            
            # Basic structure analysis
            paragraphs = self._extract_paragraphs(full_text)
            sentences = self._extract_sentences(full_text)
            
            return {
                'full_text': full_text,
                'pages': pages,
                'paragraphs': paragraphs,
                'sentences': sentences,
                'stats': {
                    'num_pages': len(pages),
                    'num_paragraphs': len(paragraphs),
                    'num_sentences': len(sentences),
                    'num_characters': len(full_text),
                    'num_words': len(full_text.split())
                }
            }
            
        except Exception as e:
            raise Exception(f"Error extracting structured text: {str(e)}")
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text."""
        if not text:
            return []
        
        # Split on double newlines or multiple spaces
        paragraphs = re.split(r'\n\s*\n|\n\s{4,}', text)
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para.split()) >= 3:  # At least 3 words
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        if not text:
            return []
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 3:  # At least 3 words
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
