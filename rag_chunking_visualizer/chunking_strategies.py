"""
Chunking Strategies for RAG Systems
Implements various text chunking approaches for optimal RAG performance.
"""

import re
import tiktoken
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ChunkingStrategyManager:
    """Manages different text chunking strategies."""
    
    def __init__(self):
        """Initialize the chunking strategy manager."""
        self.embedding_model = None
        self.tokenizer = None
    
    def apply_strategy(self, text: str, strategy: str, params: Dict[str, Any]) -> List[str]:
        """
        Apply a specific chunking strategy to text.
        
        Args:
            text: Input text to chunk
            strategy: Name of the chunking strategy
            params: Strategy-specific parameters
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        strategy_methods = {
            "Fixed Size": self._fixed_size_chunking,
            "Sentence-based": self._sentence_based_chunking,
            "Paragraph-based": self._paragraph_based_chunking,
            "Semantic": self._semantic_chunking,
            "Recursive Character": self._recursive_character_chunking,
            "Token-based": self._token_based_chunking,
            "Sliding Window": self._sliding_window_chunking
        }
        
        if strategy not in strategy_methods:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        return strategy_methods[strategy](text, params)
    
    def _fixed_size_chunking(self, text: str, params: Dict[str, Any]) -> List[str]:
        """Fixed size chunking with overlap."""
        chunk_size = params.get('chunk_size', 500)
        overlap = params.get('overlap', 50)
        
        if chunk_size <= 0:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            # Move start position with overlap
            start = end - overlap
            
            # Prevent infinite loop
            if start >= end:
                break
        
        return chunks
    
    def _sentence_based_chunking(self, text: str, params: Dict[str, Any]) -> List[str]:
        """Sentence-based chunking."""
        sentences_per_chunk = params.get('sentences_per_chunk', 3)
        overlap = params.get('overlap', 1)
        
        # Extract sentences
        sentences = self._extract_sentences(text)
        
        if not sentences:
            return [text]
        
        chunks = []
        i = 0
        
        while i < len(sentences):
            # Take sentences_per_chunk sentences
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk = ' '.join(chunk_sentences)
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            # Move to next chunk with overlap
            i += sentences_per_chunk - overlap
            
            # Prevent infinite loop
            if i <= 0:
                i = 1
        
        return chunks
    
    def _paragraph_based_chunking(self, text: str, params: Dict[str, Any]) -> List[str]:
        """Paragraph-based chunking."""
        min_chunk_size = params.get('min_chunk_size', 100)
        max_chunk_size = params.get('max_chunk_size', 1000)
        
        # Extract paragraphs
        paragraphs = self._extract_paragraphs(text)
        
        if not paragraphs:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, start new chunk
            if current_chunk and len(current_chunk + " " + paragraph) > max_chunk_size:
                if len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += " " + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk and len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _semantic_chunking(self, text: str, params: Dict[str, Any]) -> List[str]:
        """Semantic chunking based on sentence similarity."""
        threshold = params.get('threshold', 0.8)
        min_chunk_size = params.get('min_chunk_size', 100)
        
        # Initialize embedding model if needed
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Extract sentences
        sentences = self._extract_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # Generate embeddings for sentences
        embeddings = self.embedding_model.encode(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Group sentences based on similarity
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Calculate average similarity with current chunk
            similarities = [similarity_matrix[i][j] for j in range(len(current_chunk))]
            avg_similarity = np.mean(similarities)
            
            if avg_similarity >= threshold:
                current_chunk.append(sentences[i])
            else:
                # Start new chunk if current chunk meets minimum size
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= min_chunk_size:
                    chunks.append(chunk_text)
                current_chunk = [sentences[i]]
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def _recursive_character_chunking(self, text: str, params: Dict[str, Any]) -> List[str]:
        """Recursive character-based chunking."""
        chunk_size = params.get('chunk_size', 500)
        overlap = params.get('overlap', 50)
        separators = params.get('separators', ['\n\n', '\n', '. ', ' '])
        
        def split_text(text: str, separators: List[str]) -> List[str]:
            """Recursively split text using separators."""
            if not separators or len(text) <= chunk_size:
                return [text] if text.strip() else []
            
            separator = separators[0]
            remaining_separators = separators[1:]
            
            if separator in text:
                splits = text.split(separator)
                result = []
                
                for split in splits:
                    if len(split) > chunk_size:
                        result.extend(split_text(split, remaining_separators))
                    elif split.strip():
                        result.append(split.strip())
                
                return result
            else:
                return split_text(text, remaining_separators)
        
        # Split the text
        initial_chunks = split_text(text, separators)
        
        # Merge small chunks and add overlap
        final_chunks = []
        current_chunk = ""
        
        for chunk in initial_chunks:
            if len(current_chunk + " " + chunk) <= chunk_size:
                if current_chunk:
                    current_chunk += " " + chunk
                else:
                    current_chunk = chunk
            else:
                if current_chunk:
                    final_chunks.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            final_chunks.append(current_chunk)
        
        return final_chunks
    
    def _token_based_chunking(self, text: str, params: Dict[str, Any]) -> List[str]:
        """Token-based chunking using tiktoken."""
        tokens_per_chunk = params.get('tokens_per_chunk', 200)
        overlap = params.get('overlap', 20)
        model = params.get('model', 'gpt-3.5-turbo')
        
        # Initialize tokenizer if needed
        if self.tokenizer is None:
            try:
                self.tokenizer = tiktoken.encoding_for_model(model)
            except:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + tokens_per_chunk
            chunk_tokens = tokens[start:end]
            
            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            # Move start position with overlap
            start = end - overlap
            
            # Prevent infinite loop
            if start >= end:
                break
        
        return chunks
    
    def _sliding_window_chunking(self, text: str, params: Dict[str, Any]) -> List[str]:
        """Sliding window chunking."""
        window_size = params.get('window_size', 500)
        step_size = params.get('step_size', 250)
        
        if window_size <= 0 or step_size <= 0:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + window_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start += step_size
            
            # Stop if we've reached the end
            if start >= len(text):
                break
        
        return chunks
    
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
    
    def get_strategy_info(self, strategy: str) -> Dict[str, str]:
        """Get information about a chunking strategy."""
        strategy_info = {
            "Fixed Size": {
                "description": "Splits text into fixed-size chunks with optional overlap",
                "best_for": "Consistent chunk sizes, predictable processing",
                "parameters": "chunk_size, overlap"
            },
            "Sentence-based": {
                "description": "Groups sentences together into chunks",
                "best_for": "Maintaining sentence boundaries, readability",
                "parameters": "sentences_per_chunk, overlap"
            },
            "Paragraph-based": {
                "description": "Uses paragraph boundaries for chunking",
                "best_for": "Preserving document structure, topic coherence",
                "parameters": "min_chunk_size, max_chunk_size"
            },
            "Semantic": {
                "description": "Groups semantically similar sentences together",
                "best_for": "Topic coherence, semantic similarity",
                "parameters": "threshold, min_chunk_size"
            },
            "Recursive Character": {
                "description": "Recursively splits text using multiple separators",
                "best_for": "Balanced chunks, preserving structure",
                "parameters": "chunk_size, overlap, separators"
            },
            "Token-based": {
                "description": "Splits based on token count for specific models",
                "best_for": "Model-specific token limits, API optimization",
                "parameters": "tokens_per_chunk, overlap, model"
            },
            "Sliding Window": {
                "description": "Creates overlapping chunks with sliding window",
                "best_for": "Maximum context preservation, overlapping information",
                "parameters": "window_size, step_size"
            }
        }
        
        return strategy_info.get(strategy, {
            "description": "Unknown strategy",
            "best_for": "Unknown",
            "parameters": "Unknown"
        })
