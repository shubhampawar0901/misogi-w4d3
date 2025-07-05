"""
Embedding Models Manager for Plagiarism Detection
Handles different embedding models for text similarity analysis.
"""

import numpy as np
import os
from typing import List, Dict, Optional
import streamlit as st
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI
import time


class EmbeddingModelManager:
    """Manages different embedding models for text similarity analysis."""
    
    def __init__(self):
        """Initialize the embedding model manager."""
        self.sentence_transformers_cache = {}
        self.openai_client = None
        self._initialize_openai()
    
    def _initialize_openai(self):
        """Initialize OpenAI client if API key is available."""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_openai_api_key_here":
            try:
                self.openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                st.warning(f"Failed to initialize OpenAI client: {e}")
    
    def get_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """
        Get embeddings for a list of texts using the specified model.
        
        Args:
            texts: List of texts to embed
            model_name: Name of the embedding model to use
            
        Returns:
            NumPy array of embeddings
        """
        if model_name.startswith("openai-"):
            return self._get_openai_embeddings(texts, model_name)
        else:
            return self._get_sentence_transformer_embeddings(texts, model_name)
    
    def _get_sentence_transformer_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Get embeddings using Sentence Transformers."""
        try:
            # Load model (with caching)
            if model_name not in self.sentence_transformers_cache:
                with st.spinner(f"Loading {model_name} model..."):
                    self.sentence_transformers_cache[model_name] = SentenceTransformer(model_name)
            
            model = self.sentence_transformers_cache[model_name]
            
            # Generate embeddings
            with st.spinner(f"Generating embeddings with {model_name}..."):
                embeddings = model.encode(texts, convert_to_numpy=True)
            
            return embeddings
            
        except Exception as e:
            st.error(f"Error with Sentence Transformer model {model_name}: {e}")
            # Fallback to a simple model
            if model_name != "all-MiniLM-L6-v2":
                st.warning("Falling back to all-MiniLM-L6-v2 model")
                return self._get_sentence_transformer_embeddings(texts, "all-MiniLM-L6-v2")
            else:
                raise e
    
    def _get_openai_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Get embeddings using OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in your environment.")
        
        try:
            # Map model names to OpenAI model identifiers
            model_mapping = {
                "openai-text-embedding-3-small": "text-embedding-3-small",
                "openai-text-embedding-ada-002": "text-embedding-ada-002"
            }
            
            openai_model = model_mapping.get(model_name, "text-embedding-3-small")
            
            embeddings = []
            
            with st.spinner(f"Generating embeddings with {model_name}..."):
                for i, text in enumerate(texts):
                    # Rate limiting for OpenAI API
                    if i > 0:
                        time.sleep(0.1)  # Small delay between requests
                    
                    response = self.openai_client.embeddings.create(
                        model=openai_model,
                        input=text
                    )
                    
                    embeddings.append(response.data[0].embedding)
            
            return np.array(embeddings)
            
        except Exception as e:
            st.error(f"Error with OpenAI model {model_name}: {e}")
            raise e
    
    def get_model_info(self, model_name: str) -> Dict[str, str]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        model_info = {
            "all-MiniLM-L6-v2": {
                "description": "Lightweight, fast model good for general similarity tasks",
                "dimensions": "384",
                "provider": "Sentence Transformers",
                "speed": "Fast",
                "quality": "Good"
            },
            "all-mpnet-base-v2": {
                "description": "High-quality model with better semantic understanding",
                "dimensions": "768", 
                "provider": "Sentence Transformers",
                "speed": "Medium",
                "quality": "Excellent"
            },
            "paraphrase-MiniLM-L6-v2": {
                "description": "Optimized for paraphrase detection and similarity",
                "dimensions": "384",
                "provider": "Sentence Transformers", 
                "speed": "Fast",
                "quality": "Good"
            },
            "openai-text-embedding-3-small": {
                "description": "OpenAI's latest small embedding model",
                "dimensions": "1536",
                "provider": "OpenAI",
                "speed": "Medium",
                "quality": "Excellent"
            },
            "openai-text-embedding-ada-002": {
                "description": "OpenAI's previous generation embedding model",
                "dimensions": "1536", 
                "provider": "OpenAI",
                "speed": "Medium",
                "quality": "Very Good"
            }
        }
        
        return model_info.get(model_name, {
            "description": "Unknown model",
            "dimensions": "Unknown",
            "provider": "Unknown",
            "speed": "Unknown", 
            "quality": "Unknown"
        })
    
    def get_available_models(self) -> List[str]:
        """Get list of available embedding models."""
        sentence_transformer_models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-MiniLM-L6-v2"
        ]
        
        openai_models = []
        if self.openai_client:
            openai_models = [
                "openai-text-embedding-3-small",
                "openai-text-embedding-ada-002"
            ]
        
        return sentence_transformer_models + openai_models
    
    def validate_model(self, model_name: str) -> tuple[bool, str]:
        """
        Validate if a model is available and can be used.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        available_models = self.get_available_models()
        
        if model_name not in available_models:
            return False, f"Model {model_name} is not available"
        
        if model_name.startswith("openai-") and not self.openai_client:
            return False, "OpenAI API key not configured for OpenAI models"
        
        return True, ""
    
    def benchmark_model(self, model_name: str, sample_texts: List[str]) -> Dict[str, float]:
        """
        Benchmark a model's performance on sample texts.
        
        Args:
            model_name: Name of the model to benchmark
            sample_texts: Sample texts for benchmarking
            
        Returns:
            Dictionary with benchmark results
        """
        try:
            start_time = time.time()
            embeddings = self.get_embeddings(sample_texts, model_name)
            end_time = time.time()
            
            processing_time = end_time - start_time
            texts_per_second = len(sample_texts) / processing_time if processing_time > 0 else 0
            
            return {
                "processing_time": processing_time,
                "texts_per_second": texts_per_second,
                "embedding_dimensions": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
                "total_texts": len(sample_texts)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "processing_time": 0,
                "texts_per_second": 0,
                "embedding_dimensions": 0,
                "total_texts": 0
            }
    
    def clear_cache(self):
        """Clear the model cache to free memory."""
        self.sentence_transformers_cache.clear()
        st.success("Model cache cleared successfully!")
