#!/usr/bin/env python3
"""
Test script for Plagiarism Detector
Validates core functionality without Streamlit interface.
"""

import sys
import os
import numpy as np
from typing import List

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_text_preprocessor():
    """Test the text preprocessor functionality."""
    print("🧪 Testing Text Preprocessor...")
    
    try:
        from text_preprocessor import TextPreprocessor
        
        # Test basic preprocessing
        preprocessor = TextPreprocessor(["Remove extra whitespace", "Convert to lowercase"])
        
        test_text = "  This   is    A   TEST   text!!!  "
        processed = preprocessor.preprocess(test_text)
        
        print(f"Original: '{test_text}'")
        print(f"Processed: '{processed}'")
        
        # Test validation
        is_valid, error = preprocessor.validate_text("Hello world test")
        print(f"Validation: {is_valid}, Error: {error}")
        
        # Test stats
        stats = preprocessor.get_text_stats("Hello world. This is a test!")
        print(f"Stats: {stats}")
        
        print("✅ Text Preprocessor tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Text Preprocessor test failed: {e}")
        return False

def test_similarity_analyzer():
    """Test the similarity analyzer functionality."""
    print("\n🧪 Testing Similarity Analyzer...")
    
    try:
        from similarity_analyzer import SimilarityAnalyzer
        
        analyzer = SimilarityAnalyzer()
        
        # Create sample embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # Text 1
            [0.9, 0.1, 0.0],  # Text 2 (similar to 1)
            [0.0, 1.0, 0.0],  # Text 3 (different)
            [0.0, 0.9, 0.1]   # Text 4 (similar to 3)
        ])
        
        # Test similarity matrix
        similarity_matrix = analyzer.calculate_similarity_matrix(embeddings)
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print(f"Sample similarity: {similarity_matrix[0][1]:.3f}")
        
        # Test clone detection
        clones = analyzer.find_clones(similarity_matrix, threshold=0.8)
        print(f"Clones detected: {len(clones)}")
        
        # Test statistics
        stats = analyzer.calculate_similarity_statistics(similarity_matrix)
        print(f"Mean similarity: {stats['mean_similarity']:.3f}")
        
        print("✅ Similarity Analyzer tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Similarity Analyzer test failed: {e}")
        return False

def test_embedding_models():
    """Test the embedding models functionality."""
    print("\n🧪 Testing Embedding Models...")
    
    try:
        from embedding_models import EmbeddingModelManager
        
        manager = EmbeddingModelManager()
        
        # Test model info
        model_info = manager.get_model_info("all-MiniLM-L6-v2")
        print(f"Model info: {model_info['description']}")
        
        # Test available models
        available = manager.get_available_models()
        print(f"Available models: {len(available)}")
        
        # Test model validation
        is_valid, error = manager.validate_model("all-MiniLM-L6-v2")
        print(f"Model validation: {is_valid}")
        
        # Test embeddings with a simple example
        sample_texts = [
            "The cat sat on the mat.",
            "A feline rested on the rug.",
            "The weather is nice today."
        ]
        
        print("Generating embeddings...")
        embeddings = manager.get_embeddings(sample_texts, "all-MiniLM-L6-v2")
        print(f"Embeddings shape: {embeddings.shape}")
        
        print("✅ Embedding Models tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Embedding Models test failed: {e}")
        print("Note: This might fail if sentence-transformers is not installed")
        return False

def test_integration():
    """Test integration of all components."""
    print("\n🧪 Testing Integration...")
    
    try:
        from text_preprocessor import TextPreprocessor
        from embedding_models import EmbeddingModelManager
        from similarity_analyzer import SimilarityAnalyzer
        
        # Sample texts for testing
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast brown fox leaps over a sleepy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "AI includes machine learning as one of its components."
        ]
        
        # Initialize components
        preprocessor = TextPreprocessor(["Remove extra whitespace"])
        model_manager = EmbeddingModelManager()
        analyzer = SimilarityAnalyzer()
        
        # Process texts
        processed_texts = [preprocessor.preprocess(text) for text in texts]
        print(f"Processed {len(processed_texts)} texts")
        
        # Generate embeddings
        embeddings = model_manager.get_embeddings(processed_texts, "all-MiniLM-L6-v2")
        print(f"Generated embeddings: {embeddings.shape}")
        
        # Analyze similarity
        similarity_matrix = analyzer.calculate_similarity_matrix(embeddings)
        clones = analyzer.find_clones(similarity_matrix, threshold=0.7)
        
        print(f"Similarity matrix: {similarity_matrix.shape}")
        print(f"Potential clones: {len(clones)}")
        
        # Display results
        for i, (idx1, idx2, similarity) in enumerate(clones):
            print(f"Clone {i+1}: Text {idx1+1} ↔ Text {idx2+1} ({similarity:.1%})")
        
        print("✅ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Plagiarism Detector Test Suite")
    print("=" * 50)
    
    tests = [
        test_text_preprocessor,
        test_similarity_analyzer,
        test_embedding_models,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to use.")
        print("\nTo run the web application:")
        print("streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nMake sure you have installed all dependencies:")
        print("pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
