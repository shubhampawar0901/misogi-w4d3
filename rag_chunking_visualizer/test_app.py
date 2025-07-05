#!/usr/bin/env python3
"""
Test script for RAG Chunking Strategy Visualizer
Validates core functionality without Streamlit interface.
"""

import sys
import os
import numpy as np
from typing import List

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_pdf_processor():
    """Test the PDF processor functionality."""
    print("🧪 Testing PDF Processor...")
    
    try:
        from pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        
        # Test text cleaning
        dirty_text = "  This   is    a   test   text  with  extra   spaces.  "
        cleaned = processor._clean_text(dirty_text)
        print(f"Cleaned text: '{cleaned}'")
        
        # Test paragraph extraction
        sample_text = """
        This is the first paragraph.
        It has multiple sentences.
        
        This is the second paragraph.
        It also has content.
        
        
        This is the third paragraph.
        """
        
        paragraphs = processor._extract_paragraphs(sample_text)
        print(f"Extracted {len(paragraphs)} paragraphs")
        
        # Test sentence extraction
        sentences = processor._extract_sentences(sample_text)
        print(f"Extracted {len(sentences)} sentences")
        
        print("✅ PDF Processor tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ PDF Processor test failed: {e}")
        return False

def test_chunking_strategies():
    """Test the chunking strategies functionality."""
    print("\n🧪 Testing Chunking Strategies...")
    
    try:
        from chunking_strategies import ChunkingStrategyManager
        
        manager = ChunkingStrategyManager()
        
        # Sample text for testing
        sample_text = """
        This is a sample document for testing chunking strategies. It contains multiple sentences and paragraphs to demonstrate different approaches to text segmentation.
        
        The first paragraph introduces the concept of chunking. Chunking is the process of dividing large texts into smaller, manageable pieces that can be processed more effectively by language models and retrieval systems.
        
        The second paragraph discusses the importance of chunking in RAG systems. Proper chunking ensures that relevant information can be retrieved accurately while maintaining context and coherence.
        
        The third paragraph explores different chunking strategies. Each strategy has its own advantages and is suitable for different types of content and use cases.
        """
        
        # Test fixed size chunking
        fixed_chunks = manager.apply_strategy(sample_text, "Fixed Size", {"chunk_size": 200, "overlap": 50})
        print(f"Fixed Size: {len(fixed_chunks)} chunks")
        
        # Test sentence-based chunking
        sentence_chunks = manager.apply_strategy(sample_text, "Sentence-based", {"sentences_per_chunk": 2, "overlap": 1})
        print(f"Sentence-based: {len(sentence_chunks)} chunks")
        
        # Test paragraph-based chunking
        paragraph_chunks = manager.apply_strategy(sample_text, "Paragraph-based", {"min_chunk_size": 50, "max_chunk_size": 300})
        print(f"Paragraph-based: {len(paragraph_chunks)} chunks")
        
        # Test recursive character chunking
        recursive_chunks = manager.apply_strategy(sample_text, "Recursive Character", {"chunk_size": 250, "overlap": 30})
        print(f"Recursive Character: {len(recursive_chunks)} chunks")
        
        # Test sliding window chunking
        sliding_chunks = manager.apply_strategy(sample_text, "Sliding Window", {"window_size": 200, "step_size": 100})
        print(f"Sliding Window: {len(sliding_chunks)} chunks")
        
        # Test strategy info
        info = manager.get_strategy_info("Fixed Size")
        print(f"Strategy info: {info['description']}")
        
        print("✅ Chunking Strategies tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Chunking Strategies test failed: {e}")
        return False

def test_chunk_analyzer():
    """Test the chunk analyzer functionality."""
    print("\n🧪 Testing Chunk Analyzer...")
    
    try:
        from chunk_analyzer import ChunkAnalyzer
        
        analyzer = ChunkAnalyzer()
        
        # Sample chunks for testing
        sample_chunks = [
            "This is the first chunk about machine learning and artificial intelligence.",
            "The second chunk discusses natural language processing and text analysis.",
            "This third chunk covers deep learning and neural networks in detail.",
            "The fourth chunk explores computer vision and image recognition systems."
        ]
        
        # Test basic stats
        basic_stats = analyzer._calculate_basic_stats(sample_chunks)
        print(f"Basic stats: {basic_stats['total_chunks']} chunks, avg length: {basic_stats['avg_chunk_length']:.1f}")
        
        # Test size distribution analysis
        size_dist = analyzer._analyze_size_distribution(sample_chunks)
        print(f"Size distribution: {len(size_dist['percentiles'])} percentiles calculated")
        
        # Test overlap analysis
        overlap_analysis = analyzer._analyze_overlap(sample_chunks)
        print(f"Overlap analysis: avg overlap {overlap_analysis['avg_overlap']:.3f}")
        
        # Test quality score generation
        quality_scores = analyzer.generate_quality_score(sample_chunks)
        print(f"Quality scores: overall {quality_scores['overall_score']:.3f}")
        
        print("✅ Chunk Analyzer tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Chunk Analyzer test failed: {e}")
        print("Note: This might fail if sentence-transformers is not installed")
        return False

def test_visualization_engine():
    """Test the visualization engine functionality."""
    print("\n🧪 Testing Visualization Engine...")
    
    try:
        from visualization_engine import VisualizationEngine
        
        visualizer = VisualizationEngine()
        
        # Sample data for testing
        sample_chunks = [
            "Short chunk.",
            "This is a medium-length chunk with more content.",
            "This is a longer chunk that contains significantly more text and information for testing purposes.",
            "Another medium chunk.",
            "Final short chunk."
        ]
        
        # Test size distribution plot
        size_plot = visualizer.create_size_distribution_plot(sample_chunks, "Test Strategy")
        print(f"Size distribution plot created: {type(size_plot)}")
        
        # Test comparison plot
        strategy_results = {
            "Strategy A": sample_chunks,
            "Strategy B": sample_chunks[:3]
        }
        comparison_plot = visualizer.create_comparison_plot(strategy_results)
        print(f"Comparison plot created: {type(comparison_plot)}")
        
        # Test chunk length timeline
        timeline_plot = visualizer.create_chunk_length_timeline(sample_chunks, "Test Strategy")
        print(f"Timeline plot created: {type(timeline_plot)}")
        
        # Test quality radar chart
        quality_scores = {
            'size_consistency': 0.8,
            'semantic_coherence': 0.7,
            'size_appropriateness': 0.9,
            'coverage': 0.85
        }
        radar_plot = visualizer.create_quality_radar_chart(quality_scores, "Test Strategy")
        print(f"Radar chart created: {type(radar_plot)}")
        
        print("✅ Visualization Engine tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Visualization Engine test failed: {e}")
        return False

def test_integration():
    """Test integration of all components."""
    print("\n🧪 Testing Integration...")
    
    try:
        from chunking_strategies import ChunkingStrategyManager
        from chunk_analyzer import ChunkAnalyzer
        from visualization_engine import VisualizationEngine
        
        # Sample document
        document = """
        Introduction to Machine Learning
        
        Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience.
        
        Types of Machine Learning
        
        There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Each type has its own characteristics and applications.
        
        Supervised Learning
        
        Supervised learning involves training a model on a labeled dataset, where the correct output is known for each input. Common examples include classification and regression tasks.
        
        Unsupervised Learning
        
        Unsupervised learning deals with finding patterns in data without labeled examples. Clustering and dimensionality reduction are typical unsupervised learning tasks.
        
        Applications
        
        Machine learning has numerous applications across various industries, including healthcare, finance, transportation, and entertainment. The technology continues to evolve and find new use cases.
        """
        
        # Initialize components
        strategy_manager = ChunkingStrategyManager()
        analyzer = ChunkAnalyzer()
        visualizer = VisualizationEngine()
        
        print(f"Document length: {len(document)} characters")
        
        # Apply different chunking strategies
        strategies = ["Fixed Size", "Sentence-based", "Paragraph-based"]
        strategy_params = {
            "Fixed Size": {"chunk_size": 300, "overlap": 50},
            "Sentence-based": {"sentences_per_chunk": 2, "overlap": 1},
            "Paragraph-based": {"min_chunk_size": 100, "max_chunk_size": 500}
        }
        
        chunking_results = {}
        for strategy in strategies:
            params = strategy_params[strategy]
            chunks = strategy_manager.apply_strategy(document, strategy, params)
            chunking_results[strategy] = chunks
            print(f"{strategy}: {len(chunks)} chunks")
        
        # Analyze chunks
        analysis_results = {}
        for strategy, chunks in chunking_results.items():
            basic_stats = analyzer._calculate_basic_stats(chunks)
            quality_scores = analyzer.generate_quality_score(chunks)
            
            analysis_results[strategy] = {
                'basic_stats': basic_stats,
                'quality_scores': quality_scores
            }
            
            print(f"{strategy} quality score: {quality_scores['overall_score']:.3f}")
        
        # Create visualizations
        comparison_plot = visualizer.create_comparison_plot(chunking_results)
        print(f"Comparison visualization created: {type(comparison_plot)}")
        
        # Strategy comparison
        comparison = analyzer.compare_strategies(chunking_results)
        if 'recommendations' in comparison:
            print(f"Most consistent strategy: {comparison['recommendations']['most_consistent']}")
        
        print("✅ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 RAG Chunking Strategy Visualizer Test Suite")
    print("=" * 60)
    
    tests = [
        test_pdf_processor,
        test_chunking_strategies,
        test_chunk_analyzer,
        test_visualization_engine,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
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
