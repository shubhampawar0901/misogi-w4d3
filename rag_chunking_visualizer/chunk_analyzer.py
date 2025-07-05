"""
Chunk Analyzer for RAG Chunking Visualizer
Analyzes chunk quality and characteristics.
"""

import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import umap


class ChunkAnalyzer:
    """Analyzes chunk quality and characteristics."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the chunk analyzer.
        
        Args:
            embedding_model_name: Name of the embedding model to use
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
    
    def analyze_chunks(self, chunks: List[str], analysis_options: List[str]) -> Dict[str, Any]:
        """
        Analyze a list of chunks.
        
        Args:
            chunks: List of text chunks to analyze
            analysis_options: List of analysis types to perform
            
        Returns:
            Dictionary with analysis results
        """
        if not chunks:
            return {}
        
        results = {}
        
        # Basic statistics
        results['basic_stats'] = self._calculate_basic_stats(chunks)
        
        # Perform requested analyses
        if "Chunk Size Distribution" in analysis_options:
            results['size_distribution'] = self._analyze_size_distribution(chunks)
        
        if "Semantic Coherence" in analysis_options:
            results['semantic_coherence'] = self._analyze_semantic_coherence(chunks)
        
        if "Overlap Analysis" in analysis_options:
            results['overlap_analysis'] = self._analyze_overlap(chunks)
        
        if "Embedding Visualization" in analysis_options:
            results['embedding_visualization'] = self._generate_embedding_visualization(chunks)
        
        return results
    
    def _calculate_basic_stats(self, chunks: List[str]) -> Dict[str, Any]:
        """Calculate basic statistics for chunks."""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk) for chunk in chunks]
        word_counts = [len(chunk.split()) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_characters': sum(chunk_lengths),
            'total_words': sum(word_counts),
            'avg_chunk_length': np.mean(chunk_lengths),
            'median_chunk_length': np.median(chunk_lengths),
            'std_chunk_length': np.std(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'avg_word_count': np.mean(word_counts),
            'median_word_count': np.median(word_counts),
            'std_word_count': np.std(word_counts)
        }
    
    def _analyze_size_distribution(self, chunks: List[str]) -> Dict[str, Any]:
        """Analyze chunk size distribution."""
        chunk_lengths = [len(chunk) for chunk in chunks]
        
        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_values = [np.percentile(chunk_lengths, p) for p in percentiles]
        
        # Calculate histogram data
        hist, bin_edges = np.histogram(chunk_lengths, bins=20)
        
        return {
            'percentiles': dict(zip(percentiles, percentile_values)),
            'histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            },
            'outliers': self._find_size_outliers(chunk_lengths)
        }
    
    def _analyze_semantic_coherence(self, chunks: List[str]) -> float:
        """Analyze semantic coherence of chunks."""
        if len(chunks) < 2:
            return 1.0
        
        # Initialize embedding model if needed
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        try:
            # Generate embeddings for chunks
            embeddings = self.embedding_model.encode(chunks)
            
            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(embeddings)
            
            # Calculate average similarity (excluding diagonal)
            n = len(chunks)
            total_similarity = 0
            count = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    total_similarity += similarity_matrix[i][j]
                    count += 1
            
            if count == 0:
                return 1.0
            
            avg_similarity = total_similarity / count
            return float(avg_similarity)
            
        except Exception as e:
            print(f"Error calculating semantic coherence: {e}")
            return 0.0
    
    def _analyze_overlap(self, chunks: List[str]) -> Dict[str, Any]:
        """Analyze overlap between consecutive chunks."""
        if len(chunks) < 2:
            return {'avg_overlap': 0, 'overlap_ratios': []}
        
        overlap_ratios = []
        
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]
            
            # Find common substrings
            overlap_ratio = self._calculate_text_overlap(chunk1, chunk2)
            overlap_ratios.append(overlap_ratio)
        
        return {
            'avg_overlap': np.mean(overlap_ratios),
            'median_overlap': np.median(overlap_ratios),
            'std_overlap': np.std(overlap_ratios),
            'overlap_ratios': overlap_ratios
        }
    
    def _generate_embedding_visualization(self, chunks: List[str]) -> Dict[str, Any]:
        """Generate data for embedding visualization."""
        if len(chunks) < 2:
            return {}
        
        # Initialize embedding model if needed
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(chunks)
            
            # Reduce dimensionality for visualization
            visualization_data = {}
            
            # PCA reduction
            if len(chunks) >= 3:
                pca = PCA(n_components=min(3, len(chunks), embeddings.shape[1]))
                pca_embeddings = pca.fit_transform(embeddings)
                
                visualization_data['pca'] = {
                    'embeddings': pca_embeddings.tolist(),
                    'explained_variance': pca.explained_variance_ratio_.tolist()
                }
            
            # UMAP reduction (if enough samples)
            if len(chunks) >= 5:
                try:
                    umap_reducer = umap.UMAP(n_components=2, random_state=42)
                    umap_embeddings = umap_reducer.fit_transform(embeddings)
                    
                    visualization_data['umap'] = {
                        'embeddings': umap_embeddings.tolist()
                    }
                except Exception as e:
                    print(f"UMAP reduction failed: {e}")
            
            # Calculate similarity matrix for heatmap
            similarity_matrix = cosine_similarity(embeddings)
            visualization_data['similarity_matrix'] = similarity_matrix.tolist()
            
            return visualization_data
            
        except Exception as e:
            print(f"Error generating embedding visualization: {e}")
            return {}
    
    def _find_size_outliers(self, chunk_lengths: List[int]) -> Dict[str, List[int]]:
        """Find outliers in chunk sizes."""
        if len(chunk_lengths) < 4:
            return {'small_outliers': [], 'large_outliers': []}
        
        q1 = np.percentile(chunk_lengths, 25)
        q3 = np.percentile(chunk_lengths, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        small_outliers = [i for i, length in enumerate(chunk_lengths) if length < lower_bound]
        large_outliers = [i for i, length in enumerate(chunk_lengths) if length > upper_bound]
        
        return {
            'small_outliers': small_outliers,
            'large_outliers': large_outliers,
            'bounds': {'lower': lower_bound, 'upper': upper_bound}
        }
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """Calculate overlap ratio between two texts."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based overlap calculation
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def compare_strategies(self, strategy_results: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compare multiple chunking strategies."""
        if not strategy_results:
            return {}
        
        comparison = {}
        
        for strategy_name, chunks in strategy_results.items():
            if not chunks:
                continue
            
            # Basic metrics
            chunk_lengths = [len(chunk) for chunk in chunks]
            
            comparison[strategy_name] = {
                'total_chunks': len(chunks),
                'avg_chunk_size': np.mean(chunk_lengths),
                'std_chunk_size': np.std(chunk_lengths),
                'consistency_score': 1 / (1 + np.std(chunk_lengths) / np.mean(chunk_lengths)) if np.mean(chunk_lengths) > 0 else 0,
                'semantic_coherence': self._analyze_semantic_coherence(chunks)
            }
        
        # Find best strategies
        if comparison:
            best_consistency = max(comparison.keys(), key=lambda k: comparison[k]['consistency_score'])
            best_coherence = max(comparison.keys(), key=lambda k: comparison[k]['semantic_coherence'])
            
            comparison['recommendations'] = {
                'most_consistent': best_consistency,
                'most_coherent': best_coherence
            }
        
        return comparison
    
    def generate_quality_score(self, chunks: List[str]) -> Dict[str, float]:
        """Generate an overall quality score for chunks."""
        if not chunks:
            return {'overall_score': 0.0}
        
        scores = {}
        
        # Size consistency score (0-1, higher is better)
        chunk_lengths = [len(chunk) for chunk in chunks]
        if len(chunk_lengths) > 1:
            cv = np.std(chunk_lengths) / np.mean(chunk_lengths)  # Coefficient of variation
            scores['size_consistency'] = 1 / (1 + cv)
        else:
            scores['size_consistency'] = 1.0
        
        # Semantic coherence score (0-1, higher is better)
        scores['semantic_coherence'] = self._analyze_semantic_coherence(chunks)
        
        # Size appropriateness score (0-1, higher is better)
        avg_length = np.mean(chunk_lengths)
        ideal_length = 500  # Ideal chunk size for RAG
        size_diff = abs(avg_length - ideal_length) / ideal_length
        scores['size_appropriateness'] = 1 / (1 + size_diff)
        
        # Coverage score (0-1, higher is better)
        total_chars = sum(chunk_lengths)
        if total_chars > 0:
            scores['coverage'] = min(1.0, total_chars / 10000)  # Normalize by expected document size
        else:
            scores['coverage'] = 0.0
        
        # Overall score (weighted average)
        weights = {
            'size_consistency': 0.3,
            'semantic_coherence': 0.4,
            'size_appropriateness': 0.2,
            'coverage': 0.1
        }
        
        overall_score = sum(scores[metric] * weights[metric] for metric in scores)
        scores['overall_score'] = overall_score
        
        return scores
