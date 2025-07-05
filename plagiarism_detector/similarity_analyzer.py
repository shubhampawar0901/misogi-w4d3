"""
Similarity Analyzer for Plagiarism Detection
Handles similarity calculations and clone detection.
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class SimilarityAnalyzer:
    """Analyzes text similarity and detects potential clones."""
    
    def __init__(self):
        """Initialize the similarity analyzer."""
        pass
    
    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise cosine similarity matrix for embeddings.
        
        Args:
            embeddings: NumPy array of text embeddings
            
        Returns:
            Similarity matrix as NumPy array
        """
        if len(embeddings) == 0:
            return np.array([])
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix
    
    def find_clones(self, similarity_matrix: np.ndarray, threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """
        Find potential clones based on similarity threshold.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            threshold: Similarity threshold for clone detection
            
        Returns:
            List of tuples (index1, index2, similarity_score)
        """
        clones = []
        n = similarity_matrix.shape[0]
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity >= threshold:
                    clones.append((i, j, similarity))
        
        # Sort by similarity score (highest first)
        clones.sort(key=lambda x: x[2], reverse=True)
        
        return clones
    
    def calculate_similarity_statistics(self, similarity_matrix: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics for the similarity matrix.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            
        Returns:
            Dictionary with similarity statistics
        """
        if similarity_matrix.size == 0:
            return {
                'mean_similarity': 0.0,
                'median_similarity': 0.0,
                'std_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0
            }
        
        # Get upper triangle (excluding diagonal) for pairwise similarities
        n = similarity_matrix.shape[0]
        upper_triangle = []
        
        for i in range(n):
            for j in range(i + 1, n):
                upper_triangle.append(similarity_matrix[i][j])
        
        if not upper_triangle:
            return {
                'mean_similarity': 0.0,
                'median_similarity': 0.0,
                'std_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0
            }
        
        similarities = np.array(upper_triangle)
        
        return {
            'mean_similarity': float(np.mean(similarities)),
            'median_similarity': float(np.median(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities))
        }
    
    def get_similarity_distribution(self, similarity_matrix: np.ndarray, bins: int = 10) -> Dict[str, List]:
        """
        Get distribution of similarity scores.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            bins: Number of bins for histogram
            
        Returns:
            Dictionary with bin edges and counts
        """
        if similarity_matrix.size == 0:
            return {'bin_edges': [], 'counts': []}
        
        # Get upper triangle similarities
        n = similarity_matrix.shape[0]
        similarities = []
        
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(similarity_matrix[i][j])
        
        if not similarities:
            return {'bin_edges': [], 'counts': []}
        
        counts, bin_edges = np.histogram(similarities, bins=bins, range=(0, 1))
        
        return {
            'bin_edges': bin_edges.tolist(),
            'counts': counts.tolist()
        }
    
    def rank_similarities(self, similarity_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Rank all text pairs by similarity score.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            
        Returns:
            List of tuples (index1, index2, similarity_score) sorted by similarity
        """
        pairs = []
        n = similarity_matrix.shape[0]
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                pairs.append((i, j, similarity))
        
        # Sort by similarity score (highest first)
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs
    
    def detect_clusters(self, similarity_matrix: np.ndarray, threshold: float = 0.7) -> List[List[int]]:
        """
        Detect clusters of similar texts.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            threshold: Similarity threshold for clustering
            
        Returns:
            List of clusters, where each cluster is a list of text indices
        """
        n = similarity_matrix.shape[0]
        visited = [False] * n
        clusters = []
        
        def dfs(node: int, cluster: List[int]):
            """Depth-first search to find connected components."""
            visited[node] = True
            cluster.append(node)
            
            for neighbor in range(n):
                if not visited[neighbor] and similarity_matrix[node][neighbor] >= threshold:
                    dfs(neighbor, cluster)
        
        for i in range(n):
            if not visited[i]:
                cluster = []
                dfs(i, cluster)
                if len(cluster) > 1:  # Only include clusters with multiple texts
                    clusters.append(cluster)
        
        return clusters
    
    def calculate_text_uniqueness(self, similarity_matrix: np.ndarray, text_index: int) -> float:
        """
        Calculate uniqueness score for a specific text.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            text_index: Index of the text to analyze
            
        Returns:
            Uniqueness score (0 = not unique, 1 = completely unique)
        """
        if similarity_matrix.size == 0 or text_index >= similarity_matrix.shape[0]:
            return 1.0
        
        # Get similarities with all other texts
        similarities = []
        n = similarity_matrix.shape[0]
        
        for i in range(n):
            if i != text_index:
                similarities.append(similarity_matrix[text_index][i])
        
        if not similarities:
            return 1.0
        
        # Uniqueness is inverse of maximum similarity with other texts
        max_similarity = max(similarities)
        uniqueness = 1.0 - max_similarity
        
        return max(0.0, uniqueness)  # Ensure non-negative
    
    def generate_similarity_report(self, similarity_matrix: np.ndarray, texts: List[str], 
                                 threshold: float = 0.8) -> Dict:
        """
        Generate comprehensive similarity analysis report.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            texts: List of original texts
            threshold: Similarity threshold for clone detection
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        clones = self.find_clones(similarity_matrix, threshold)
        stats = self.calculate_similarity_statistics(similarity_matrix)
        clusters = self.detect_clusters(similarity_matrix, threshold * 0.9)  # Slightly lower threshold for clustering
        
        # Calculate uniqueness for each text
        uniqueness_scores = []
        for i in range(len(texts)):
            uniqueness = self.calculate_text_uniqueness(similarity_matrix, i)
            uniqueness_scores.append(uniqueness)
        
        # Identify most and least unique texts
        most_unique_idx = np.argmax(uniqueness_scores) if uniqueness_scores else 0
        least_unique_idx = np.argmin(uniqueness_scores) if uniqueness_scores else 0
        
        return {
            'total_texts': len(texts),
            'total_comparisons': len(texts) * (len(texts) - 1) // 2,
            'clones_detected': len(clones),
            'clone_pairs': clones,
            'similarity_stats': stats,
            'clusters': clusters,
            'uniqueness_scores': uniqueness_scores,
            'most_unique_text': {
                'index': most_unique_idx,
                'score': uniqueness_scores[most_unique_idx] if uniqueness_scores else 0,
                'preview': texts[most_unique_idx][:100] + "..." if len(texts[most_unique_idx]) > 100 else texts[most_unique_idx]
            },
            'least_unique_text': {
                'index': least_unique_idx,
                'score': uniqueness_scores[least_unique_idx] if uniqueness_scores else 0,
                'preview': texts[least_unique_idx][:100] + "..." if len(texts[least_unique_idx]) > 100 else texts[least_unique_idx]
            },
            'threshold_used': threshold
        }
