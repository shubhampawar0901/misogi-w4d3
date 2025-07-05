"""
Visualization Engine for RAG Chunking Visualizer
Creates interactive visualizations for chunk analysis.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional


class VisualizationEngine:
    """Creates interactive visualizations for chunk analysis."""
    
    def __init__(self):
        """Initialize the visualization engine."""
        self.color_palette = px.colors.qualitative.Set3
    
    def create_size_distribution_plot(self, chunks: List[str], strategy_name: str) -> go.Figure:
        """Create a chunk size distribution plot."""
        chunk_sizes = [len(chunk) for chunk in chunks]
        
        fig = px.histogram(
            x=chunk_sizes,
            nbins=20,
            title=f"Chunk Size Distribution - {strategy_name}",
            labels={'x': 'Chunk Size (characters)', 'y': 'Frequency'},
            color_discrete_sequence=[self.color_palette[0]]
        )
        
        # Add statistics annotations
        mean_size = np.mean(chunk_sizes)
        median_size = np.median(chunk_sizes)
        
        fig.add_vline(
            x=mean_size,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_size:.0f}"
        )
        
        fig.add_vline(
            x=median_size,
            line_dash="dot",
            line_color="blue",
            annotation_text=f"Median: {median_size:.0f}"
        )
        
        fig.update_layout(
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_comparison_plot(self, strategy_results: Dict[str, List[str]]) -> go.Figure:
        """Create a comparison plot for multiple strategies."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Chunk Count', 'Average Size', 'Size Consistency', 'Size Distribution'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        strategies = list(strategy_results.keys())
        colors = self.color_palette[:len(strategies)]
        
        # Chunk count
        chunk_counts = [len(chunks) for chunks in strategy_results.values()]
        fig.add_trace(
            go.Bar(x=strategies, y=chunk_counts, name="Chunk Count", marker_color=colors),
            row=1, col=1
        )
        
        # Average size
        avg_sizes = [np.mean([len(chunk) for chunk in chunks]) for chunks in strategy_results.values()]
        fig.add_trace(
            go.Bar(x=strategies, y=avg_sizes, name="Avg Size", marker_color=colors),
            row=1, col=2
        )
        
        # Size consistency (inverse of coefficient of variation)
        consistency_scores = []
        for chunks in strategy_results.values():
            chunk_sizes = [len(chunk) for chunk in chunks]
            if len(chunk_sizes) > 1 and np.mean(chunk_sizes) > 0:
                cv = np.std(chunk_sizes) / np.mean(chunk_sizes)
                consistency = 1 / (1 + cv)
            else:
                consistency = 1.0
            consistency_scores.append(consistency)
        
        fig.add_trace(
            go.Bar(x=strategies, y=consistency_scores, name="Consistency", marker_color=colors),
            row=2, col=1
        )
        
        # Size distribution (box plot)
        for i, (strategy, chunks) in enumerate(strategy_results.items()):
            chunk_sizes = [len(chunk) for chunk in chunks]
            fig.add_trace(
                go.Box(y=chunk_sizes, name=strategy, marker_color=colors[i]),
                row=2, col=2
            )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Chunking Strategy Comparison"
        )
        
        return fig
    
    def create_semantic_heatmap(self, similarity_matrix: List[List[float]], strategy_name: str) -> go.Figure:
        """Create a semantic similarity heatmap."""
        fig = px.imshow(
            similarity_matrix,
            title=f"Chunk Similarity Matrix - {strategy_name}",
            labels=dict(x="Chunk Index", y="Chunk Index", color="Similarity"),
            color_continuous_scale="RdYlBu_r"
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_embedding_scatter(self, embeddings_2d: List[List[float]], 
                                chunks: List[str], strategy_name: str) -> go.Figure:
        """Create a 2D scatter plot of chunk embeddings."""
        if not embeddings_2d or len(embeddings_2d[0]) < 2:
            return go.Figure()
        
        x_coords = [emb[0] for emb in embeddings_2d]
        y_coords = [emb[1] for emb in embeddings_2d]
        
        # Create hover text with chunk previews
        hover_texts = []
        for i, chunk in enumerate(chunks):
            preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
            hover_texts.append(f"Chunk {i+1}<br>{preview}")
        
        fig = go.Figure(data=go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=8,
                color=range(len(chunks)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Chunk Index")
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Chunk Embeddings Visualization - {strategy_name}",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            height=500
        )
        
        return fig
    
    def create_overlap_analysis_plot(self, overlap_ratios: List[float], strategy_name: str) -> go.Figure:
        """Create an overlap analysis plot."""
        chunk_pairs = [f"Chunk {i+1}-{i+2}" for i in range(len(overlap_ratios))]
        
        fig = go.Figure(data=go.Scatter(
            x=chunk_pairs,
            y=overlap_ratios,
            mode='lines+markers',
            line=dict(color=self.color_palette[0], width=2),
            marker=dict(size=8, color=self.color_palette[1])
        ))
        
        # Add average line
        avg_overlap = np.mean(overlap_ratios)
        fig.add_hline(
            y=avg_overlap,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average: {avg_overlap:.3f}"
        )
        
        fig.update_layout(
            title=f"Chunk Overlap Analysis - {strategy_name}",
            xaxis_title="Consecutive Chunk Pairs",
            yaxis_title="Overlap Ratio",
            height=400,
            xaxis_tickangle=45
        )
        
        return fig
    
    def create_quality_radar_chart(self, quality_scores: Dict[str, float], strategy_name: str) -> go.Figure:
        """Create a radar chart for quality metrics."""
        metrics = ['Size Consistency', 'Semantic Coherence', 'Size Appropriateness', 'Coverage']
        values = [
            quality_scores.get('size_consistency', 0),
            quality_scores.get('semantic_coherence', 0),
            quality_scores.get('size_appropriateness', 0),
            quality_scores.get('coverage', 0)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=strategy_name,
            line_color=self.color_palette[0]
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=f"Quality Metrics - {strategy_name}",
            height=400
        )
        
        return fig
    
    def create_multi_strategy_radar(self, strategy_quality_scores: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create a radar chart comparing multiple strategies."""
        metrics = ['Size Consistency', 'Semantic Coherence', 'Size Appropriateness', 'Coverage']
        
        fig = go.Figure()
        
        for i, (strategy, scores) in enumerate(strategy_quality_scores.items()):
            values = [
                scores.get('size_consistency', 0),
                scores.get('semantic_coherence', 0),
                scores.get('size_appropriateness', 0),
                scores.get('coverage', 0)
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=strategy,
                line_color=self.color_palette[i % len(self.color_palette)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Strategy Quality Comparison",
            height=500
        )
        
        return fig
    
    def create_chunk_length_timeline(self, chunks: List[str], strategy_name: str) -> go.Figure:
        """Create a timeline showing chunk lengths."""
        chunk_indices = list(range(1, len(chunks) + 1))
        chunk_lengths = [len(chunk) for chunk in chunks]
        
        fig = go.Figure(data=go.Scatter(
            x=chunk_indices,
            y=chunk_lengths,
            mode='lines+markers',
            line=dict(color=self.color_palette[0], width=2),
            marker=dict(size=6, color=self.color_palette[1]),
            hovertemplate='Chunk %{x}<br>Length: %{y} chars<extra></extra>'
        ))
        
        # Add average line
        avg_length = np.mean(chunk_lengths)
        fig.add_hline(
            y=avg_length,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average: {avg_length:.0f}"
        )
        
        fig.update_layout(
            title=f"Chunk Length Timeline - {strategy_name}",
            xaxis_title="Chunk Index",
            yaxis_title="Chunk Length (characters)",
            height=400
        )
        
        return fig
    
    def create_summary_dashboard(self, strategy_results: Dict[str, List[str]], 
                               analysis_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create a comprehensive summary dashboard."""
        strategies = list(strategy_results.keys())
        n_strategies = len(strategies)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Chunk Counts', 'Average Sizes', 'Size Distributions', 'Quality Scores'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "bar"}]]
        )
        
        colors = self.color_palette[:n_strategies]
        
        # Chunk counts
        chunk_counts = [len(chunks) for chunks in strategy_results.values()]
        fig.add_trace(
            go.Bar(x=strategies, y=chunk_counts, name="Chunks", marker_color=colors),
            row=1, col=1
        )
        
        # Average sizes
        avg_sizes = [np.mean([len(chunk) for chunk in chunks]) for chunks in strategy_results.values()]
        fig.add_trace(
            go.Bar(x=strategies, y=avg_sizes, name="Avg Size", marker_color=colors),
            row=1, col=2
        )
        
        # Size distributions
        for i, (strategy, chunks) in enumerate(strategy_results.items()):
            chunk_sizes = [len(chunk) for chunk in chunks]
            fig.add_trace(
                go.Box(y=chunk_sizes, name=strategy, marker_color=colors[i]),
                row=2, col=1
            )
        
        # Quality scores (if available)
        if analysis_results:
            quality_scores = []
            for strategy in strategies:
                if 'quality_score' in analysis_results.get(strategy, {}):
                    quality_scores.append(analysis_results[strategy]['quality_score'].get('overall_score', 0))
                else:
                    quality_scores.append(0)
            
            fig.add_trace(
                go.Bar(x=strategies, y=quality_scores, name="Quality", marker_color=colors),
                row=2, col=2
            )
        
        fig.update_layout(
            height=700,
            showlegend=False,
            title_text="Chunking Strategy Analysis Dashboard"
        )
        
        return fig
