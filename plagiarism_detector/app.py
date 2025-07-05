"""
Plagiarism Detector - Semantic Similarity Analyzer
A web application for detecting potential plagiarism using semantic similarity analysis.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
import time

# Load environment variables
load_dotenv()

# Import our custom modules (will be created)
try:
    from embedding_models import EmbeddingModelManager
    from similarity_analyzer import SimilarityAnalyzer
    from text_preprocessor import TextPreprocessor
except ImportError:
    st.error("Required modules not found. Please ensure all files are present.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .similarity-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 5px 0;
    }
    .similarity-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 10px;
        margin: 5px 0;
    }
    .similarity-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 10px;
        margin: 5px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'texts' not in st.session_state:
        st.session_state.texts = []
    if 'similarity_results' not in st.session_state:
        st.session_state.similarity_results = None
    if 'model_comparison' not in st.session_state:
        st.session_state.model_comparison = None

def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Plagiarism Detector</h1>', unsafe_allow_html=True)
    st.markdown("**Semantic Similarity Analyzer for Text Comparison**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        available_models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2", 
            "paraphrase-MiniLM-L6-v2",
            "openai-text-embedding-3-small",
            "openai-text-embedding-ada-002"
        ]
        
        selected_models = st.multiselect(
            "Select Embedding Models",
            available_models,
            default=["all-MiniLM-L6-v2"],
            help="Choose one or more models for comparison"
        )
        
        # Similarity threshold
        similarity_threshold = st.slider(
            "Similarity Threshold (%)",
            min_value=50,
            max_value=95,
            value=int(float(os.getenv("SIMILARITY_THRESHOLD", "0.8")) * 100),
            help="Texts above this threshold are considered potential clones"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            max_length = st.number_input(
                "Max Text Length",
                min_value=100,
                max_value=50000,
                value=int(os.getenv("MAX_TEXT_LENGTH", "10000")),
                help="Maximum characters per text input"
            )
            
            preprocessing_options = st.multiselect(
                "Text Preprocessing",
                ["Remove extra whitespace", "Convert to lowercase", "Remove special characters"],
                default=["Remove extra whitespace"],
                help="Select preprocessing steps"
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Input")
        
        # Dynamic text input
        num_texts = st.number_input(
            "Number of texts to compare",
            min_value=2,
            max_value=10,
            value=len(st.session_state.texts) if st.session_state.texts else 3,
            help="Add multiple texts for comparison"
        )
        
        # Ensure we have the right number of text areas
        while len(st.session_state.texts) < num_texts:
            st.session_state.texts.append("")
        while len(st.session_state.texts) > num_texts:
            st.session_state.texts.pop()
        
        # Text input areas
        for i in range(num_texts):
            st.session_state.texts[i] = st.text_area(
                f"Text {i+1}",
                value=st.session_state.texts[i],
                height=150,
                max_chars=max_length,
                key=f"text_{i}",
                help=f"Enter text {i+1} for comparison"
            )
    
    with col2:
        st.header("🎯 Quick Actions")
        
        # Sample texts button
        if st.button("📋 Load Sample Texts", help="Load example texts for testing"):
            sample_texts = [
                "The quick brown fox jumps over the lazy dog. This is a common pangram used in typography.",
                "A fast brown fox leaps over a sleepy dog. This sentence contains all letters of the alphabet.",
                "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
                "Artificial intelligence encompasses machine learning, which uses algorithms to learn patterns from data automatically."
            ]
            st.session_state.texts = sample_texts[:num_texts]
            st.rerun()
        
        # Clear all button
        if st.button("🗑️ Clear All Texts", help="Clear all text inputs"):
            st.session_state.texts = [""] * num_texts
            st.session_state.similarity_results = None
            st.session_state.model_comparison = None
            st.rerun()
        
        # Analysis button
        analyze_button = st.button(
            "🔍 Analyze Similarity",
            type="primary",
            help="Start similarity analysis",
            disabled=not selected_models or not any(text.strip() for text in st.session_state.texts)
        )
    
    # Analysis section
    if analyze_button:
        # Validate inputs
        valid_texts = [text.strip() for text in st.session_state.texts if text.strip()]
        
        if len(valid_texts) < 2:
            st.error("Please enter at least 2 texts for comparison.")
            return
        
        if not selected_models:
            st.error("Please select at least one embedding model.")
            return
        
        # Perform analysis
        with st.spinner("🔄 Analyzing texts..."):
            try:
                # Initialize components
                preprocessor = TextPreprocessor(preprocessing_options)
                model_manager = EmbeddingModelManager()
                analyzer = SimilarityAnalyzer()
                
                # Preprocess texts
                processed_texts = [preprocessor.preprocess(text) for text in valid_texts]
                
                # Analyze with each model
                results = {}
                progress_bar = st.progress(0)
                
                for i, model_name in enumerate(selected_models):
                    progress_bar.progress((i + 1) / len(selected_models))
                    
                    # Get embeddings
                    embeddings = model_manager.get_embeddings(processed_texts, model_name)
                    
                    # Calculate similarity matrix
                    similarity_matrix = analyzer.calculate_similarity_matrix(embeddings)
                    
                    # Find potential clones
                    clones = analyzer.find_clones(similarity_matrix, similarity_threshold / 100)
                    
                    results[model_name] = {
                        'similarity_matrix': similarity_matrix,
                        'clones': clones,
                        'embeddings': embeddings
                    }
                
                progress_bar.empty()
                
                # Store results
                st.session_state.similarity_results = results
                st.session_state.processed_texts = processed_texts
                
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                return
    
    # Display results
    if st.session_state.similarity_results:
        display_results(
            st.session_state.similarity_results,
            st.session_state.processed_texts,
            similarity_threshold / 100,
            selected_models
        )

def display_results(results: Dict, texts: List[str], threshold: float, selected_models: List[str]):
    """Display analysis results."""
    st.header("📊 Analysis Results")
    
    # Model comparison tabs
    if len(selected_models) > 1:
        tabs = st.tabs([f"📈 {model}" for model in selected_models] + ["🔄 Model Comparison"])
        
        # Individual model results
        for i, model_name in enumerate(selected_models):
            with tabs[i]:
                display_model_results(results[model_name], texts, threshold, model_name)
        
        # Model comparison
        with tabs[-1]:
            display_model_comparison(results, texts, threshold, selected_models)
    else:
        # Single model results
        model_name = selected_models[0]
        display_model_results(results[model_name], texts, threshold, model_name)

def display_model_results(result: Dict, texts: List[str], threshold: float, model_name: str):
    """Display results for a single model."""
    similarity_matrix = result['similarity_matrix']
    clones = result['clones']
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Pairs</h3>
            <h2>{len(texts) * (len(texts) - 1) // 2}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🚨 Potential Clones</h3>
            <h2>{len(clones)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_similarity = np.mean([similarity_matrix[i][j] for i in range(len(texts)) for j in range(i+1, len(texts))])
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Avg Similarity</h3>
            <h2>{avg_similarity:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        max_similarity = np.max([similarity_matrix[i][j] for i in range(len(texts)) for j in range(i+1, len(texts))])
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔥 Max Similarity</h3>
            <h2>{max_similarity:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Similarity matrix heatmap
    st.subheader("🔥 Similarity Matrix")
    
    fig = px.imshow(
        similarity_matrix,
        labels=dict(x="Text Index", y="Text Index", color="Similarity"),
        x=[f"Text {i+1}" for i in range(len(texts))],
        y=[f"Text {i+1}" for i in range(len(texts))],
        color_continuous_scale="RdYlBu_r",
        title=f"Similarity Matrix - {model_name}"
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Clone detection results
    if clones:
        st.subheader("🚨 Potential Clones Detected")
        
        for i, (idx1, idx2, similarity) in enumerate(clones):
            similarity_class = "similarity-high" if similarity > 0.9 else "similarity-medium" if similarity > threshold else "similarity-low"
            
            st.markdown(f"""
            <div class="{similarity_class}">
                <h4>Clone Pair {i+1}: Text {idx1+1} ↔ Text {idx2+1}</h4>
                <p><strong>Similarity: {similarity:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.text_area(f"Text {idx1+1}", texts[idx1], height=100, disabled=True)
            with col2:
                st.text_area(f"Text {idx2+1}", texts[idx2], height=100, disabled=True)
    else:
        st.info("✅ No potential clones detected above the threshold.")
    
    # Detailed similarity table
    with st.expander("📋 Detailed Similarity Table"):
        similarity_data = []
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                similarity_data.append({
                    'Text Pair': f"Text {i+1} ↔ Text {j+1}",
                    'Similarity': f"{similarity_matrix[i][j]:.1%}",
                    'Status': '🚨 Potential Clone' if similarity_matrix[i][j] >= threshold else '✅ Original'
                })
        
        df = pd.DataFrame(similarity_data)
        st.dataframe(df, use_container_width=True)

def display_model_comparison(results: Dict, texts: List[str], threshold: float, models: List[str]):
    """Display comparison between different models."""
    st.subheader("🔄 Model Performance Comparison")
    
    # Comparison metrics
    comparison_data = []
    for model_name in models:
        result = results[model_name]
        similarity_matrix = result['similarity_matrix']
        clones = result['clones']
        
        avg_similarity = np.mean([similarity_matrix[i][j] for i in range(len(texts)) for j in range(i+1, len(texts))])
        max_similarity = np.max([similarity_matrix[i][j] for i in range(len(texts)) for j in range(i+1, len(texts))])
        
        comparison_data.append({
            'Model': model_name,
            'Clones Detected': len(clones),
            'Average Similarity': f"{avg_similarity:.1%}",
            'Max Similarity': f"{max_similarity:.1%}",
            'Sensitivity': 'High' if len(clones) > len(texts) // 2 else 'Medium' if len(clones) > 0 else 'Low'
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    # Visualization of model differences
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Clones Detected by Model', 'Average Similarity by Model'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Clones detected
    fig.add_trace(
        go.Bar(
            x=[row['Model'] for row in comparison_data],
            y=[row['Clones Detected'] for row in comparison_data],
            name='Clones Detected',
            marker_color='lightcoral'
        ),
        row=1, col=1
    )
    
    # Average similarity
    avg_similarities = [float(row['Average Similarity'].strip('%')) / 100 for row in comparison_data]
    fig.add_trace(
        go.Bar(
            x=[row['Model'] for row in comparison_data],
            y=avg_similarities,
            name='Average Similarity',
            marker_color='lightblue'
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model recommendations
    st.subheader("💡 Model Recommendations")
    
    best_model = max(comparison_data, key=lambda x: x['Clones Detected'])
    most_conservative = min(comparison_data, key=lambda x: x['Clones Detected'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **🎯 Most Sensitive Model**
        
        **{best_model['Model']}** detected the most potential clones ({best_model['Clones Detected']}).
        
        Best for: Strict plagiarism detection, academic integrity checks.
        """)
    
    with col2:
        st.info(f"""
        **🛡️ Most Conservative Model**
        
        **{most_conservative['Model']}** detected the fewest false positives ({most_conservative['Clones Detected']}).
        
        Best for: Content similarity analysis, avoiding false alarms.
        """)

if __name__ == "__main__":
    main()
