# RAG Chunking Strategy Visualizer

A comprehensive web application for analyzing and visualizing different text chunking strategies for Retrieval-Augmented Generation (RAG) systems.

## Features

### 📚 **Multiple Input Methods**
- **PDF Upload**: Extract and analyze text from PDF documents
- **Text Input**: Paste any text content for analysis
- **Sample Documents**: Pre-loaded examples (academic papers, technical docs, narratives)

### 🔧 **Chunking Strategies**
- **Fixed Size**: Traditional fixed-character chunking with overlap
- **Sentence-based**: Groups sentences together maintaining readability
- **Paragraph-based**: Preserves document structure and topic boundaries
- **Semantic**: Groups semantically similar content using embeddings
- **Recursive Character**: Hierarchical splitting with multiple separators
- **Token-based**: Model-specific token counting for API optimization
- **Sliding Window**: Overlapping chunks for maximum context preservation

### 📊 **Advanced Analysis**
- **Size Distribution**: Statistical analysis of chunk sizes
- **Semantic Coherence**: Measures topic consistency within chunks
- **Overlap Analysis**: Quantifies information overlap between chunks
- **Embedding Visualization**: 2D/3D visualization of semantic relationships
- **Quality Scoring**: Multi-metric evaluation of chunking effectiveness

### 🎯 **Interactive Visualizations**
- **Distribution Plots**: Histograms and box plots of chunk characteristics
- **Similarity Heatmaps**: Visual representation of semantic relationships
- **Scatter Plots**: Embedding space visualization with UMAP/PCA
- **Comparison Charts**: Side-by-side strategy performance analysis
- **Quality Radar Charts**: Multi-dimensional quality assessment

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models** (optional, will download automatically):
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### Basic Workflow

1. **Choose Input Method**:
   - Upload a PDF document
   - Paste text directly
   - Load a sample document

2. **Select Chunking Strategies**:
   - Choose one or more strategies to compare
   - Adjust parameters for each strategy

3. **Configure Analysis**:
   - Select analysis types to perform
   - Choose embedding model for semantic analysis

4. **Run Analysis**:
   - Click "Analyze Chunking Strategies"
   - View comprehensive results and visualizations

### Strategy Configuration

#### Fixed Size Chunking
- **Chunk Size**: Number of characters per chunk (100-2000)
- **Overlap**: Character overlap between chunks (0-200)
- **Best for**: Predictable processing, consistent memory usage

#### Sentence-based Chunking
- **Sentences per Chunk**: Number of sentences to group (1-10)
- **Sentence Overlap**: Overlapping sentences between chunks (0-3)
- **Best for**: Maintaining readability, preserving sentence boundaries

#### Semantic Chunking
- **Similarity Threshold**: Minimum similarity for grouping (0.5-0.95)
- **Min Chunk Size**: Minimum characters per chunk (50-500)
- **Best for**: Topic coherence, semantic consistency

#### Token-based Chunking
- **Tokens per Chunk**: Number of tokens per chunk (50-1000)
- **Token Overlap**: Overlapping tokens between chunks (0-100)
- **Best for**: API optimization, model-specific limits

## Analysis Types

### Chunk Size Distribution
- Statistical analysis of chunk sizes
- Histograms and percentile analysis
- Outlier detection and consistency metrics

### Semantic Coherence
- Measures topic consistency within chunks
- Uses sentence embeddings and cosine similarity
- Provides coherence scores for strategy comparison

### Overlap Analysis
- Quantifies information overlap between consecutive chunks
- Helps optimize overlap parameters
- Identifies redundancy and information gaps

### Embedding Visualization
- 2D/3D visualization of chunk relationships
- UMAP and PCA dimensionality reduction
- Interactive scatter plots with chunk previews

## Quality Metrics

### Size Consistency
- Measures uniformity of chunk sizes
- Lower coefficient of variation = higher consistency
- Important for predictable processing

### Semantic Coherence
- Average similarity between chunks
- Higher coherence = better topic preservation
- Critical for retrieval accuracy

### Size Appropriateness
- Proximity to optimal chunk size (~500 characters)
- Balances context and specificity
- Affects retrieval and generation quality

### Coverage
- Proportion of document content preserved
- Ensures comprehensive information retention
- Prevents information loss during chunking

## Strategy Recommendations

### For Academic Papers
- **Paragraph-based** or **Semantic** chunking
- Preserves logical structure and topic flow
- Maintains citation and reference integrity

### For Technical Documentation
- **Recursive Character** or **Sentence-based** chunking
- Respects formatting and code blocks
- Maintains procedural step integrity

### For Narrative Content
- **Sentence-based** or **Paragraph-based** chunking
- Preserves story flow and character development
- Maintains dialogue and scene boundaries

### For API-Optimized RAG
- **Token-based** chunking
- Respects model token limits
- Optimizes API costs and response times

## Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **PDF Processing**: PyPDF2 for text extraction
- **Chunking**: LangChain text splitters + custom implementations
- **Embeddings**: Sentence Transformers for semantic analysis
- **Visualization**: Plotly for interactive charts

### Performance
- **Local Processing**: No external API dependencies (except OpenAI embeddings)
- **Efficient Analysis**: Optimized algorithms for large documents
- **Memory Management**: Streaming processing for large files
- **Caching**: Model caching for faster subsequent runs

### Supported Formats
- **PDF**: Text extraction with metadata
- **Plain Text**: Direct input processing
- **Structured Text**: Automatic paragraph/sentence detection

## Best Practices

### Document Preparation
- Ensure clean PDF text extraction
- Remove headers/footers if necessary
- Verify text encoding and formatting

### Strategy Selection
- Start with 2-3 strategies for comparison
- Consider document type and use case
- Test with representative content samples

### Parameter Tuning
- Begin with default parameters
- Adjust based on document characteristics
- Monitor quality metrics for optimization

### Performance Optimization
- Use appropriate embedding models for your domain
- Consider chunk size vs. processing time trade-offs
- Monitor memory usage with large documents

## Troubleshooting

### Common Issues

1. **PDF Extraction Errors**: Ensure PDF contains extractable text
2. **Memory Issues**: Reduce document size or use lighter embedding models
3. **Slow Processing**: Use smaller embedding models or fewer strategies
4. **Poor Semantic Analysis**: Try different embedding models

### Performance Tips
- Start with smaller documents for testing
- Use "all-MiniLM-L6-v2" for faster processing
- Limit analysis to essential metrics for large documents
- Clear browser cache if visualizations don't load

## Contributing

Contributions are welcome! Areas for improvement:
- Additional chunking strategies
- New analysis metrics
- Enhanced visualizations
- Performance optimizations

## License

This project is open source and available under the MIT License.
