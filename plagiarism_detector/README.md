# Plagiarism Detector - Semantic Similarity Analyzer

A sophisticated web application for detecting potential plagiarism using semantic similarity analysis with multiple embedding models.

## Features

### 🔍 **Multi-Model Analysis**
- **Sentence Transformers**: all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L6-v2
- **OpenAI Embeddings**: text-embedding-3-small, text-embedding-ada-002
- **Model Comparison**: Side-by-side performance analysis

### 📊 **Advanced Similarity Analysis**
- **Similarity Matrix**: Visual heatmap of pairwise similarities
- **Clone Detection**: Automatic identification of potential plagiarism
- **Threshold Customization**: Adjustable similarity thresholds
- **Statistical Analysis**: Comprehensive similarity statistics

### 🎯 **Smart Text Processing**
- **Dynamic Input**: Support for 2-10 texts simultaneously
- **Text Preprocessing**: Configurable cleaning options
- **Input Validation**: Length limits and content validation
- **Sample Data**: Built-in examples for testing

### 📈 **Rich Visualizations**
- **Interactive Heatmaps**: Plotly-powered similarity matrices
- **Performance Metrics**: Model comparison charts
- **Clone Highlighting**: Visual identification of similar content
- **Detailed Reports**: Comprehensive analysis summaries

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key if using OpenAI models
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### Basic Analysis

1. **Select Models**: Choose one or more embedding models from the sidebar
2. **Enter Texts**: Input 2-10 texts in the provided text areas
3. **Configure Settings**: Adjust similarity threshold and preprocessing options
4. **Analyze**: Click "Analyze Similarity" to start the analysis
5. **Review Results**: Examine the similarity matrix, clone detection, and statistics

### Advanced Features

#### Model Comparison
- Select multiple models to compare their performance
- View side-by-side results and recommendations
- Understand which models work best for your use case

#### Text Preprocessing
- **Remove extra whitespace**: Normalize spacing
- **Convert to lowercase**: Case-insensitive comparison
- **Remove special characters**: Focus on content words

#### Clone Detection
- Texts above the similarity threshold are flagged as potential clones
- Color-coded results: Red (high similarity), Orange (medium), Green (low)
- Detailed pair-by-pair comparison

## How It Works

### Embedding Generation
The application converts texts into high-dimensional vectors (embeddings) that capture semantic meaning:

1. **Sentence Transformers**: Pre-trained models optimized for semantic similarity
2. **OpenAI Embeddings**: State-of-the-art commercial embedding models
3. **Preprocessing**: Optional text cleaning and normalization

### Similarity Calculation
Cosine similarity is calculated between all text pairs:
- **Range**: 0 (completely different) to 1 (identical)
- **Threshold**: Configurable cutoff for clone detection
- **Matrix**: Visual representation of all pairwise similarities

### Clone Detection
Potential plagiarism is identified using:
- **Similarity Threshold**: Texts above threshold are flagged
- **Ranking**: Results sorted by similarity score
- **Clustering**: Groups of similar texts are identified

## Model Comparison

### Sentence Transformers
- **all-MiniLM-L6-v2**: Fast, lightweight, good for general use
- **all-mpnet-base-v2**: High quality, better semantic understanding
- **paraphrase-MiniLM-L6-v2**: Optimized for paraphrase detection

### OpenAI Models
- **text-embedding-3-small**: Latest OpenAI model, excellent quality
- **text-embedding-ada-002**: Previous generation, very good quality

### Recommendations
- **Academic Use**: all-mpnet-base-v2 or OpenAI models for highest accuracy
- **Fast Processing**: all-MiniLM-L6-v2 for quick analysis
- **Paraphrase Detection**: paraphrase-MiniLM-L6-v2 for detecting rewording

## Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_api_key_here    # Required for OpenAI models
SIMILARITY_THRESHOLD=0.8            # Default similarity threshold
MAX_TEXT_LENGTH=10000              # Maximum characters per text
DEFAULT_MODEL=all-MiniLM-L6-v2     # Default embedding model
```

### Similarity Thresholds
- **90-100%**: Nearly identical (definite plagiarism)
- **80-90%**: Very similar (likely plagiarism)
- **70-80%**: Moderately similar (possible plagiarism)
- **Below 70%**: Different content (unlikely plagiarism)

## Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **Embeddings**: Sentence Transformers + OpenAI API
- **Similarity**: Scikit-learn cosine similarity
- **Visualization**: Plotly interactive charts

### Performance
- **Local Models**: Fast processing, no API costs
- **OpenAI Models**: Higher quality, requires API key
- **Caching**: Models cached for faster subsequent runs
- **Batch Processing**: Efficient handling of multiple texts

## Troubleshooting

### Common Issues

1. **OpenAI API Error**: Ensure valid API key in environment variables
2. **Model Loading Error**: Check internet connection for downloading models
3. **Memory Issues**: Reduce number of texts or use smaller models
4. **Slow Performance**: Use lighter models like all-MiniLM-L6-v2

### Performance Tips
- Start with 2-3 texts for initial testing
- Use local models for faster processing
- Clear model cache if memory issues occur
- Preprocess texts to improve accuracy

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.
