# MISOGI Week 4 Day 3 Assignment

Complete implementation of three advanced AI/ML applications demonstrating modern software engineering practices and cutting-edge technologies.

## 🚀 **Projects Overview**

### 1. **Discord MCP Server** 📡
**Location**: `discord_bot_mcp/`

A production-ready Model Context Protocol (MCP) server for Discord integration with enterprise-grade features.

**Key Features**:
- ✅ Complete MCP protocol implementation
- ✅ 5 Discord tools (send_message, get_messages, get_channel_info, search_messages, moderate_content)
- ✅ API key authentication with granular permissions
- ✅ Multi-tenancy support
- ✅ Rate limiting and audit logging
- ✅ MCP Inspector integration
- ✅ Comprehensive unit tests

### 2. **Plagiarism Detector** 🔍
**Location**: `plagiarism_detector/`

Advanced semantic similarity analyzer using multiple embedding models for plagiarism detection.

**Key Features**:
- ✅ Multi-model analysis (5 embedding models)
- ✅ Interactive similarity matrix visualization
- ✅ Clone detection with configurable thresholds
- ✅ Model performance comparison
- ✅ Beautiful Plotly visualizations
- ✅ Text preprocessing pipeline

### 3. **RAG Chunking Strategy Visualizer** 📚
**Location**: `rag_chunking_visualizer/`

Comprehensive tool for analyzing and visualizing different text chunking strategies for RAG systems.

**Key Features**:
- ✅ 7 chunking strategies implementation
- ✅ PDF upload and text extraction
- ✅ Advanced chunk analysis (size, coherence, overlap)
- ✅ Embedding visualization with UMAP/PCA
- ✅ Quality scoring system
- ✅ Interactive comparison dashboards

## 🛠 **Technology Stack**

### **Backend**
- **Python 3.10+** - Core language
- **FastAPI** - Web framework for MCP server
- **Streamlit** - Web interface for analysis tools
- **MCP SDK** - Model Context Protocol implementation

### **AI/ML**
- **Sentence Transformers** - Local embedding models
- **OpenAI API** - Commercial embedding models
- **scikit-learn** - Machine learning utilities
- **UMAP** - Dimensionality reduction
- **tiktoken** - Token counting

### **Data Processing**
- **PyPDF2** - PDF text extraction
- **LangChain** - Text splitting utilities
- **pandas** - Data manipulation
- **numpy** - Numerical computing

### **Visualization**
- **Plotly** - Interactive charts and graphs
- **matplotlib** - Static plotting
- **seaborn** - Statistical visualization

### **Development**
- **UV** - Fast Python package manager
- **pytest** - Testing framework
- **black** - Code formatting
- **mypy** - Type checking

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.10 or higher
- Git
- UV package manager (recommended) or pip

### **Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shubhampawar0901/misogi-w4d3.git
   cd misogi-w4d3
   ```

2. **Set up each project**:

   **Discord MCP Server**:
   ```bash
   cd discord_bot_mcp
   uv sync  # or pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your Discord bot token
   uv run discord-mcp-server --help
   ```

   **Plagiarism Detector**:
   ```bash
   cd plagiarism_detector
   pip install -r requirements.txt
   streamlit run app.py
   ```

   **RAG Chunking Visualizer**:
   ```bash
   cd rag_chunking_visualizer
   pip install -r requirements.txt
   streamlit run app.py
   ```

## 📖 **Usage Examples**

### **Discord MCP Server**
```bash
# Start the MCP server
uv run discord-mcp-server

# Test with MCP Inspector
mcp-inspector
# Load the mcp_inspector_config.json file
```

### **Plagiarism Detector**
1. Open the Streamlit app
2. Select embedding models
3. Enter texts to compare
4. Analyze similarity and view results

### **RAG Chunking Visualizer**
1. Upload a PDF or paste text
2. Select chunking strategies
3. Configure parameters
4. Analyze and compare results

## 🧪 **Testing**

Each project includes comprehensive tests:

```bash
# Discord MCP Server
cd discord_bot_mcp
uv run pytest tests/ -v

# Plagiarism Detector
cd plagiarism_detector
python test_app.py

# RAG Chunking Visualizer
cd rag_chunking_visualizer
python test_app.py
```

## 📁 **Project Structure**

```
misogi-w4d3/
├── .gitignore                     # Comprehensive gitignore
├── README.md                      # This file
├── questions/                     # Original assignment questions
│   ├── q1.md
│   ├── q2.md
│   └── q3.md
├── discord_bot_mcp/              # Question 1: Discord MCP Server
│   ├── discord_mcp/              # Main package
│   ├── tests/                    # Unit tests
│   ├── pyproject.toml           # UV configuration
│   └── README.md                # Project documentation
├── plagiarism_detector/          # Question 2: Plagiarism Detector
│   ├── app.py                   # Main Streamlit app
│   ├── embedding_models.py     # Model management
│   ├── similarity_analyzer.py  # Analysis engine
│   └── README.md               # Project documentation
└── rag_chunking_visualizer/     # Question 3: RAG Chunking Visualizer
    ├── app.py                   # Main Streamlit app
    ├── chunking_strategies.py   # Chunking implementations
    ├── chunk_analyzer.py        # Analysis engine
    └── README.md               # Project documentation
```

## 🔧 **Configuration**

### **Environment Variables**
Each project uses environment variables for configuration:

- **Discord Bot Token** (Discord MCP Server)
- **OpenAI API Key** (Plagiarism Detector, RAG Visualizer)
- **Various thresholds and limits**

See individual `.env.example` files for details.

## 🎯 **Key Features Demonstrated**

### **Software Engineering**
- ✅ Modular architecture
- ✅ Comprehensive testing
- ✅ Type hints and documentation
- ✅ Error handling and validation
- ✅ Configuration management

### **AI/ML Integration**
- ✅ Multiple embedding models
- ✅ Semantic similarity analysis
- ✅ Advanced text processing
- ✅ Dimensionality reduction
- ✅ Interactive visualizations

### **Production Readiness**
- ✅ Authentication and authorization
- ✅ Rate limiting and monitoring
- ✅ Audit logging
- ✅ Multi-tenancy support
- ✅ Comprehensive error handling
