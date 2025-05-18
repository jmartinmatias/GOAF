# GOAF - Algorithmic Function Explorer

An intelligent function registry with semantic understanding, algorithm analysis, and AI-powered function generation capabilities.

## Features

### Core Features

- **Function Registration** - Register Python functions with detailed metadata extraction
- **Algorithmic Analysis** - Understand the time/space complexity and algorithmic patterns
- **Semantic Search** - Find functions using natural language queries and vector embeddings
- **Interactive UI** - Explore, compare, and execute functions through an intuitive Streamlit interface
- **Function Execution** - Test functions with different inputs and analyze performance
- **Pattern Explorer** - Discover common algorithmic patterns across your function library

### AI-Powered Features

- **Semantic Understanding** - AI-powered analysis of what functions actually do (beyond docstrings)
- **Advanced Search** - Enhanced search capabilities using semantic understanding
- **Function Generation** - Generate new functions from natural language descriptions
- **Function Components** - Break down functions into reusable components
- **AI-Assisted Documentation** - Generate detailed semantic documentation for functions

### Planned

- **Function Chaining** - Compose complex operations from simpler functions
- **Agent-Based Distribution** - Delegate tasks to specialized function agents
- **Performance Profiling** - Advanced metrics for function execution
- **Visualization Tools** - Advanced algorithm and performance visualizations

## Setup

1. **Clone the repository**
    
    bash
    
    ```bash
    git clone https://github.com/YOUR_USERNAME/GOAF.git
    cd GOAF
    ```
    
2. **Create a virtual environment**
    
    bash
    
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    
3. **Install dependencies**
    
    bash
    
    ```bash
    pip install -r requirements.txt
    ```
    
4. **Set up API keys** Create a `.env` file in the project root:
    
    ```
    # .env
    GOOGLE_API_KEY=your_gemini_api_key_here
    # Optional fallback
    OPENAI_API_KEY=your_openai_api_key_here
    ```
    
5. **Run the application**
    
    bash
    
    ```bash
    # Run with PyTorch compatibility fix (recommended)
    streamlit run app.py --server.fileWatcherType none
    ```
    

## Gemini API Integration

GOAF integrates with Google's Gemini AI to provide semantic analysis and function generation. The application specifically uses the `gemini-2.5-flash-preview-04-17` model.

### API Key Setup

1. Get an API key from [Google AI Studio](https://ai.google.dev/)
2. Add your API key to the `.env` file as `GOOGLE_API_KEY=your_key_here`
3. Restart the application

### Compatibility

If you encounter issues with the Gemini API, the app includes fallbacks to other model versions:

- gemini-1.5-pro
- gemini-1.0-pro
- gemini-pro

## PyTorch-Streamlit Compatibility

This project includes fixes for known compatibility issues between PyTorch and Streamlit's hot-reload feature. For the best experience, run with:

bash

```bash
streamlit run app.py --server.fileWatcherType none
```

Alternatively, create a `.streamlit/config.toml` file with:

toml

```toml
[server]
fileWatcherType = "none"
```

## Usage Guide

### Function Registration

Functions are automatically registered when you start the application. You can also add generated functions through the UI.

### Exploring Functions

Navigate to "Function Overview" to browse all registered functions, organized by algorithmic category.

### Algorithmic Search

Use the "Algorithmic Search" page to find functions using natural language queries, algorithm patterns, or implementation properties.

### Function Generation

Navigate to "Function Generator" to create new functions based on natural language descriptions, leveraging existing functions as building blocks.

### Algorithm Comparison

Use the "Algorithm Comparison" page to compare different implementations side-by-side, including:

- Time and space complexity
- Control flow structure
- Data structure usage
- Execution performance

### Function Execution

Test functions with different inputs and analyze their behavior on the "Function Execution" page.

## Project Structure

```
GOAF/
├── app.py                  # Main Streamlit application
├── .env                    # Environment variables (API keys)
├── requirements.txt        # Project dependencies
├── streamlit_torch_fix.py  # PyTorch compatibility fix for Streamlit
├── src/
│   ├── core/
│   │   ├── registry.py       # Function registry implementation
│   │   ├── vectorizer.py     # Function vectorization and analysis
│   │   └── semantic_analyzer.py # Gemini-powered semantic analysis
│   └── utils/
│       └── visualization.py  # Visualization utilities
└── example_functions/
    └── __init__.py         # Example functions to populate the registry
```

## Troubleshooting

### Gemini API Issues

- If you see "404 models/gemini-X is not found" errors, update to the latest google-generativeai package:
    
    bash
    
    ```bash
    pip install google-generativeai --upgrade
    ```
    
- Make sure your API key has access to the Gemini models

### PyTorch/Streamlit Issues

- If you see errors about "torch._classes.**path**._path", use the `--server.fileWatcherType none` flag
- For a permanent fix, create a `.streamlit/config.toml` file disabling hot-reload

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.