# GOAF - Algorithmic Function Explorer

An intelligent function registry with semantic understanding, algorithm analysis, and AI-powered function generation capabilities.

## Features

### Implemented
- **Function Registration** - Register Python functions with detailed metadata extraction
- **Algorithmic Analysis** - Understand the time/space complexity and algorithmic patterns
- **Semantic Search** - Find functions using natural language queries and vector embeddings
- **Interactive UI** - Explore, compare, and execute functions through an intuitive Streamlit interface
- **Function Execution** - Test functions with different inputs and analyze performance
- **Pattern Explorer** - Discover common algorithmic patterns across your function library

### New with Gemini API Integration
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
    ```bash
    git clone https://github.com/YOUR_USERNAME/GOAF.git
    cd GOAF

2. **Create a virtual environment**
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**
    pip install -r requirements.txt

4. **Set up API keys** Create a .env file in the project root:
    # .env
    GEMINI_API_KEY=your_gemini_api_key_here

5. **Run the application**
    streamlit run app.py


**API Integration**
GOAF integrates with Google's Gemini API to provide advanced semantic analysis and function generation. To use these features:

1. Get an API key from Google AI Studio
2. Add your API key to the .env file
3. Restart the application

**Usage Guide**
#Function Registration#
Functions are automatically registered when you start the application. You can also add generated functions through the UI.

#Exploring Functions#
Navigate to "Function Overview" to browse all registered functions, organized by algorithmic category.

#Algorithmic Search#
Use the "Algorithmic Search" page to find functions using natural language queries, algorithm patterns, or implementation properties.

#Function Generation#
Navigate to "Function Generator" to create new functions based on natural language descriptions, leveraging existing functions as building blocks.

#Algorithm Comparison#
Use the "Algorithm Comparison" page to compare different implementations side-by-side, including:
- Time and space complexity
- Control flow structure
- Data structure usage
- Execution performance

#Function Execution#
Test functions with different inputs and analyze their behavior on the "Function Execution" page.

#Project Structure#
GOAF/
├── app.py                 # Main Streamlit application
├── .env                   # Environment variables (API keys)
├── requirements.txt       # Project dependencies
├── src/
│   ├── core/
│   │   ├── registry.py    # Function registry implementation
│   │   ├── vectorizer.py  # Function vectorization and analysis
│   │   └── semantic_analyzer.py # Gemini-powered semantic analysis
│   └── utils/
│       └── visualization.py # Visualization utilities
└── example_functions/
    └── __init__.py        # Example functions to populate the registry

#Contributing#
Contributions are welcome! Please feel free to submit a Pull Request.

#License#
This project is licensed under the MIT License - see the LICENSE file for details.
