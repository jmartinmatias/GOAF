# src/core/semantic_analyzer.py
import os
import inspect
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("semantic_analyzer")

# Load environment variables
load_dotenv()

def initialize_gemini():
    """Initialize Gemini with API key from environment variables."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    try:
        genai.configure(api_key=api_key)
        logger.info("Gemini API initialized successfully")
        return True
    except Exception as e:
        raise RuntimeError(f"Error configuring Gemini API: {str(e)}")
    
class GeminiAnalyzer:
    """Uses Gemini API to generate semantic analysis of functions."""
    
    def __init__(self):
        """Initialize the Gemini analyzer."""
        try:
            # Initialize the Gemini API
            initialize_gemini()
            
            # Use the specific model name provided
            model_name = "gemini-2.5-flash-preview-04-17"
            logger.info(f"Using Gemini model: {model_name}")
            
            try:
                self.model = genai.GenerativeModel(model_name)
                self.available = True
            except Exception as e:
                logger.error(f"Error creating model instance with {model_name}: {str(e)}")
                # Fallback to trying other models if this one fails
                fallback_models = [
                    "gemini-1.5-pro",
                    "gemini-1.0-pro",
                    "gemini-pro"
                ]
                
                for fallback in fallback_models:
                    try:
                        logger.info(f"Trying fallback model: {fallback}")
                        self.model = genai.GenerativeModel(fallback)
                        self.available = True
                        logger.info(f"Successfully connected to fallback model: {fallback}")
                        break
                    except Exception as e:
                        logger.warning(f"Could not use fallback model {fallback}: {str(e)}")
                else:
                    # If all fallbacks fail
                    self.model = None
                    self.available = False
                    logger.error("All Gemini models failed to initialize")
        except (ValueError, RuntimeError) as e:
            logger.error(f"Gemini initialization failed: {str(e)}")
            self.model = None
            self.available = False
    
    def analyze_function(self, func):
        """Generate semantic description and usage scenarios for a function."""
        if not self.available:
            logger.warning("Semantic analysis unavailable - Gemini API not initialized")
            return self._default_analysis(func)
            
        try:
            function_code = inspect.getsource(func)
            function_name = func.__name__
            docstring = inspect.getdoc(func) or ""
            
            prompt = f"""
            Analyze this Python function:
            
            ```python
            {function_code}
            ```
            
            Provide the following in JSON format:
            1. "semantic_purpose": What this function actually does in plain language (1-2 sentences)
            2. "use_cases": 3-5 practical scenarios where this function would be useful (as a list)
            3. "limitations": Any edge cases or situations where this might not work well (as a list)
            4. "related_concepts": Similar algorithms or patterns to be aware of (as a list)
            5. "decomposition": How this function could be broken into smaller, reusable parts (as a list of component descriptions)
            
            Return only valid JSON with these fields.
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse the JSON response
            try:
                # Extract JSON from response if needed
                text = response.text
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                    
                result = json.loads(text)
                logger.info(f"Successfully analyzed function {function_name}")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse JSON from response: {str(e)}")
                return self._extract_fields_from_text(response.text, func)
                
        except Exception as e:
            logger.error(f"Error analyzing function {func.__name__}: {str(e)}")
            return self._default_analysis(func)
    
    def _default_analysis(self, func):
        """Provide default analysis when Gemini API is unavailable."""
        return {
            "semantic_purpose": inspect.getdoc(func) or f"Function that performs {func.__name__}",
            "use_cases": [f"Situations requiring {func.__name__}"],
            "limitations": ["No semantic analysis available"],
            "related_concepts": [],
            "decomposition": []
        }
    
    def _extract_fields_from_text(self, text, func):
        """Extract structured information from unstructured text response."""
        result = self._default_analysis(func)
        
        # Try to find each section
        sections = [
            ("semantic_purpose", "semantic purpose"),
            ("use_cases", "use cases"),
            ("limitations", "limitations"),
            ("related_concepts", "related concepts"),
            ("decomposition", "decomposition")
        ]
        
        for key, heading in sections:
            if heading.lower() in text.lower():
                # Get text after the heading
                section_text = text.lower().split(heading.lower())[1].split("\n\n")[0]
                # Clean up and convert to list if needed
                if key == "semantic_purpose":
                    result[key] = section_text.strip().strip(":").strip()
                else:
                    # Extract list items
                    items = [item.strip().strip("-").strip() 
                             for item in section_text.split("\n") 
                             if item.strip() and "-" in item]
                    if items:
                        result[key] = items
        
        return result
    
    def generate_function(self, description, components=None):
        """Generate a new function based on the description."""
        if not self.available:
            logger.warning("Function generation unavailable - Gemini API not initialized")
            return "# Function generation requires Gemini API key (GOOGLE_API_KEY)"
            
        try:
            # Prepare prompt
            prompt = f"""
            Create a Python function that does the following:
            {description}
            """
            
            # Add components if available
            if components:
                prompt += f"""
                
                These components from similar functions might be helpful:
                {components}
                """
                
            prompt += """
            
            Return only the Python code for the function, including:
            1. A clear docstring with description, parameters, and return values
            2. Type hints for parameters and return value
            3. Input validation where appropriate
            4. Descriptive variable names
            5. Comments for complex logic
            
            The function should follow PEP 8 style guidelines.
            """
            
            response = self.model.generate_content(prompt)
            
            # Extract code from response
            text = response.text
            if "```python" in text:
                code = text.split("```python")[1].split("```")[0].strip()
            elif "```" in text:
                code = text.split("```")[1].split("```")[0].strip()
            else:
                code = text.strip()
                
            return code
            
        except Exception as e:
            logger.error(f"Error generating function: {str(e)}")
            return f"# Error generating function: {str(e)}"