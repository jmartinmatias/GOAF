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

class GeminiAnalyzer:
    """Uses Gemini API to generate semantic analysis of functions."""
    
    def __init__(self, api_key=None):
        # Use provided API key or get from environment
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("No Gemini API key found. Semantic analysis will be unavailable.")
            self.model = None
            return
            
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("Gemini analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {str(e)}")
            self.model = None
    
    def analyze_function(self, func):
        """Generate semantic description and usage scenarios for a function."""
        if not self.model:
            logger.warning("Gemini model not initialized. Cannot analyze function.")
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
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from response: {response.text}")
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
        if not self.model:
            logger.warning("Gemini model not initialized. Cannot generate function.")
            return "# Function generation requires Gemini API key"
            
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