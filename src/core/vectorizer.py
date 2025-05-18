import inspect
from typing import Dict, Any, Callable, List, Optional, Union
import numpy as np
import logging

logger = logging.getLogger("vectorizer")

class FunctionVectorizer:
    """Converts functions to vector embeddings for semantic search."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the vectorizer with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"Initialized vectorizer with model: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Using fallback vectorizer.")
            self.model = None
    
    def get_function_text(self, func: Callable) -> str:
        """
        Extract textual representation of a function for embedding.
        
        Args:
            func: The function to extract text from
            
        Returns:
            A string containing the function's name, docstring, and signature
        """
        # Get function name
        func_name = func.__name__
        
        # Get docstring
        docstring = inspect.getdoc(func) or ""
        
        # Get signature
        try:
            signature = str(inspect.signature(func))
        except ValueError:
            signature = "()"
        
        # Get source code (removing indentation)
        try:
            source_lines = inspect.getsourcelines(func)[0]
            # Remove the definition line and dedent
            source_code = "".join(source_lines[1:])
            # Remove common indentation
            source_code = inspect.cleandoc(source_code)
        except (IOError, TypeError):
            source_code = ""
        
        # Extract parameter names
        params = []
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
        except ValueError:
            pass
        
        # Combine all text
        all_text = (
            f"Function Name: {func_name}\n"
            f"Function Description: {docstring}\n"
            f"Function Parameters: {', '.join(params)}\n"
            f"Function Signature: {func_name}{signature}\n"
        )
        
        return all_text
    
    def vectorize_function(self, func: Callable) -> np.ndarray:
        """
        Create a vector embedding for a function.
        
        Args:
            func: The function to vectorize
            
        Returns:
            A numpy array containing the vector embedding
        """
        if self.model is None:
            # Fallback to simple bag-of-words if no model
            return self._simple_embedding(self.get_function_text(func))
            
        function_text = self.get_function_text(func)
        return self.model.encode(function_text)
    
    def vectorize_query(self, query: str) -> np.ndarray:
        """
        Create a vector embedding for a natural language query.
        
        Args:
            query: The natural language query
            
        Returns:
            A numpy array containing the vector embedding
        """
        if self.model is None:
            # Fallback to simple bag-of-words if no model
            return self._simple_embedding(query)
            
        return self.model.encode(query)
    
    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Calculate cosine similarity
        return float(np.dot(vec1_norm, vec2_norm))
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """
        Create a simple embedding using bag of words approach.
        This is a fallback when sentence-transformers is not available.
        
        Args:
            text: Text to embed
            
        Returns:
            Simple vector representation
        """
        # Simple word frequency embedding
        words = text.lower().split()
        unique_words = set(words)
        
        # Create a simple vector of word counts
        vector = np.zeros(len(unique_words))
        word_to_index = {word: i for i, word in enumerate(unique_words)}
        
        for word in words:
            vector[word_to_index[word]] += 1
            
        # Normalize
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linal