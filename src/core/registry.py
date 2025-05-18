#registry.py
import inspect
import time
from typing import Dict, Any, Callable, List, Optional, Union
import logging
import numpy as np
from src.core.vectorizer import FunctionVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("function_registry")

class FunctionRegistry:
    """Registry for storing and executing functions with metadata tracking."""
    
    def __init__(self, vectorizer=None):
        """Initialize an empty function registry."""
        self.functions = {}  # Store actual function objects
        self.metadata = {}   # Store function metadata
        self.stats = {}      # Store execution statistics
        self.vectors = {}    # Store function vector embeddings
        
        # Initialize vectorizer
        self.vectorizer = vectorizer or FunctionVectorizer()
        
        logger.info("Function registry initialized")
    
    def register(self, func: Callable) -> str:
        """
        Register a function with the registry.
        
        Args:
            func: The function to register
            
        Returns:
            The name of the registered function
        """
        func_name = func.__name__
        self.functions[func_name] = func
        
        # Extract basic metadata
        self.metadata[func_name] = {
            "name": func_name,
            "docstring": inspect.getdoc(func) or "",
            "signature": str(inspect.signature(func)),
            "parameters": [param for param in inspect.signature(func).parameters],
            "module": func.__module__,
            "registered_at": time.time()
        }

        # Add code analysis patterns to metadata
        code_patterns = self.vectorizer.analyze_code_patterns(func)
        self.metadata[func_name]["code_patterns"] = code_patterns

        # Vectorize with focus on implementation
        try:
            vector = self.vectorizer.vectorize_function(func)
            self.vectors[func_name] = vector
        except Exception as e:
            logger.warning(f"Failed to vectorize function {func_name}: {str(e)}")
                    
        # Initialize statistics
        self.stats[func_name] = {
            "calls": 0,
            "total_time": 0,
            "avg_time": 0,
            "success": 0,
            "errors": 0
        }
        
        logger.info(f"Registered function: {func_name}")
        return func_name
    
    def execute(self, func_name: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute a registered function with performance tracking.
        
        Args:
            func_name: Name of the function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Dictionary containing execution results and metadata
        """
        if func_name not in self.functions:
            error_msg = f"Function '{func_name}' not found in registry"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "result": None,
                "execution_time": 0
            }
        
        func = self.functions[func_name]
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            status = "success"
            error = None
            self.stats[func_name]["success"] += 1
        except Exception as e:
            result = None
            status = "error"
            error = str(e)
            self.stats[func_name]["errors"] += 1
            logger.error(f"Error executing {func_name}: {error}")
        
        execution_time = time.time() - start_time
        
        # Update statistics
        stats = self.stats[func_name]
        stats["calls"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["calls"]
        
        return {
            "result": result,
            "status": status,
            "error": error,
            "execution_time": execution_time
        }
    
    def search(self, query: str) -> List[str]:
        """
        Simple search for functions based on name or docstring.
        
        Args:
            query: Search string to match against function names and docstrings
            
        Returns:
            List of function names that match the query
        """
        query = query.lower()
        matches = []
        
        for func_name, metadata in self.metadata.items():
            if (query in func_name.lower() or 
                query in metadata["docstring"].lower()):
                matches.append(func_name)
        
        return matches
    
    def get_metadata(self, func_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific function.
        
        Args:
            func_name: Name of the function
            
        Returns:
            Dictionary of function metadata or None if not found
        """
        return self.metadata.get(func_name)
    
    def get_stats(self, func_name: str) -> Optional[Dict[str, Any]]:
        """
        Get execution statistics for a specific function.
        
        Args:
            func_name: Name of the function
            
        Returns:
            Dictionary of execution statistics or None if not found
        """
        return self.stats.get(func_name)
    
    def list_functions(self) -> List[str]:
        """
        Get a list of all registered function names.
        
        Returns:
            List of function names
        """
        return list(self.functions.keys())
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for functions semantically similar to a natural language query.
        
        Args:
            query: Natural language query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing matching function info with similarity scores
        """
        # Extract potential code patterns from the query
        query_patterns = self._extract_patterns_from_query(query)
        
        # Get initial vector-based matches
        if not self.vectors:
            logger.warning("No function vectors available. Using keyword search instead.")
            return [{"name": name, "similarity": 1.0, "metadata": self.metadata[name]} 
                   for name in self.search(query)]
        
        try:
            # Vectorize the query
            query_vector = self.vectorizer.vectorize_query(query)
            
            # Calculate similarity with all function vectors
            similarities = []
            for func_name, vector in self.vectors.items():
                # Base similarity from vectors
                vector_similarity = self.vectorizer.calculate_similarity(query_vector, vector)
                
                # Boost similarity for matching code patterns
                pattern_boost = 0.0
                if query_patterns and "code_patterns" in self.metadata[func_name]:
                    func_patterns = self.metadata[func_name]["code_patterns"]
                    for pattern, wanted in query_patterns.items():
                        if pattern in func_patterns and func_patterns[pattern] == wanted:
                            pattern_boost += 0.1  # Boost for each matching pattern
                
                # Combined similarity score
                adjusted_similarity = min(1.0, vector_similarity + pattern_boost)
                
                similarities.append((func_name, adjusted_similarity, vector_similarity < adjusted_similarity))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k results with metadata
            results = []
            for func_name, similarity, matched_on_code in similarities[:top_k]:
                results.append({
                    "name": func_name,
                    "similarity": similarity,
                    "metadata": self.metadata[func_name],
                    "matched_on_code": matched_on_code
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def _extract_patterns_from_query(self, query: str) -> Dict[str, bool]:
        """
        Extract likely code patterns from a natural language query.
        
        Args:
            query: The natural language query
            
        Returns:
            Dictionary of likely code patterns based on query
        """
        patterns = {}
        
        # Look for transformation hints
        if any(term in query.lower() for term in ["calculate", "compute", "add", "sum", "multiply"]):
            patterns["transforms_data"] = True
            
        # Look for filtering hints
        if any(term in query.lower() for term in ["filter", "find", "where", "condition"]):
            patterns["filters_data"] = True
        
        # Look for I/O hints
        if any(term in query.lower() for term in ["file", "read", "write", "save", "load"]):
            patterns["performs_io"] = True
        
        # Look for error handling hints
        if any(term in query.lower() for term in ["error", "exception", "handle", "try"]):
            patterns["handles_errors"] = True
            
        # Look for string processing hints
        if any(term in query.lower() for term in ["string", "text", "format", "concat"]):
            patterns["manipulates_strings"] = True
            
        # Look for list processing hints
        if any(term in query.lower() for term in ["list", "array", "collection", "items"]):
            patterns["uses_functional_patterns"] = True
            
        return patterns


# Decorator for easy registration
def register_function(registry):
    """
    Decorator to register a function with a registry.
    
    Args:
        registry: FunctionRegistry instance
        
    Returns:
        Decorator function
    """
    def decorator(func):
        registry.register(func)
        return func
    return decorator