import inspect
import time
from typing import Dict, Any, Callable, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("function_registry")

class FunctionRegistry:
    """Registry for storing and executing functions with metadata tracking."""
    
    def __init__(self):
        """Initialize an empty function registry."""
        self.functions = {}  # Store actual function objects
        self.metadata = {}   # Store function metadata
        self.stats = {}      # Store execution statistics
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