#vectorizer.py
import inspect
import logging
import ast
from typing import Dict, Any, Callable, List, Optional, Union, Set
import numpy as np

logger = logging.getLogger("vectorizer")

class FunctionVectorizer:
    """Vectorizes functions based on their code implementation for semantic search."""
    
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
    
    def get_function_code_representation(self, func: Callable) -> str:
        """
        Extract and process the actual implementation code of a function.
        
        Args:
            func: The function to analyze
            
        Returns:
            A string representation focusing on implementation details
        """
        try:
            # Get the source code
            source_lines = inspect.getsourcelines(func)[0]
            
            # Remove function definition line and dedent
            implementation_code = "".join(source_lines[1:])
            implementation_code = inspect.cleandoc(implementation_code)
            
            # Extract abstract syntax tree for deeper analysis
            tree = ast.parse(implementation_code)
            
            # Process AST to extract key operations, control flow, and data structures
            operations = self._extract_operations_from_ast(tree)
            
            # Create a processed representation that focuses on semantic code elements
            representation = f"""
            Function Implementation:
            {implementation_code}
            
            Key Operations: {', '.join(operations)}
            """
            
            return representation
        except Exception as e:
            logger.error(f"Error extracting code representation: {str(e)}")
            return ""
    
    def _extract_operations_from_ast(self, tree) -> List[str]:
        """
        Extract important operations and patterns from code AST.
        
        Args:
            tree: The AST of the function code
            
        Returns:
            List of operation descriptors found in the code
        """
        operations = []
        
        # AST visitor to extract meaningful code operations
        class OperationVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                # Capture function calls
                if isinstance(node.func, ast.Name):
                    operations.append(f"calls_{node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    operations.append(f"method_{node.func.attr}")
                self.generic_visit(node)
                
            def visit_BinOp(self, node):
                # Capture math/string operations
                op_type = type(node.op).__name__
                operations.append(f"operation_{op_type}")
                self.generic_visit(node)
                
            def visit_Compare(self, node):
                # Capture comparisons
                for op in node.ops:
                    op_type = type(op).__name__
                    operations.append(f"comparison_{op_type}")
                self.generic_visit(node)
                
            def visit_If(self, node):
                operations.append("control_if")
                self.generic_visit(node)
                
            def visit_For(self, node):
                operations.append("control_loop")
                self.generic_visit(node)
                
            def visit_While(self, node):
                operations.append("control_loop")
                self.generic_visit(node)
                
            def visit_Try(self, node):
                operations.append("control_exception")
                self.generic_visit(node)
                
            def visit_Return(self, node):
                operations.append("return")
                self.generic_visit(node)
                
            def visit_List(self, node):
                operations.append("data_list")
                self.generic_visit(node)
                
            def visit_Dict(self, node):
                operations.append("data_dict")
                self.generic_visit(node)
                
            def visit_ListComp(self, node):
                operations.append("comprehension_list")
                self.generic_visit(node)
                
            def visit_DictComp(self, node):
                operations.append("comprehension_dict")
                self.generic_visit(node)
        
        visitor = OperationVisitor()
        visitor.visit(tree)
        
        # Return unique operations
        return list(set(operations))
    
    def analyze_code_patterns(self, func: Callable) -> Dict[str, bool]:
        """
        Extract semantic patterns from code implementation.
        
        Args:
            func: The function to analyze
            
        Returns:
            Dictionary of identified code patterns
        """
        patterns = {}
        
        try:
            # Get source code
            source = inspect.getsource(func)
            
            # Identify data transformation patterns
            if "return" in source and any(op in source for op in ["+", "-", "*", "/"]):
                patterns["transforms_data"] = True
                
            # Identify data filtering patterns
            if any(p in source for p in ["if", "filter", "where"]):
                patterns["filters_data"] = True
                
            # Identify I/O operations
            if any(p in source for p in ["open(", "read", "write", "load", "save"]):
                patterns["performs_io"] = True
                
            # Identify error handling
            if "try:" in source:
                patterns["handles_errors"] = True
            
            # Check for pure function patterns (no side effects)
            if not any(p in source for p in ["print", "global", "nonlocal"]):
                patterns["pure_function"] = True
                
            # Check for list comprehensions or functional patterns
            if any(p in source for p in ["[", "for", "in", "map", "filter", "reduce"]):
                patterns["uses_functional_patterns"] = True
                
            # Check for string manipulation
            if any(p in source for p in [".split", ".join", ".replace", ".format", "f\""]):
                patterns["manipulates_strings"] = True
            
            # Check for numeric operations
            if any(p in source for p in ["sum(", "min(", "max(", "len(", "range("]):
                patterns["numeric_operations"] = True
                
            return patterns
        except Exception as e:
            logger.error(f"Error analyzing code patterns: {str(e)}")
            return {}
    
    def vectorize_function(self, func: Callable) -> np.ndarray:
        """
        Create a vector embedding focused on function implementation.
        
        Args:
            func: The function to vectorize
            
        Returns:
            A numpy array containing the vector embedding
        """
        if self.model is None:
            # Fallback to simple bag-of-words if no model
            return self._simple_embedding(self.get_function_text(func))
            
        # Get code-focused representation
        code_representation = self.get_function_code_representation(func)
        
        # Get basic metadata (reduced weight compared to code)
        metadata_representation = f"""
        Function Name: {func.__name__}
        Parameters: {', '.join(inspect.signature(func).parameters.keys())}
        """
        
        # Give higher weight to code representation (3:1 ratio)
        combined_representation = f"{code_representation}\n{code_representation}\n{code_representation}\n{metadata_representation}"
        
        # Vectorize the combined representation
        return self.model.encode(combined_representation)
    
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
            vector = vector / np.linalg.norm(vector)
            
        return vector