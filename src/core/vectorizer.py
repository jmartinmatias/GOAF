#vectorizer.py
import inspect
import logging
import ast, re
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
    
class AlgorithmicVectorizer(FunctionVectorizer):
    """Vectorizes functions with deep algorithmic understanding."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__(model_name)
        # Initialize algorithm pattern templates
        self.algorithm_patterns = self._load_algorithm_patterns()
        # Initialize complexity analyzer
        self.complexity_analyzer = ComplexityAnalyzer()
        
    def analyze_algorithmic_purpose(self, func: Callable) -> Dict[str, Any]:
        """
        Analyze a function to understand its algorithmic purpose and behavior.
        
        Args:
            func: The function to analyze
            
        Returns:
            Dictionary with algorithmic purpose analysis
        """
        # Get source and AST
        source = inspect.getsource(func)
        tree = ast.parse(source)
        
        # Program flow analysis
        flow_graph = self._build_control_flow_graph(tree)
        
        # Data flow analysis
        data_flows = self._analyze_data_flow(tree)
        
        # Algorithmic pattern detection
        algorithm_matches = self._detect_algorithm_patterns(tree, flow_graph)
        
        # Complexity analysis
        complexity = self.complexity_analyzer.estimate_complexity(tree)
        
        # Input-output relationship analysis
        io_relationship = self._analyze_io_relationship(func, tree)
        
        # Code purpose detection
        block_purposes = self._detect_code_block_purposes(tree)
        
        return {
            "algorithm_patterns": algorithm_matches,
            "complexity": complexity,
            "data_flows": data_flows,
            "control_flow": self._summarize_control_flow(flow_graph),
            "io_relationship": io_relationship,
            "block_purposes": block_purposes,
            "algorithmic_categories": self._categorize_algorithm(algorithm_matches, flow_graph)
        }
    
    def _build_control_flow_graph(self, tree) -> Dict:
        """Build a control flow graph from AST."""
        flow_graph = {"nodes": [], "edges": []}
        
        # Visitor to extract control flow
        class FlowVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_node = 0
                self.node_map = {}
            
            def visit_If(self, node):
                # Create nodes for if condition and branches
                cond_node = self.current_node
                flow_graph["nodes"].append({"id": cond_node, "type": "if_condition"})
                self.node_map[node] = cond_node
                self.current_node += 1
                
                # Process then branch
                then_node = self.current_node
                flow_graph["nodes"].append({"id": then_node, "type": "then_branch"})
                flow_graph["edges"].append({"from": cond_node, "to": then_node, "condition": "true"})
                self.current_node += 1
                
                # Visit then branch body
                for child in node.body:
                    self.visit(child)
                    if hasattr(child, 'id') and child in self.node_map:
                        flow_graph["edges"].append({"from": then_node, "to": self.node_map[child]})
                
                # Process else branch if exists
                if node.orelse:
                    else_node = self.current_node
                    flow_graph["nodes"].append({"id": else_node, "type": "else_branch"})
                    flow_graph["edges"].append({"from": cond_node, "to": else_node, "condition": "false"})
                    self.current_node += 1
                    
                    for child in node.orelse:
                        self.visit(child)
                        if hasattr(child, 'id') and child in self.node_map:
                            flow_graph["edges"].append({"from": else_node, "to": self.node_map[child]})
            
            def visit_For(self, node):
                # Create nodes for loop
                loop_node = self.current_node
                flow_graph["nodes"].append({"id": loop_node, "type": "for_loop"})
                self.node_map[node] = loop_node
                self.current_node += 1
                
                # Create body node
                body_node = self.current_node  
                flow_graph["nodes"].append({"id": body_node, "type": "loop_body"})
                flow_graph["edges"].append({"from": loop_node, "to": body_node, "type": "iteration"})
                flow_graph["edges"].append({"from": body_node, "to": loop_node, "type": "loop_back"})
                self.current_node += 1
                
                # Visit loop body
                for child in node.body:
                    self.visit(child)
        
        # Run visitor
        visitor = FlowVisitor()
        visitor.visit(tree)
        
        return flow_graph
    
    def _detect_algorithm_patterns(self, tree, flow_graph) -> List[Dict[str, Any]]:
        """Detect common algorithm patterns in the code."""
        # Set up algorithm pattern matchers
        matches = []
        
        # Search for patterns
        for pattern in self.algorithm_patterns:
            if self._match_algorithm_pattern(tree, pattern):
                matches.append({
                    "name": pattern["name"],
                    "category": pattern["category"],
                    "confidence": pattern["match_confidence"],
                    "description": pattern["description"]
                })
        
        # Detect sorting algorithms
        if self._detect_sorting_algorithm(tree):
            matches.append({
                "name": "sorting_algorithm", 
                "category": "array_processing",
                "confidence": 0.85,
                "description": "Algorithm that sorts a collection of elements"
            })
            
        # Detect recursive algorithms
        if self._detect_recursion(tree):
            matches.append({
                "name": "recursive_algorithm", 
                "category": "control_flow",
                "confidence": 0.9,
                "description": "Algorithm that uses recursion to solve problems"
            })
            
        # Detect dynamic programming
        if self._detect_dynamic_programming(tree):
            matches.append({
                "name": "dynamic_programming", 
                "category": "optimization",
                "confidence": 0.7,
                "description": "Algorithm using memoization or tabulation for optimization"
            })
            
        return matches
    
    def _detect_code_block_purposes(self, tree) -> List[Dict[str, str]]:
        """Detect the purpose of different code blocks."""
        purposes = []
        
        # Visitor to analyze block purposes
        class PurposeVisitor(ast.NodeVisitor):
            def visit_If(self, node):
                # Analyze if condition
                condition_purpose = self._analyze_condition_purpose(node.test)
                purposes.append({
                    "type": "if_block",
                    "purpose": f"Conditionally execute code based on {condition_purpose}"
                })
                self.generic_visit(node)
                
            def visit_For(self, node):
                # Analyze loop purpose
                if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                    if node.iter.func.id == 'range':
                        purposes.append({
                            "type": "for_loop",
                            "purpose": "Iterate over a numeric range"
                        })
                    else:
                        purposes.append({
                            "type": "for_loop",
                            "purpose": f"Iterate over elements from {node.iter.func.id}"
                        })
                else:
                    purposes.append({
                        "type": "for_loop",
                        "purpose": "Iterate over a collection"
                    })
                self.generic_visit(node)
                
            def visit_Try(self, node):
                # Analyze try-except blocks
                purposes.append({
                    "type": "try_block",
                    "purpose": "Handle potential errors"
                })
                self.generic_visit(node)
                
            def visit_FunctionDef(self, node):
                # Skip the function definition itself
                for child in node.body:
                    self.visit(child)
                    
            def _analyze_condition_purpose(self, condition):
                """Analyze the purpose of a condition expression."""
                if isinstance(condition, ast.Compare):
                    # Equality check
                    if isinstance(condition.ops[0], ast.Eq):
                        return "equality check"
                    # Range check
                    elif isinstance(condition.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                        return "range check"
                    
                return "condition check"
        
        # Run visitor
        visitor = PurposeVisitor()
        visitor.visit(tree)
        
        return purposes
    
    def _analyze_io_relationship(self, func, tree) -> Dict[str, Any]:
        """Analyze the relationship between inputs and outputs."""
        result = {
            "input_dependencies": {},
            "output_determinism": "deterministic",
            "side_effects": [],
            "input_validation": False,
            "output_transformation_type": "unknown"
        }
        
        # Get parameter names
        params = list(inspect.signature(func).parameters.keys())
        
        # Check which parameters affect the output
        for param in params:
            # This is a simplified analysis - in reality, this would require
            # sophisticated program slicing techniques
            result["input_dependencies"][param] = "required"
        
        # Detect side effects
        class SideEffectVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in ["print", "write", "open"]:
                        result["side_effects"].append(func_name)
                        result["output_determinism"] = "non-deterministic"
                self.generic_visit(node)
        
        visitor = SideEffectVisitor()
        visitor.visit(tree)
        
        # Detect input validation
        if any("raise" in line for line in inspect.getsource(func).split("\n")):
            result["input_validation"] = True
        
        # Analyze output transformation
        output_classifications = []
        
        # Check if this is a transformation function
        if any(op in inspect.getsource(func) for op in ["+", "-", "*", "/"]):
            output_classifications.append("mathematical_transformation")
            
        if "return" in inspect.getsource(func) and any(x in inspect.getsource(func) for x in ["list", "[]", "dict", "{}"]):
            output_classifications.append("data_structure_transformation")
            
        if "return" in inspect.getsource(func) and any(x in inspect.getsource(func) for x in ["join", "split", "replace"]):
            output_classifications.append("string_transformation")
            
        if output_classifications:
            result["output_transformation_type"] = output_classifications
        
        return result
    
    def _load_algorithm_patterns(self) -> List[Dict[str, Any]]:
        """Load predefined algorithm patterns."""
        return [
            {
                "name": "binary_search",
                "category": "search",
                "match_confidence": 0.8,
                "description": "Efficient search algorithm that works on sorted arrays",
                "pattern": {
                    "operations": ["divide_and_conquer", "array_access"],
                    "flow": ["conditional_loop"]
                }
            },
            {
                "name": "merge_sort",
                "category": "sorting",
                "match_confidence": 0.8,
                "description": "Efficient sorting algorithm using divide-and-conquer",
                "pattern": {
                    "operations": ["recursive_calls", "array_splitting", "merging"],
                    "flow": ["recursion"]
                }
            },
            {
                "name": "breadth_first_search",
                "category": "graph_traversal",
                "match_confidence": 0.8,
                "description": "Level-by-level graph traversal algorithm",
                "pattern": {
                    "operations": ["queue_operations", "node_visits"],
                    "flow": ["queue_based_loop"]
                }
            },
            {
                "name": "depth_first_search",
                "category": "graph_traversal",
                "match_confidence": 0.8,
                "description": "Deep-first graph traversal algorithm",
                "pattern": {
                    "operations": ["stack_operations", "node_visits"],
                    "flow": ["stack_based_recursion"]
                }
            },
            {
                "name": "map_reduce",
                "category": "data_processing",
                "match_confidence": 0.7,
                "description": "Processes and aggregates data in two phases",
                "pattern": {
                    "operations": ["mapping_function", "reduce_function"],
                    "flow": ["collection_transformation"]
                }
            }
        ]
        
    def _detect_sorting_algorithm(self, tree) -> bool:
        """Detect if the code implements a sorting algorithm."""
        # This is a simplified detection - would need much more sophisticated analysis
        source = ast.unparse(tree)
        
        # Check for sorting algorithm keywords
        sorting_indicators = [
            "sort", "sorted", "order", "ascending", "descending", "compare"
        ]
        
        # Check for nested loops (common in sorting)
        has_nested_loops = False
        
        class NestedLoopVisitor(ast.NodeVisitor):
            def __init__(self):
                self.loop_depth = 0
                self.max_depth = 0
                
            def visit_For(self, node):
                self.loop_depth += 1
                self.max_depth = max(self.max_depth, self.loop_depth)
                self.generic_visit(node)
                self.loop_depth -= 1
                
            def visit_While(self, node):
                self.loop_depth += 1
                self.max_depth = max(self.max_depth, self.loop_depth)
                self.generic_visit(node)
                self.loop_depth -= 1
        
        visitor = NestedLoopVisitor()
        visitor.visit(tree)
        has_nested_loops = visitor.max_depth >= 2
        
        # Look for sorting keywords and patterns
        has_sorting_keywords = any(word in source.lower() for word in sorting_indicators)
        
        return has_nested_loops and has_sorting_keywords
    
    def _detect_recursion(self, tree) -> bool:
        """Detect if the function uses recursion."""
        # Get function name
        function_name = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                break
        
        if not function_name:
            return False
        
        # Check if function calls itself
        class RecursionVisitor(ast.NodeVisitor):
            def __init__(self, func_name):
                self.func_name = func_name
                self.has_recursion = False
                
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == self.func_name:
                    self.has_recursion = True
                self.generic_visit(node)
        
        visitor = RecursionVisitor(function_name)
        visitor.visit(tree)
        
        return visitor.has_recursion
    
    def _detect_dynamic_programming(self, tree) -> bool:
        """Detect if the function uses dynamic programming patterns."""
        source = ast.unparse(tree)
        
        # Check for memoization patterns (storing results)
        has_memoization = False
        
        # Look for dictionary/array being updated with results
        memoization_patterns = [
            r"\[\w+\]\s*=", # Array assignment
            r"\[\w+\]\[\w+\]\s*=", # 2D array assignment
            r"memo\[\w+\]\s*=", # Common memoization variable
            r"cache\[\w+\]\s*=", # Common cache variable
            r"dp\[\w+\]\s*=", # Common dynamic programming variable
        ]
        
        for pattern in memoization_patterns:
            if re.search(pattern, source):
                has_memoization = True
                break
        
        # Check for tabulation pattern (filling array iteratively)
        has_tabulation = False
        
        class TabulationVisitor(ast.NodeVisitor):
            def visit_For(self, node):
                # Check if there's array assignment inside the loop
                for child in ast.walk(node):
                    if isinstance(child, ast.Subscript) and isinstance(child.ctx, ast.Store):
                        has_tabulation = True
                        break
                self.generic_visit(node)
        
        visitor = TabulationVisitor()
        visitor.visit(tree)
        
        return has_memoization or has_tabulation
    
    def _match_algorithm_pattern(self, tree, pattern) -> bool:
        """Match a specific algorithm pattern against the AST."""
        # This would be a sophisticated pattern matching algorithm
        # For now, we'll use a simplified approach
        source = ast.unparse(tree)
        
        # Check operations
        operations_match = all(op in source.lower() for op in pattern["pattern"]["operations"])
        
        # Check flow patterns
        flow_match = all(flow in source.lower() for flow in pattern["pattern"]["flow"])
        
        return operations_match and flow_match
    
    def _summarize_control_flow(self, flow_graph) -> Dict[str, int]:
        """Summarize control flow characteristics."""
        summary = {
            "if_statements": 0,
            "loops": 0,
            "try_blocks": 0,
            "nested_depth": 0
        }
        
        for node in flow_graph["nodes"]:
            if "if" in node["type"]:
                summary["if_statements"] += 1
            elif "loop" in node["type"]:
                summary["loops"] += 1
        
        # Calculate maximum nesting depth
        # This would require traversing the flow graph to find longest path
        
        return summary
    
    def _analyze_data_flow(self, tree) -> Dict[str, Any]:
        """Analyze data flow patterns in the code."""
        data_flow = {
            "input_transformations": [],
            "variable_dependencies": {},
            "data_structures_used": []
        }
        
        # Detect data structures
        class DataStructureVisitor(ast.NodeVisitor):
            def visit_List(self, node):
                if "list" not in data_flow["data_structures_used"]:
                    data_flow["data_structures_used"].append("list")
                self.generic_visit(node)
                
            def visit_Dict(self, node):
                if "dict" not in data_flow["data_structures_used"]:
                    data_flow["data_structures_used"].append("dict")
                self.generic_visit(node)
                
            def visit_Set(self, node):
                if "set" not in data_flow["data_structures_used"]:
                    data_flow["data_structures_used"].append("set")
                self.generic_visit(node)
                
            def visit_Tuple(self, node):
                if "tuple" not in data_flow["data_structures_used"]:
                    data_flow["data_structures_used"].append("tuple")
                self.generic_visit(node)
        
        visitor = DataStructureVisitor()
        visitor.visit(tree)
        
        # Detect common transformations
        class TransformationVisitor(ast.NodeVisitor):
            def visit_BinOp(self, node):
                op_type = type(node.op).__name__
                transformation = f"operation_{op_type}"
                if transformation not in data_flow["input_transformations"]:
                    data_flow["input_transformations"].append(transformation)
                self.generic_visit(node)
                
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in ["map", "filter", "reduce"]:
                        transformation = f"functional_{func_name}"
                        if transformation not in data_flow["input_transformations"]:
                            data_flow["input_transformations"].append(transformation)
                    elif func_name in ["sum", "max", "min"]:
                        transformation = f"aggregation_{func_name}"
                        if transformation not in data_flow["input_transformations"]:
                            data_flow["input_transformations"].append(transformation)
                self.generic_visit(node)
        
        visitor = TransformationVisitor()
        visitor.visit(tree)
        
        return data_flow
    
    def _categorize_algorithm(self, algorithm_matches, flow_graph) -> List[str]:
        """Categorize the algorithm based on detected patterns and flow."""
        categories = set()
        
        # Add categories from algorithm matches
        for match in algorithm_matches:
            categories.add(match["category"])
        
        # Analyze flow graph for additional categories
        has_loops = any("loop" in node["type"] for node in flow_graph["nodes"])
        has_conditionals = any("if" in node["type"] for node in flow_graph["nodes"])
        
        if has_loops:
            categories.add("iterative")
        if has_conditionals:
            categories.add("conditional")
            
        return list(categories)

    def vectorize_function(self, func: Callable) -> Dict[str, Any]:
        """
        Create a comprehensive vector embedding focused on algorithmic understanding.
        
        Args:
            func: The function to vectorize
            
        Returns:
            A dictionary with both the embedding vector and algorithmic analysis
        """
        # Get standard code embedding
        code_embedding = super().vectorize_function(func)
        
        # Get algorithmic analysis
        algorithmic_analysis = self.analyze_algorithmic_purpose(func)
        
        # Create enhanced textual representation for embedding
        algorithmic_text = self._generate_algorithmic_description(func, algorithmic_analysis)
        
        # Create algorithm-focused embedding
        algorithmic_embedding = self.model.encode(algorithmic_text)
        
        # Combine embeddings (weighted more toward algorithmic understanding)
        combined_embedding = 0.3 * code_embedding + 0.7 * algorithmic_embedding
        
        # Normalize the combined embedding
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
        
        return {
            "vector": combined_embedding,
            "algorithmic_analysis": algorithmic_analysis,
            "algorithmic_description": algorithmic_text
        }
    
    def _generate_algorithmic_description(self, func: Callable, analysis: Dict[str, Any]) -> str:
        """Generate a detailed algorithmic description from the analysis."""
        # Get function basics
        func_name = func.__name__
        docstring = inspect.getdoc(func) or ""
        
        # Build algorithmic description
        description = [
            f"Function '{func_name}' algorithm analysis:",
            f"Purpose: {docstring}",
            ""
        ]
        
        # Add algorithm pattern information
        if analysis["algorithm_patterns"]:
            description.append("Algorithm patterns detected:")
            for pattern in analysis["algorithm_patterns"]:
                description.append(f"- {pattern['name']}: {pattern['description']} (confidence: {pattern['confidence']})")
            description.append("")
        
        # Add complexity information
        description.append(f"Algorithmic complexity: {analysis['complexity']['time_complexity']}")
        description.append(f"Space complexity: {analysis['complexity']['space_complexity']}")
        description.append("")
        
        # Add data flow information
        description.append("Data processing characteristics:")
        if analysis["data_flows"]["data_structures_used"]:
            description.append(f"- Uses data structures: {', '.join(analysis['data_flows']['data_structures_used'])}")
        if analysis["data_flows"]["input_transformations"]:
            description.append(f"- Transformations: {', '.join(analysis['data_flows']['input_transformations'])}")
        description.append("")
        
        # Add control flow information
        description.append("Control flow characteristics:")
        for key, value in analysis["control_flow"].items():
            description.append(f"- {key.replace('_', ' ').title()}: {value}")
        description.append("")
        
        # Add input-output relationship information
        description.append("Input-output relationship:")
        description.append(f"- Determinism: {analysis['io_relationship']['output_determinism']}")
        if analysis["io_relationship"]["side_effects"]:
            description.append(f"- Side effects: {', '.join(analysis['io_relationship']['side_effects'])}")
        if analysis["io_relationship"]["input_validation"]:
            description.append("- Performs input validation")
        if isinstance(analysis["io_relationship"]["output_transformation_type"], list):
            description.append(f"- Transformation types: {', '.join(analysis['io_relationship']['output_transformation_type'])}")
        description.append("")
        
        # Add block purposes
        if analysis["block_purposes"]:
            description.append("Code block purposes:")
            for purpose in analysis["block_purposes"]:
                description.append(f"- {purpose['type']}: {purpose['purpose']}")
        
        return "\n".join(description)


class ComplexityAnalyzer:
    """Analyzes algorithmic complexity of code."""
    
    def estimate_complexity(self, tree) -> Dict[str, str]:
        """
        Estimate the time and space complexity of an algorithm.
        
        Args:
            tree: AST of the function
            
        Returns:
            Dictionary with complexity estimations
        """
        # Count nested loops to estimate time complexity
        max_loop_depth = self._get_max_loop_depth(tree)
        
        # Estimate complexity based on loop nesting
        if max_loop_depth == 0:
            time_complexity = "O(1)"  # Constant time
        elif max_loop_depth == 1:
            time_complexity = "O(n)"  # Linear time
        elif max_loop_depth == 2:
            time_complexity = "O(n²)"  # Quadratic time
        elif max_loop_depth == 3:
            time_complexity = "O(n³)"  # Cubic time
        else:
            time_complexity = f"O(n^{max_loop_depth})"  # Polynomial time
        
        # Check for recursion to refine complexity estimate
        if self._has_recursion(tree):
            time_complexity = "O(2^n)"  # Exponential time (simplified assumption)
        
        # Check for divide and conquer patterns
        if self._has_divide_and_conquer(tree):
            time_complexity = "O(n log n)"  # Typical for efficient divide and conquer
        
        # Estimate space complexity
        space_complexity = self._estimate_space_complexity(tree)
        
        return {
            "time_complexity": time_complexity,
            "space_complexity": space_complexity
        }
    
    def _get_max_loop_depth(self, tree) -> int:
        """Get the maximum nesting depth of loops in the code."""
        max_depth = 0
        current_depth = 0
        
        class LoopVisitor(ast.NodeVisitor):
            def visit_For(self, node):
                nonlocal current_depth, max_depth
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                self.generic_visit(node)
                current_depth -= 1
                
            def visit_While(self, node):
                nonlocal current_depth, max_depth
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                self.generic_visit(node)
                current_depth -= 1
        
        visitor = LoopVisitor()
        visitor.visit(tree)
        
        return max_depth
    
    def _has_recursion(self, tree) -> bool:
        """Check if the function contains recursive calls."""
        function_name = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                break
        
        if not function_name:
            return False
        
        has_recursion = False
        
        class RecursionVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                nonlocal has_recursion
                if isinstance(node.func, ast.Name) and node.func.id == function_name:
                    has_recursion = True
                self.generic_visit(node)
        
        visitor = RecursionVisitor()
        visitor.visit(tree)
        
        return has_recursion
    
    def _has_divide_and_conquer(self, tree) -> bool:
        """Check if the function uses divide and conquer strategies."""
        # This is a simplified detector - a real one would be more sophisticated
        source = ast.unparse(tree)
        
        # Look for recursive calls with divided input
        divide_patterns = [
            r"(\w+)\(.*?\/.*?\)",  # Function call with division
            r"(\w+)\(.*?\[:\w+\/\d+\].*?\)",  # Function call with sliced array to midpoint
            r"(\w+)\(.*?\[:\w+\/\/\d+\].*?\)",  # Function call with integer division slicing
            r"mid\s*=.*?(\/\/|\/)\s*2",  # Mid-point calculation
        ]
        
        for pattern in divide_patterns:
            if re.search(pattern, source):
                return True
        
        return False
    
    def _estimate_space_complexity(self, tree) -> str:
        """Estimate the space complexity of the algorithm."""
        # Count variables and data structures
        variable_count = 0
        has_recursive_calls = False
        creates_large_structures = False
        
        class SpaceVisitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                nonlocal variable_count, creates_large_structures
                variable_count += len(node.targets)
                
                # Check if assigning a large structure
                if isinstance(node.value, (ast.List, ast.Dict, ast.Set)):
                    creates_large_structures = True
                elif isinstance(node.value, ast.ListComp):
                    creates_large_structures = True
                elif isinstance(node.value, ast.Call):
                    if hasattr(node.value.func, 'id') and node.value.func.id in ['list', 'dict', 'set']:
                        creates_large_structures = True
                
                self.generic_visit(node)
                
            def visit_Call(self, node):
                nonlocal has_recursive_calls
                if isinstance(node.func, ast.Name) and node.func.id == tree.body[0].name:
                    has_recursive_calls = True
                self.generic_visit(node)
        
        visitor = SpaceVisitor()
        visitor.visit(tree)
        
        # Estimate complexity based on variables and structures
        if has_recursive_calls:
            return "O(n)"  # Recursive call stack
        elif creates_large_structures:
            return "O(n)"  # Creates data structures proportional to input
        elif variable_count > 10:
            return "O(1) with high constant factor"  # Many variables but constant
        else:
            return "O(1)"  # Constant space