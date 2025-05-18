#app.py
import streamlit as st
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.core.registry import FunctionRegistry
from example_functions import *

# Set page configuration
st.set_page_config(
    page_title="Algorithmic Function Explorer",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .algorithm-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .algorithm-header {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .complexity-badge {
        background-color: #f0f0f0;
        border-radius: 4px;
        padding: 2px 8px;
        margin-right: 5px;
        font-family: monospace;
    }
    .complexity-badge.linear {
        background-color: #d1f5d3;
    }
    .complexity-badge.quadratic {
        background-color: #fff3cd;
    }
    .complexity-badge.exponential {
        background-color: #f8d7da;
    }
    .pattern-label {
        display: inline-block;
        background-color: #e9ecef;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
    }
    .stAlert {
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the registry
@st.cache_resource
def get_registry():
    registry = FunctionRegistry()
    
    # Register a variety of functions with different algorithmic properties
    registry.register(add)
    registry.register(subtract)
    registry.register(multiply)
    registry.register(divide)
    registry.register(greet)
    registry.register(count_words)
    registry.register(is_palindrome)
    registry.register(capitalize_words)
    registry.register(fetch_webpage)
    registry.register(send_email)
    registry.register(calculate_average)
    registry.register(filter_even_numbers)
    registry.register(sort_numbers)
    registry.register(find_max)
    registry.register(find_min)
    registry.register(read_file_lines)
    registry.register(write_to_file)
    registry.register(parse_json)
    registry.register(format_date)
    registry.register(check_url_status)
    registry.register(binary_search)
    registry.register(merge_sort)
    registry.register(quick_sort)
    registry.register(depth_first_search)
    registry.register(breadth_first_search)
    registry.register(fibonacci_recursive)
    registry.register(fibonacci_dynamic)
    registry.register(knapsack_problem)
    registry.register(levenshtein_distance)
    registry.register(is_prime)
    
    return registry

registry = get_registry()

# Streamlit App
st.title("Algorithmic Function Explorer")
st.markdown("Discover, analyze, and understand functions through their algorithmic properties")

# Sidebar menu
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Select page", [
        "Function Overview", 
        "Algorithmic Search", 
        "Algorithm Comparison", 
        "Function Execution",
        "Pattern Explorer"
    ])
    
    st.markdown("---")
    
    # Filter options
    st.subheader("Filter Functions")
    
    # Get all available categories from the registry
    all_categories = set()
    for func_name in registry.list_functions():
        metadata = registry.get_metadata(func_name)
        if "algorithmic_analysis" in metadata:
            analysis = metadata["algorithmic_analysis"]
            if "algorithm_patterns" in analysis:
                for pattern in analysis["algorithm_patterns"]:
                    all_categories.add(pattern["category"])
    
    if all_categories:
        selected_categories = st.multiselect("By Algorithm Category", list(all_categories))
    
    # Complexity filter
    complexity_options = [
        "O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n²)", "O(2^n)"
    ]
    selected_complexity = st.multiselect("By Time Complexity", complexity_options)
    
    # Reset filters button
    if st.button("Reset Filters"):
        selected_categories = []
        selected_complexity = []

    st.markdown("---")
    st.info(
        """
        **Algorithmic Function Explorer**
        
        A system that understands not just what functions do, but how they work algorithmically.
        """
    )

# Apply filters to function list
def filter_functions(functions):
    if not (selected_categories or selected_complexity):
        return functions
        
    filtered = []
    for func_name in functions:
        metadata = registry.get_metadata(func_name)
        include = True
        
        if "algorithmic_analysis" in metadata:
            analysis = metadata["algorithmic_analysis"]
            
            # Filter by category
            if selected_categories:
                category_match = False
                if "algorithm_patterns" in analysis:
                    for pattern in analysis["algorithm_patterns"]:
                        if pattern["category"] in selected_categories:
                            category_match = True
                            break
                if not category_match:
                    include = False
            
            # Filter by complexity
            if selected_complexity and include:
                complexity_match = False
                if "complexity" in analysis:
                    time_complexity = analysis["complexity"]["time_complexity"]
                    if time_complexity in selected_complexity:
                        complexity_match = True
                if not complexity_match:
                    include = False
        else:
            # If no analysis available, exclude if any filters are active
            if selected_categories or selected_complexity:
                include = False
                
        if include:
            filtered.append(func_name)
            
    return filtered

# Function to render algorithm analysis
def render_algorithm_analysis(metadata):
    if "algorithmic_analysis" not in metadata:
        st.warning("No algorithmic analysis available for this function")
        return
        
    analysis = metadata["algorithmic_analysis"]
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Algorithm patterns
        if "algorithm_patterns" in analysis and analysis["algorithm_patterns"]:
            st.subheader("Algorithm Patterns")
            for pattern in analysis["algorithm_patterns"]:
                confidence = int(pattern["confidence"] * 100)
                st.markdown(f"""
                <div class="algorithm-card">
                    <div class="algorithm-header">{pattern["name"].replace("_", " ").title()}</div>
                    <p>{pattern["description"]}</p>
                    <div>Category: <span class="pattern-label">{pattern["category"].replace("_", " ").title()}</span></div>
                    <div>Confidence: {confidence}%</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific algorithm patterns detected")
            
        # Block purposes
        if "block_purposes" in analysis and analysis["block_purposes"]:
            st.subheader("Code Block Purposes")
            for purpose in analysis["block_purposes"]:
                st.markdown(f"- **{purpose['type'].replace('_', ' ').title()}**: {purpose['purpose']}")
                
    with col2:
        # Complexity information
        if "complexity" in analysis:
            st.subheader("Algorithmic Complexity")
            
            # Apply styling based on complexity
            time_complexity = analysis["complexity"]["time_complexity"]
            time_class = "linear" if "n" in time_complexity and "²" not in time_complexity and "^" not in time_complexity else \
                       "quadratic" if "²" in time_complexity or "n^2" in time_complexity else \
                       "exponential" if "^" in time_complexity or "2^n" in time_complexity else ""
            
            space_complexity = analysis["complexity"]["space_complexity"]
            space_class = "linear" if "n" in space_complexity and "²" not in space_complexity and "^" not in space_complexity else \
                        "quadratic" if "²" in space_complexity or "n^2" in space_complexity else \
                        "exponential" if "^" in space_complexity or "2^n" in space_complexity else ""
            
            st.markdown(f"""
            <div>Time: <span class="complexity-badge {time_class}">{time_complexity}</span></div>
            <div>Space: <span class="complexity-badge {space_class}">{space_complexity}</span></div>
            """, unsafe_allow_html=True)
        
        # Data flow information
        if "data_flows" in analysis:
            st.subheader("Data Processing")
            
            if "data_structures_used" in analysis["data_flows"] and analysis["data_flows"]["data_structures_used"]:
                st.markdown("**Data Structures:**")
                for ds in analysis["data_flows"]["data_structures_used"]:
                    st.markdown(f"- {ds.title()}")
            
            if "input_transformations" in analysis["data_flows"] and analysis["data_flows"]["input_transformations"]:
                st.markdown("**Transformations:**")
                for transform in analysis["data_flows"]["input_transformations"]:
                    st.markdown(f"- {transform.replace('_', ' ').title()}")
        
        # I/O relationship
        if "io_relationship" in analysis:
            st.subheader("Input-Output Behavior")
            
            determinism = analysis["io_relationship"]["output_determinism"]
            st.markdown(f"**Determinism:** {determinism.title()}")
            
            if "side_effects" in analysis["io_relationship"] and analysis["io_relationship"]["side_effects"]:
                st.markdown("**Side Effects:**")
                for effect in analysis["io_relationship"]["side_effects"]:
                    st.markdown(f"- {effect}")
            
            if analysis["io_relationship"].get("input_validation", False):
                st.info("Performs input validation")
                
            if "output_transformation_type" in analysis["io_relationship"] and analysis["io_relationship"]["output_transformation_type"] != "unknown":
                if isinstance(analysis["io_relationship"]["output_transformation_type"], list):
                    st.markdown("**Transformation Types:**")
                    for transform in analysis["io_relationship"]["output_transformation_type"]:
                        st.markdown(f"- {transform.replace('_', ' ').title()}")
                else:
                    st.markdown(f"**Transformation Type:** {analysis['io_relationship']['output_transformation_type'].replace('_', ' ').title()}")
                    
        # Control flow summary
        if "control_flow" in analysis:
            st.subheader("Control Flow")
            for key, value in analysis["control_flow"].items():
                if value > 0:
                    st.markdown(f"- {key.replace('_', ' ').title()}: {value}")

# Function to display source code with syntax highlighting
def display_source_code(func):
    try:
        source = inspect.getsource(func)
        st.code(source, language="python")
    except Exception as e:
        st.error(f"Could not retrieve source code: {str(e)}")

# Pages
if page == "Function Overview":
    st.header("Function Catalog")
    
    # Get filtered functions
    func_names = filter_functions(registry.list_functions())
    
    if not func_names:
        st.warning("No functions match the selected filters")
    else:
        st.success(f"Showing {len(func_names)} functions")
        
        # Group functions by category
        categories = {}
        uncategorized = []
        
        for func_name in func_names:
            metadata = registry.get_metadata(func_name)
            cats = set()
            
            if "algorithmic_analysis" in metadata:
                analysis = metadata["algorithmic_analysis"]
                if "algorithm_patterns" in analysis:
                    for pattern in analysis["algorithm_patterns"]:
                        cats.add(pattern["category"])
            
            if cats:
                for cat in cats:
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append(func_name)
            else:
                uncategorized.append(func_name)
                
        # Display functions by category
        for category, funcs in categories.items():
            with st.expander(f"{category.replace('_', ' ').title()} ({len(funcs)} functions)"):
                for func_name in funcs:
                    metadata = registry.get_metadata(func_name)
                    with st.expander(f"{func_name}{metadata['signature']}"):
                        st.markdown(f"**Description:** {metadata['docstring']}")
                        
                        # Show complexity if available
                        if "algorithmic_analysis" in metadata and "complexity" in metadata["algorithmic_analysis"]:
                            time_complexity = metadata["algorithmic_analysis"]["complexity"]["time_complexity"]
                            st.markdown(f"**Time Complexity:** {time_complexity}")
                        
                        # Tabs for different views
                        tab1, tab2, tab3 = st.tabs(["Source Code", "Algorithm Analysis", "Execution Stats"])
                        
                        with tab1:
                            display_source_code(registry.functions[func_name])
                            
                        with tab2:
                            render_algorithm_analysis(metadata)
                            
                        with tab3:
                            stats = registry.get_stats(func_name)
                            st.markdown(f"**Called:** {stats['calls']} times")
                            st.markdown(f"**Success Rate:** {(stats['success'] / max(1, stats['calls'])) * 100:.1f}%")
                            st.markdown(f"**Average Execution Time:** {stats['avg_time']:.6f} seconds")
        
        # Display uncategorized functions
        if uncategorized:
            with st.expander(f"Other Functions ({len(uncategorized)})"):
                for func_name in uncategorized:
                    metadata = registry.get_metadata(func_name)
                    with st.expander(f"{func_name}{metadata['signature']}"):
                        st.markdown(f"**Description:** {metadata['docstring']}")
                        
                        # Tabs for different views
                        tab1, tab2 = st.tabs(["Source Code", "Execution Stats"])
                        
                        with tab1:
                            display_source_code(registry.functions[func_name])
                            
                        with tab2:
                            stats = registry.get_stats(func_name)
                            st.markdown(f"**Called:** {stats['calls']} times")
                            st.markdown(f"**Success Rate:** {(stats['success'] / max(1, stats['calls'])) * 100:.1f}%")
                            st.markdown(f"**Average Execution Time:** {stats['avg_time']:.6f} seconds")

elif page == "Algorithmic Search":
    st.header("Algorithmic Search")
    st.write("Find functions based on their algorithmic properties and implementation patterns")
    
    # Search options
    search_method = st.radio(
        "Search Method",
        ["Natural Language", "Algorithm Pattern", "Implementation Properties"],
        horizontal=True
    )
    
    if search_method == "Natural Language":
        query = st.text_input("What algorithm or function do you need?", placeholder="E.g., sort a list efficiently or find the shortest path")
        
        if query:
            results = registry.semantic_search(query, top_k=10)
            
            if results:
                st.success(f"Found {len(results)} relevant functions")
                
                # Display results in a more visual way
                for i, result in enumerate(results):
                    func_name = result["name"]
                    similarity = result["similarity"]
                    metadata = result["metadata"]
                    
                    # Create an expandable section for each result
                    with st.expander(f"{i+1}. {func_name}{metadata['signature']} ({similarity:.2f} relevance)"):
                        st.markdown(f"**Description:** {metadata['docstring']}")
                        
                        # Show why this matched
                        if result.get("matched_on_code"):
                            st.info("Matched based on implementation patterns")
                        
                        # Tabs for different views
                        tab1, tab2 = st.tabs(["Algorithm Analysis", "Source Code"])
                        
                        with tab1:
                            render_algorithm_analysis(metadata)
                            
                        with tab2:
                            display_source_code(registry.functions[func_name])
                            
                        # Execute button
                        if st.button(f"Execute {func_name}", key=f"exec_{func_name}_{i}"):
                            st.session_state["selected_func"] = func_name
                            st.session_state["page"] = "Function Execution"
                            st.experimental_rerun()
            else:
                st.warning("No functions found matching your query")
                
                # Suggest some searches
                st.subheader("Try searching for:")
                suggestions = [
                    "sorting algorithms",
                    "graph traversal",
                    "dynamic programming",
                    "string manipulation",
                    "recursive functions",
                    "mathematical operations"
                ]
                
                cols = st.columns(3)
                for i, suggestion in enumerate(suggestions):
                    col = cols[i % 3]
                    if col.button(suggestion, key=f"sug_{i}"):
                        st.session_state["query"] = suggestion
                        st.experimental_rerun()
    
    elif search_method == "Algorithm Pattern":
        # Search by specific algorithm patterns
        col1, col2 = st.columns(2)
        
        with col1:
            pattern_category = st.selectbox(
                "Algorithm Category",
                ["Any Category", "sorting", "search", "graph_traversal", "optimization", 
                 "data_processing", "string_processing", "numeric_operations"]
            )
            
        with col2:
            pattern_complexity = st.selectbox(
                "Time Complexity",
                ["Any Complexity", "O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n²)", "O(2^n)"]
            )
        
        key_operations = st.multiselect(
            "Key Operations",
            ["recursion", "iteration", "divide_and_conquer", "dynamic_programming", 
             "greedy", "backtracking", "comparison_based", "hashing"]
        )
        
        data_structure = st.selectbox(
            "Primary Data Structure",
            ["Any", "array/list", "dictionary/map", "string", "tree", "graph", "stack", "queue"]
        )
        
        if st.button("Find Matching Algorithms"):
            # Apply manual filters to find matching functions
            results = []
            
            for func_name in registry.list_functions():
                metadata = registry.get_metadata(func_name)
                
                if "algorithmic_analysis" not in metadata:
                    continue
                    
                analysis = metadata["algorithmic_analysis"]
                match = True
                
                # Check category
                if pattern_category != "Any Category":
                    category_match = False
                    if "algorithm_patterns" in analysis:
                        for pattern in analysis["algorithm_patterns"]:
                            if pattern["category"] == pattern_category:
                                category_match = True
                                break
                    if not category_match:
                        match = False
                
                # Check complexity
                if pattern_complexity != "Any Complexity" and match:
                    if "complexity" not in analysis or analysis["complexity"]["time_complexity"] != pattern_complexity:
                        match = False
                
                # Check key operations (simplified)
                if key_operations and match:
                    operations_match = False
                    for op in key_operations:
                        if "algorithmic_description" in metadata and op in metadata["algorithmic_description"].lower():
                            operations_match = True
                            break
                    if not operations_match:
                        match = False
                
                # Check data structure
                if data_structure != "Any" and match:
                    ds_match = False
                    if "data_flows" in analysis and "data_structures_used" in analysis["data_flows"]:
                        for ds in analysis["data_flows"]["data_structures_used"]:
                            if data_structure.lower() in ds.lower():
                                ds_match = True
                                break
                    if not ds_match:
                        match = False
                
                if match:
                    results.append({
                        "name": func_name,
                        "metadata": metadata,
                        "similarity": 1.0  # All matches are equally relevant in this case
                    })
            
            if results:
                st.success(f"Found {len(results)} matching functions")
                
                for i, result in enumerate(results):
                    func_name = result["name"]
                    metadata = result["metadata"]
                    
                    with st.expander(f"{i+1}. {func_name}{metadata['signature']}"):
                        st.markdown(f"**Description:** {metadata['docstring']}")
                        
                        # Tabs for different views
                        tab1, tab2 = st.tabs(["Algorithm Analysis", "Source Code"])
                        
                        with tab1:
                            render_algorithm_analysis(metadata)
                            
                        with tab2:
                            display_source_code(registry.functions[func_name])
            else:
                st.warning("No functions match the selected criteria")
    
    elif search_method == "Implementation Properties":
        # Search by specific code implementation properties
        st.subheader("Find Functions by Implementation Characteristics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            has_loops = st.checkbox("Contains Loops")
            has_recursion = st.checkbox("Uses Recursion")
            has_conditionals = st.checkbox("Has Conditional Logic")
            
        with col2:
            handles_errors = st.checkbox("Includes Error Handling")
            is_pure = st.checkbox("Is Pure Function (No Side Effects)")
            has_validation = st.checkbox("Performs Input Validation")
        
        if st.button("Find Matching Functions"):
            # Apply implementation filters
            results = []
            
            for func_name in registry.list_functions():
                metadata = registry.get_metadata(func_name)
                
                if "algorithmic_analysis" not in metadata:
                    continue
                    
                analysis = metadata["algorithmic_analysis"]
                match = True
                
                # Check loops
                if has_loops and match:
                    if "control_flow" not in analysis or analysis["control_flow"].get("loops", 0) == 0:
                        match = False
                
                # Check recursion
                if has_recursion and match:
                    recursion_found = False
                    if "algorithm_patterns" in analysis:
                        for pattern in analysis["algorithm_patterns"]:
                            if "recursive" in pattern["name"]:
                                recursion_found = True
                                break
                    if not recursion_found:
                        match = False
                
                # Check conditionals
                if has_conditionals and match:
                    if "control_flow" not in analysis or analysis["control_flow"].get("if_statements", 0) == 0:
                        match = False
                
                # Check error handling
                if handles_errors and match:
                    if "block_purposes" not in analysis:
                        match = False
                    else:
                        has_error_block = False
                        for purpose in analysis["block_purposes"]:
                            if "error" in purpose["purpose"].lower() or "exception" in purpose["purpose"].lower():
                                has_error_block = True
                                break
                        if not has_error_block:
                            match = False
                
                # Check pure function
                if is_pure and match:
                    if "io_relationship" not in analysis or \
                       analysis["io_relationship"]["output_determinism"] != "deterministic" or \
                       analysis["io_relationship"].get("side_effects", []):
                        match = False
                
                # Check input validation
                if has_validation and match:
                    if "io_relationship" not in analysis or \
                       not analysis["io_relationship"].get("input_validation", False):
                        match = False
                
                if match:
                    results.append({
                        "name": func_name,
                        "metadata": metadata,
                        "similarity": 1.0
                    })
            
            if results:
                st.success(f"Found {len(results)} matching functions")
                
                for i, result in enumerate(results):
                    func_name = result["name"]
                    metadata = result["metadata"]
                    
                    with st.expander(f"{i+1}. {func_name}{metadata['signature']}"):
                        st.markdown(f"**Description:** {metadata['docstring']}")
                        
                        # Tabs for different views
                        tab1, tab2 = st.tabs(["Algorithm Analysis", "Source Code"])
                        
                        with tab1:
                            render_algorithm_analysis(metadata)
                            
                        with tab2:
                            display_source_code(registry.functions[func_name])
            else:
                st.warning("No functions match the selected criteria")

elif page == "Algorithm Comparison":
    st.header("Algorithm Comparison")
    st.write("Compare implementation approaches and complexity across different functions")
    
    # Let the user select multiple functions to compare
    all_functions = registry.list_functions()
    selected_functions = st.multiselect("Select functions to compare", all_functions)
    
    if selected_functions:
        # Gather comparison data
        comparison_data = []
        
        for func_name in selected_functions:
            metadata = registry.get_metadata(func_name)
            
            # Basic function info
            func_data = {
                "name": func_name,
                "description": metadata["docstring"],
                "time_complexity": "Unknown",
                "space_complexity": "Unknown",
                "algorithm_type": "Unknown",
                "data_structures": [],
                "control_flow": {}
            }
            
            # Add algorithmic analysis if available
            if "algorithmic_analysis" in metadata:
                analysis = metadata["algorithmic_analysis"]
                
                # Add complexity
                if "complexity" in analysis:
                    func_data["time_complexity"] = analysis["complexity"]["time_complexity"]
                    func_data["space_complexity"] = analysis["complexity"]["space_complexity"]
                
                # Add algorithm type
                if "algorithm_patterns" in analysis and analysis["algorithm_patterns"]:
                    patterns = [p["name"] for p in analysis["algorithm_patterns"]]
                    func_data["algorithm_type"] = ", ".join(patterns)
                
                # Add data structures
                if "data_flows" in analysis and "data_structures_used" in analysis["data_flows"]:
                    func_data["data_structures"] = analysis["data_flows"]["data_structures_used"]
                
                # Add control flow
                if "control_flow" in analysis:
                    func_data["control_flow"] = analysis["control_flow"]
            
            comparison_data.append(func_data)
        
        # Create comparison view
        st.subheader("Side-by-Side Comparison")
        
        # Comparison table for basic metrics
        comparison_df = pd.DataFrame([{
            "Function": data["name"],
            "Time Complexity": data["time_complexity"],
            "Space Complexity": data["space_complexity"],
            "Algorithm Type": data["algorithm_type"],
            "Data Structures": ", ".join(data["data_structures"]) if data["data_structures"] else "None"
        } for data in comparison_data])
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visual comparison of control flow
        st.subheader("Control Flow Comparison")
        
        control_flow_data = []
        for data in comparison_data:
            flow_data = data["control_flow"]
            if flow_data:
                for metric, value in flow_data.items():
                    control_flow_data.append({
                        "Function": data["name"],
                        "Metric": metric.replace("_", " ").title(),
                        "Count": value
                    })
        
        if control_flow_data:
            control_flow_df = pd.DataFrame(control_flow_data)
            
            # Plot
            fig = px.bar(
                control_flow_df, 
                x="Function", 
                y="Count", 
                color="Metric", 
                barmode="group",
                title="Control Flow Structure Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No control flow data available for comparison")
        
        # Detailed comparison
        st.subheader("Detailed Analysis")
        
        tabs = st.tabs([data["name"] for data in comparison_data])
        
        for i, tab in enumerate(tabs):
            with tab:
                func_name = comparison_data[i]["name"]
                metadata = registry.get_metadata(func_name)
                
                st.markdown(f"### {func_name}{metadata['signature']}")
                st.markdown(f"**Description:** {metadata['docstring']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Source code
                    st.subheader("Source Code")
                    display_source_code(registry.functions[func_name])
                
                with col2:
                    # Algorithm analysis
                    st.subheader("Algorithmic Analysis")
                    render_algorithm_analysis(metadata)
    else:
        st.info("Select at least one function to begin comparison")
        
        # Suggest some comparison sets
        st.subheader("Try these comparisons:")
        
        comparison_sets = [
            ("Sorting Algorithms", ["merge_sort", "quick_sort"]),
            ("Search Algorithms", ["binary_search", "depth_first_search", "breadth_first_search"]),
            ("Fibonacci Implementations", ["fibonacci_recursive", "fibonacci_dynamic"]),
            ("String Operations", ["capitalize_words", "is_palindrome"])
        ]
        
        for title, funcs in comparison_sets:
            # Check if all functions in this set exist
            if all(func in all_functions for func in funcs):
                if st.button(f"Compare {title}"):
                    st.session_state["selected_functions"] = funcs
                    st.experimental_rerun()

elif page == "Function Execution":
    st.header("Function Execution")
    st.write("Test functions with different inputs and analyze their behavior")
    
    # Get the selected function from session state or let user select
    if "selected_func" in st.session_state:
        selected_func = st.session_state["selected_func"]
        del st.session_state["selected_func"]
    else:
        selected_func = st.selectbox("Select a function", registry.list_functions())
    
    if selected_func:
        metadata = registry.get_metadata(selected_func)
        st.subheader(f"{selected_func}{metadata['signature']}")
        st.markdown(f"**Description:** {metadata['docstring']}")
        
        # Show algorithmic properties
        if "algorithmic_analysis" in metadata and "complexity" in metadata["algorithmic_analysis"]:
            complexity = metadata["algorithmic_analysis"]["complexity"]
            st.markdown(f"**Time Complexity:** {complexity['time_complexity']} | **Space Complexity:** {complexity['space_complexity']}")
        
        # Create input fields for function parameters
        st.subheader("Function Parameters")
        
        params = {}
        for param_name in metadata['parameters']:
            # Try to infer parameter type from metadata
            param_type = "text"  # Default
            
            # Check param name for hints
            if any(hint in param_name.lower() for hint in ["number", "num", "count", "size", "index"]):
                param_type = "number"
            elif any(hint in param_name.lower() for hint in ["list", "array", "items", "elements"]):
                param_type = "list"
            elif any(hint in param_name.lower() for hint in ["bool", "flag", "enable", "is_"]):
                param_type = "bool"
            
            # Create appropriate input field
            if param_type == "number":
                params[param_name] = st.number_input(f"{param_name}", step=1)
            elif param_type == "list":
                list_input = st.text_input(f"{param_name} (comma-separated)", "1, 2, 3")
                params[param_name] = [item.strip() for item in list_input.split(",")]
            elif param_type == "bool":
                params[param_name] = st.checkbox(f"{param_name}")
            else:
                params[param_name] = st.text_input(f"{param_name}")
        
        # Execute button
        if st.button("Execute Function"):
            # Process parameters to correct types
            processed_params = {}
            for name, value in params.items():
                if isinstance(value, str):
                    # Try to convert to appropriate type
                    if value.lower() == 'true':
                        processed_params[name] = True
                    elif value.lower() == 'false':
                        processed_params[name] = False
                    else:
                        try:
                            # Try as number
                            if '.' in value:
                                processed_params[name] = float(value)
                            else:
                                processed_params[name] = int(value)
                        except ValueError:
                            # Keep as string
                            processed_params[name] = value
                elif isinstance(value, list):
                    # Process each item in the list
                    processed_list = []
                    for item in value:
                        if isinstance(item, str):
                            try:
                                # Convert to number if possible
                                if '.' in item:
                                    processed_list.append(float(item))
                                else:
                                    processed_list.append(int(item))
                            except ValueError:
                                processed_list.append(item)
                        else:
                            processed_list.append(item)
                    processed_params[name] = processed_list
                else:
                    # Keep as is
                    processed_params[name] = value
            
            # Execute the function
            with st.spinner("Executing..."):
                result = registry.execute(selected_func, **processed_params)
            
            # Display results
            if result['status'] == 'success':
                st.success("Function executed successfully!")
                
                # Show the result in an appropriate format
                st.subheader("Result")
                
                if isinstance(result['result'], (list, dict)):
                    st.json(result['result'])
                elif isinstance(result['result'], (int, float)):
                    st.metric("Output", result['result'])
                else:
                    st.write(result['result'])
                
                st.info(f"Execution time: {result['execution_time']:.6f} seconds")
                
                # Update execution history in session state
                if "execution_history" not in st.session_state:
                    st.session_state["execution_history"] = []
                
                st.session_state["execution_history"].append({
                    "function": selected_func,
                    "params": processed_params,
                    "result": result['result'],
                    "execution_time": result['execution_time']
                })
            else:
                st.error(f"Error: {result['error']}")
        
        # Show execution history
        if "execution_history" in st.session_state and st.session_state["execution_history"]:
            st.subheader("Execution History")
            
            history = [entry for entry in st.session_state["execution_history"] 
                      if entry["function"] == selected_func]
            
            if history:
                for i, entry in enumerate(reversed(history)):
                    with st.expander(f"Execution {len(history)-i}"):
                        st.write("**Parameters:**")
                        st.json(entry["params"])
                        st.write("**Result:**")
                        st.write(entry["result"])
                        st.write(f"**Execution Time:** {entry['execution_time']:.6f} seconds")
            else:
                st.info("No execution history for this function")
        
        # Show source code
        with st.expander("View Source Code"):
            display_source_code(registry.functions[selected_func])

elif page == "Pattern Explorer":
    st.header("Algorithm Pattern Explorer")
    st.write("Discover and understand common algorithmic patterns in your function library")
    
    # Get algorithm patterns
    all_patterns = {}
    
    for func_name in registry.list_functions():
        metadata = registry.get_metadata(func_name)
        
        if "algorithmic_analysis" in metadata and "algorithm_patterns" in metadata["algorithmic_analysis"]:
            for pattern in metadata["algorithmic_analysis"]["algorithm_patterns"]:
                pattern_name = pattern["name"]
                
                if pattern_name not in all_patterns:
                    all_patterns[pattern_name] = {
                        "name": pattern_name,
                        "category": pattern["category"],
                        "description": pattern["description"],
                        "functions": []
                    }
                
                all_patterns[pattern_name]["functions"].append(func_name)
    
    # Display patterns
    if all_patterns:
        st.success(f"Found {len(all_patterns)} algorithmic patterns in your function library")
        
        # Group patterns by category
        patterns_by_category = {}
        
        for pattern in all_patterns.values():
            category = pattern["category"]
            
            if category not in patterns_by_category:
                patterns_by_category[category] = []
            
            patterns_by_category[category].append(pattern)
        
        # Create tabs for each category
        if patterns_by_category:
            tabs = st.tabs([category.replace("_", " ").title() for category in patterns_by_category.keys()])
            
            for i, (category, patterns) in enumerate(patterns_by_category.items()):
                with tabs[i]:
                    for pattern in patterns:
                        with st.expander(f"{pattern['name'].replace('_', ' ').title()} ({len(pattern['functions'])} functions)"):
                            st.markdown(f"**Description:** {pattern['description']}")
                            
                            st.subheader("Functions Using This Pattern")
                            for func_name in pattern["functions"]:
                                metadata = registry.get_metadata(func_name)
                                st.markdown(f"- {func_name}{metadata['signature']}")
                            
                            # Button to compare these functions
                            if st.button("Compare These Functions", key=f"compare_{pattern['name']}"):
                                st.session_state["selected_functions"] = pattern["functions"]
                                st.session_state["page"] = "Algorithm Comparison"
                                st.experimental_rerun()
        else:
            st.warning("No algorithm patterns categorized")
    else:
        st.warning("No algorithm patterns detected in the function library")
        
        # Suggest adding example algorithms
        st.info("Add more complex algorithms to your library to see patterns")
        
        # Example algorithmic functions
        example_code = """
        def binary_search(arr, target):
            '''Efficiently find an element in a sorted array.'''
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
            
        def depth_first_search(graph, start):
            '''Traverse a graph using depth-first search.'''
            visited = set()
            result = []
            
            def dfs(node):
                if node in visited:
                    return
                visited.add(node)
                result.append(node)
                for neighbor in graph.get(node, []):
                    dfs(neighbor)
            
            dfs(start)
            return result
        """
        
        st.code(example_code, language="python")