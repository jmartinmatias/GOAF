#app.py
import streamlit as st
from src.core.registry import FunctionRegistry
from example_functions import *

# Initialize the registry
registry = FunctionRegistry()

# Register a variety of functions
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

# Streamlit App
st.title("Function Registry Dashboard")

# Sidebar menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select page", ["Functions", "Search", "Code-Based Search", "Execute"])

if page == "Functions":
    st.header("Registered Functions")
    
    for func_name in registry.list_functions():
        metadata = registry.get_metadata(func_name)
        stats = registry.get_stats(func_name)
        
        st.subheader(f"{func_name}{metadata['signature']}")
        st.write(metadata['docstring'])
        
        # Display code patterns if available
        if "code_patterns" in metadata:
            patterns = metadata["code_patterns"]
            if patterns:
                pattern_text = ", ".join([p.replace("_", " ").title() for p, v in patterns.items() if v])
                if pattern_text:
                    st.caption(f"Code characteristics: {pattern_text}")
                    
        st.text(f"Called {stats['calls']} times")
        st.markdown("---")
        
elif page == "Search":
    st.header("Keyword Search")
    st.write("Find functions based on name or description")
    
    query = st.text_input("Enter search term")
    
    if query:
        results = registry.search(query)
        if results:
            st.success(f"Found {len(results)} matching functions")
            for func_name in results:
                metadata = registry.get_metadata(func_name)
                st.subheader(f"{func_name}{metadata['signature']}")
                st.write(metadata['docstring'])
                st.markdown("---")
        else:
            st.warning("No functions found matching your query")
            
elif page == "Code-Based Search":
    st.header("Code-Based Semantic Function Search")
    st.write("Find functions based on their implementation, not just descriptions")
    
    query = st.text_input("What functionality are you looking for?")
    
    if query:
        results = registry.semantic_search(query)
        if results:
            st.success(f"Found {len(results)} semantically related functions")
            for result in results:
                func_name = result["name"]
                similarity = result["similarity"]
                metadata = result["metadata"]
                
                st.subheader(f"{func_name}{metadata['signature']}")
                st.write(metadata['docstring'])
                
                # Show code patterns if available
                if "code_patterns" in metadata:
                    patterns = metadata["code_patterns"]
                    if patterns:
                        st.write("**Code Patterns:**")
                        for pattern, value in patterns.items():
                            if value:
                                st.write(f"- {pattern.replace('_', ' ').title()}")
                
                # Visual indicators for match quality
                st.progress(similarity)
                
                # Indicate if match was boosted by code analysis
                if result.get("matched_on_code"):
                    st.info("✓ Matched based on code implementation patterns")
                    
                st.text(f"Similarity: {similarity:.2f}")
                st.markdown("---")
        else:
            st.warning("No functions found matching your query")
            
            # Suggest queries
            st.subheader("Try these example queries:")
            example_queries = [
                "calculate numeric statistics",
                "manipulate strings and text",
                "filter or process lists",
                "read or write files",
                "handle network operations"
            ]
            for query in example_queries:
                if st.button(query):
                    st.experimental_rerun()
                    
elif page == "Execute":
    st.header("Execute Function")
    
    func_names = registry.list_functions()
    selected_func = st.selectbox("Select a function", func_names)
    
    if selected_func:
        metadata = registry.get_metadata(selected_func)
        st.subheader(f"{selected_func}{metadata['signature']}")
        st.write(metadata['docstring'])
        
        # Show code patterns for the selected function
        if "code_patterns" in metadata:
            patterns = metadata["code_patterns"]
            if patterns:
                pattern_text = ", ".join([p.replace("_", " ").title() for p, v in patterns.items() if v])
                if pattern_text:
                    st.caption(f"Code characteristics: {pattern_text}")
        
        # Create input fields for function parameters
        params = {}
        for param_name in metadata['parameters']:
            params[param_name] = st.text_input(f"Enter {param_name}")
            
        if st.button("Execute"):
            # Convert string inputs to appropriate types (simple version)
            processed_params = {}
            for name, value in params.items():
                if not value:  # Skip empty values
                    continue
                try:
                    # Try to convert to number if possible
                    if value.lower() == 'true':
                        processed_params[name] = True
                    elif value.lower() == 'false':
                        processed_params[name] = False
                    elif '.' in value and value.replace('.', '').isdigit():
                        processed_params[name] = float(value)
                    elif value.isdigit():
                        processed_params[name] = int(value)
                    elif value.startswith('[') and value.endswith(']'):
                        # Handle simple list input
                        items = value[1:-1].split(',')
                        processed_params[name] = [item.strip() for item in items]
                    else:
                        # Keep as string if not a number
                        processed_params[name] = value
                except ValueError:
                    # Keep as string if conversion fails
                    processed_params[name] = value
                    
            # Execute the function
            result = registry.execute(selected_func, **processed_params)
            
            if result['status'] == 'success':
                st.success("Function executed successfully!")
                st.json({
                    "result": result['result'],
                    "execution_time": f"{result['execution_time']:.6f} seconds"
                })
            else:
                st.error(f"Error: {result['error']}")

# Display app info in the sidebar
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Function Registry Dashboard**
    
    A code-aware function registry system that understands implementation patterns.
    """
)