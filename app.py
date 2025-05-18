import streamlit as st
from src.core.registry import FunctionRegistry

# Initialize the registry
registry = FunctionRegistry()

# Register example functions
@registry.register
def add(a, b):
    """Add two numbers together."""
    return a + b

@registry.register
def multiply(a, b):
    """Multiply two numbers together."""
    return a * b

@registry.register
def greet(name):
    """Greet a person by name."""
    return f"Hello, {name}!"

# Streamlit App
st.title("Function Registry Dashboard")

# Sidebar menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select page", ["Functions", "Search", "Execute"])

if page == "Functions":
    st.header("Registered Functions")
    
    for func_name in registry.list_functions():
        metadata = registry.get_metadata(func_name)
        stats = registry.get_stats(func_name)
        
        st.subheader(f"{func_name}{metadata['signature']}")
        st.write(metadata['docstring'])
        st.text(f"Called {stats['calls']} times")
        st.markdown("---")
        
elif page == "Search":
    st.header("Search Functions")
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
            
elif page == "Execute":
    st.header("Execute Function")
    
    func_names = registry.list_functions()
    selected_func = st.selectbox("Select a function", func_names)
    
    if selected_func:
        metadata = registry.get_metadata(selected_func)
        st.subheader(f"{selected_func}{metadata['signature']}")
        st.write(metadata['docstring'])
        
        # Create input fields for function parameters
        params = {}
        for param_name in metadata['parameters']:
            params[param_name] = st.text_input(f"Enter {param_name}")
            
        if st.button("Execute"):
            # Convert string inputs to appropriate types (simple version)
            processed_params = {}
            for name, value in params.items():
                try:
                    # Try to convert to number if possible
                    processed_params[name] = float(value) if '.' in value else int(value)
                except ValueError:
                    # Keep as string if not a number
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
    
    A simple demo of the function registry system.
    """
)