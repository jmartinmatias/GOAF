from src.core.registry import FunctionRegistry

# Create a registry instance
registry = FunctionRegistry()

# Register some test functions
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

# Test function execution
print("\nTesting function execution:")
result1 = registry.execute("add", 5, 3)
print(f"add(5, 3) = {result1['result']} (execution time: {result1['execution_time']:.6f}s)")

result2 = registry.execute("multiply", 4, 7)
print(f"multiply(4, 7) = {result2['result']} (execution time: {result2['execution_time']:.6f}s)")

result3 = registry.execute("greet", "World")
print(f"greet('World') = {result3['result']} (execution time: {result3['execution_time']:.6f}s)")

# Test function search
print("\nTesting function search:")
search_result = registry.search("numb")
print(f"Search for 'numb' found: {search_result}")

# List all functions
print("\nAll registered functions:")
for func_name in registry.list_functions():
    metadata = registry.get_metadata(func_name)
    stats = registry.get_stats(func_name)
    print(f"- {func_name}{metadata['signature']}: {metadata['docstring']}")
    print(f"  Called {stats['calls']} times, avg time: {stats['avg_time']:.6f}s")