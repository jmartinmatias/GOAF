"""
Example functions demonstrating various algorithmic patterns for the function registry.
Includes basic utility functions as well as classic algorithms with different complexity classes.
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from collections import defaultdict, deque


def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Add two numbers together."""
    return a + b


def subtract(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Subtract the second number from the first."""
    return a - b


def multiply(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Multiply two numbers together."""
    return a * b


def divide(a: Union[int, float], b: Union[int, float]) -> float:
    """Divide the first number by the second."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"


def count_words(text: str) -> int:
    """Count the number of words in a text."""
    if not text:
        return 0
    return len(text.split())


def is_palindrome(text: str) -> bool:
    """Check if a text is a palindrome (reads the same forward and backward)."""
    clean_text = ''.join(c.lower() for c in text if c.isalnum())
    return clean_text == clean_text[::-1]


def capitalize_words(text: str) -> str:
    """Capitalize the first letter of each word in a text."""
    return ' '.join(word.capitalize() for word in text.split())


def fetch_webpage(url: str) -> str:
    """Fetch the content of a webpage (placeholder implementation)."""
    # In a real implementation, this would use requests or another HTTP library
    try:
        # Simulated network operation
        if not url.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return f"Content from {url}"
    except Exception as e:
        raise ConnectionError(f"Failed to fetch {url}: {str(e)}")


def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email (placeholder implementation)."""
    # In a real implementation, this would use SMTP or an email API
    try:
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', to):
            raise ValueError("Invalid email address")
        # Simulated email sending
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to send email: {str(e)}")


def calculate_average(numbers: List[Union[int, float]]) -> float:
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)


def filter_even_numbers(numbers: List[int]) -> List[int]:
    """Filter a list to only include even numbers."""
    return [num for num in numbers if num % 2 == 0]


def sort_numbers(numbers: List[Union[int, float]], reverse: bool = False) -> List[Union[int, float]]:
    """Sort a list of numbers in ascending or descending order."""
    return sorted(numbers, reverse=reverse)


def find_max(numbers: List[Union[int, float]]) -> Union[int, float]:
    """Find the maximum value in a list of numbers."""
    if not numbers:
        raise ValueError("Cannot find maximum of an empty list")
    return max(numbers)


def find_min(numbers: List[Union[int, float]]) -> Union[int, float]:
    """Find the minimum value in a list of numbers."""
    if not numbers:
        raise ValueError("Cannot find minimum of an empty list")
    return min(numbers)


def read_file_lines(file_path: str) -> List[str]:
    """Read a text file and return a list of lines."""
    try:
        with open(file_path, 'r') as file:
            return file.readlines()
    except Exception as e:
        raise IOError(f"Error reading file {file_path}: {str(e)}")


def write_to_file(file_path: str, content: str) -> bool:
    """Write content to a text file."""
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return True
    except Exception as e:
        raise IOError(f"Error writing to file {file_path}: {str(e)}")


def parse_json(json_string: str) -> Dict[str, Any]:
    """Parse a JSON string into a dictionary."""
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")


def format_date(date_str: str, input_format: str = "%Y-%m-%d", output_format: str = "%B %d, %Y") -> str:
    """Format a date string from one format to another."""
    try:
        date_obj = datetime.strptime(date_str, input_format)
        return date_obj.strftime(output_format)
    except ValueError as e:
        raise ValueError(f"Invalid date format: {str(e)}")


def check_url_status(url: str) -> Dict[str, Any]:
    """Check if a URL is accessible (placeholder implementation)."""
    # In a real implementation, this would use requests and check HTTP status
    try:
        if not url.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        
        # Simulated status check
        if 'example.com' in url:
            status = 200
        elif 'notfound' in url:
            status = 404
        else:
            status = 200
            
        return {
            "url": url,
            "status": status,
            "accessible": status == 200
        }
    except Exception as e:
        return {
            "url": url,
            "error": str(e),
            "accessible": False
        }


# Classic Algorithm Implementations

def binary_search(arr: List[Union[int, float]], target: Union[int, float]) -> int:
    """
    Binary search algorithm that finds the position of a target value within a sorted array.
    
    Time complexity: O(log n)
    Space complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        # Check if target is present at mid
        if arr[mid] == target:
            return mid
        
        # If target is greater, ignore left half
        elif arr[mid] < target:
            left = mid + 1
        
        # If target is smaller, ignore right half
        else:
            right = mid - 1
    
    # Element is not present in array
    return -1


def merge_sort(arr: List[Union[int, float]]) -> List[Union[int, float]]:
    """
    Merge sort algorithm - efficiently sorts an array by dividing and conquering.
    
    Time complexity: O(n log n)
    Space complexity: O(n)
    """
    # Base case: lists with 0 or 1 element are already sorted
    if len(arr) <= 1:
        return arr
    
    # Divide the array into two halves
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    
    # Recursively sort both halves
    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)
    
    # Merge the sorted halves
    return _merge(left_half, right_half)


def _merge(left: List[Union[int, float]], right: List[Union[int, float]]) -> List[Union[int, float]]:
    """Helper function for merge_sort: merges two sorted arrays."""
    result = []
    i = j = 0
    
    # Merge elements from both arrays in order
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add any remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result


def quick_sort(arr: List[Union[int, float]]) -> List[Union[int, float]]:
    """
    Quick sort algorithm - efficiently sorts an array using a divide and conquer strategy.
    
    Time complexity: O(n log n) average case, O(n²) worst case
    Space complexity: O(log n) due to the recursive call stack
    """
    # Make a copy to avoid modifying the original
    arr = arr.copy()
    
    # Inner recursive function
    def _quick_sort(arr: List[Union[int, float]], low: int, high: int) -> None:
        if low < high:
            # Partition the array and get the pivot position
            pivot_idx = _partition(arr, low, high)
            
            # Recursively sort the subarrays
            _quick_sort(arr, low, pivot_idx - 1)  # Before pivot
            _quick_sort(arr, pivot_idx + 1, high)  # After pivot
    
    # Start the recursion
    _quick_sort(arr, 0, len(arr) - 1)
    return arr


def _partition(arr: List[Union[int, float]], low: int, high: int) -> int:
    """Helper function for quick_sort: partitions the array around a pivot."""
    # Choose the rightmost element as the pivot
    pivot = arr[high]
    
    # Index of smaller element
    i = low - 1
    
    for j in range(low, high):
        # If current element is smaller than or equal to pivot
        if arr[j] <= pivot:
            # Increment index of smaller element
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    # Swap the pivot element with the element at i+1
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    
    return i + 1


def depth_first_search(graph: Dict[Any, List[Any]], start: Any) -> List[Any]:
    """
    Depth-first search algorithm for graph traversal.
    
    Time complexity: O(V + E) where V is number of vertices and E is number of edges
    Space complexity: O(V)
    """
    visited = set()  # Set to keep track of visited nodes
    result = []  # List to store the traversal order
    
    def dfs_recursive(node):
        # Mark the current node as visited
        visited.add(node)
        result.append(node)
        
        # Recur for all adjacent vertices
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs_recursive(neighbor)
    
    # Start DFS from the given node
    dfs_recursive(start)
    return result


def breadth_first_search(graph: Dict[Any, List[Any]], start: Any) -> List[Any]:
    """
    Breadth-first search algorithm for graph traversal.
    
    Time complexity: O(V + E) where V is number of vertices and E is number of edges
    Space complexity: O(V)
    """
    # Mark the source node as visited and enqueue it
    visited = {start}
    queue = deque([start])
    result = []
    
    while queue:
        # Dequeue a vertex from queue
        node = queue.popleft()
        result.append(node)
        
        # Get all adjacent vertices
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                # Mark it visited and enqueue it
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result


def fibonacci_recursive(n: int) -> int:
    """
    Calculate the nth Fibonacci number using recursion.
    
    Time complexity: O(2^n) - exponential
    Space complexity: O(n) due to the recursive call stack
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_dynamic(n: int) -> int:
    """
    Calculate the nth Fibonacci number using dynamic programming.
    
    Time complexity: O(n)
    Space complexity: O(n)
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    # Initialize array to store Fibonacci numbers
    fib = [0] * (n + 1)
    fib[1] = 1
    
    # Fill the array
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]
    
    return fib[n]


def knapsack_problem(values: List[int], weights: List[int], capacity: int) -> int:
    """
    Solve the 0/1 knapsack problem using dynamic programming.
    
    Time complexity: O(n * capacity) where n is the number of items
    Space complexity: O(n * capacity)
    """
    n = len(values)
    
    # Create a table for dynamic programming
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Fill the dp table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            # If current item weight is less than or equal to capacity
            if weights[i - 1] <= w:
                # Maximum of including or excluding the current item
                dp[i][w] = max(
                    values[i - 1] + dp[i - 1][w - weights[i - 1]],  # Include
                    dp[i - 1][w]  # Exclude
                )
            else:
                # Can't include current item
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][capacity]


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance (edit distance) between two strings.
    
    Time complexity: O(m * n) where m and n are the lengths of the strings
    Space complexity: O(m * n)
    """
    # Create a table for dynamic programming
    m, n = len(s1), len(s2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Initialize the first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],     # Deletion
                    dp[i][j - 1],     # Insertion
                    dp[i - 1][j - 1]  # Substitution
                )
    
    return dp[m][n]


def is_prime(n: int) -> bool:
    """
    Check if a number is prime using trial division.
    
    Time complexity: O(sqrt(n))
    Space complexity: O(1)
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    
    # Check if n is divisible by 2 or 3
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Check divisibility by numbers of form 6k ± 1
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True