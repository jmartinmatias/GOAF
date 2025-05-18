#example_functions.py
"""
Example functions demonstrating various code patterns for the function registry.
Functions cover a range of operations, from simple calculations to network and file I/O.
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union


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