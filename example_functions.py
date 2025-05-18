
"""Example functions to demonstrate the function registry and semantic search."""

def add(a, b):
    """Add two numbers together."""
    return a + b

def subtract(a, b):
    """Subtract the second number from the first."""
    return a - b

def multiply(a, b):
    """Multiply two numbers together."""
    return a * b

def divide(a, b):
    """Divide the first number by the second."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def greet(name):
    """Greet a person by name."""
    return f"Hello, {name}!"

def count_words(text):
    """Count the number of words in a text."""
    return len(text.split())

def is_palindrome(text):
    """Check if a text is a palindrome (reads the same forward and backward)."""
    clean_text = ''.join(c.lower() for c in text if c.isalnum())
    return clean_text == clean_text[::-1]

def capitalize_words(text):
    """Capitalize the first letter of each word in a text."""
    return ' '.join(word.capitalize() for word in text.split())

def fetch_webpage(url):
    """Fetch the content of a webpage (placeholder implementation)."""
    return f"Content from {url}"

def send_email(to, subject, body):
    """Send an email (placeholder implementation)."""
    return f"Email sent to {to} with subject: {subject}"

def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)