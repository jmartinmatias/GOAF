# setup.py
from setuptools import setup, find_packages

setup(
    name="algo_explorer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.20.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "plotly>=5.0.0",
        "sentence-transformers>=2.2.0",
        "python-dotenv>=0.19.0",
        "google-generativeai>=0.1.0",
    ],
    python_requires=">=3.8",
)