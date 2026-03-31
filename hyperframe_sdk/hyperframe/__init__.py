"""
HyperFrame -- a high-performance Rust-backed DataFrame for Python.

Built progressively throughout the "High-Performance Data Engineering" course.
"""
from .frame import DataFrame
from .io import read_csv

__version__ = "0.1.0"
__all__ = ["DataFrame", "read_csv"]
