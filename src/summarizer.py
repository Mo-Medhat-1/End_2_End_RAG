"""
Summarizer — page-level text summarization utilities.

Provides convenient access to page description functions
used during the ingestion pipeline.
"""
from src.metadata import describe_page, add_page_descriptions

__all__ = [
    "describe_page",
    "add_page_descriptions",
]
