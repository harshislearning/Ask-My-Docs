"""Streamlit front end and its HTTP client.

The UI is a client of the API, not of the pipeline. One code path into
retrieval and generation, and using the UI exercises the API contract.
"""

from .api_client import ApiError, AskMyDocsClient

__all__ = ["ApiError", "AskMyDocsClient"]
