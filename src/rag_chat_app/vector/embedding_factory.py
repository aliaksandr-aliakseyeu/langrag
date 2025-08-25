import os
from typing import Optional
from langchain.embeddings.base import Embeddings


def create_huggingface_embeddings(model_name: str = "all-MiniLM-L6-v2") -> Embeddings:
    """
    Create HuggingFace embeddings.

    Args:
        model_name: Name of the HuggingFace model

    Returns:
        HuggingFace embeddings instance

    Raises:
        ImportError: If langchain_huggingface is not installed
        ValueError: If the model fails to load
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    try:
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        raise ValueError(
            f"Failed to create HuggingFace embeddings with model '{model_name}'. "
            f"Please check if the model exists and is accessible. Error: {e}"
        ) from e


def create_openai_embeddings(
    model: str = "text-embedding-3-small", api_key: Optional[str] = None
) -> Embeddings:
    """
    Create OpenAI embeddings.

    Args:
        model: OpenAI embedding model name
        api_key: OpenAI API key (uses environment variable if not provided)

    Returns:
        OpenAI embeddings instance

    Raises:
        ValueError: If no API key is provided and OPENAI_API_KEY env var is not set
        ImportError: If langchain_openai is not installed
    """
    from langchain_openai import OpenAIEmbeddings

    # Get API key from parameter or environment variable
    final_api_key = api_key or os.getenv("OPENAI_API_KEY")

    if not final_api_key:
        raise ValueError(
            "OpenAI API key is required. Please provide it via the 'api_key' parameter "
            "or set the 'OPENAI_API_KEY' environment variable."
        )

    try:
        return OpenAIEmbeddings(model=model, api_key=final_api_key)
    except Exception as e:
        raise ValueError(
            f"Failed to create OpenAI embeddings with model '{model}'. "
            f"Please check your API key and model name. Error: {e}"
        ) from e
