import weaviate
from weaviate.auth import AuthApiKey
from weaviate.connect import ConnectionParams
from sentence_transformers import SentenceTransformer
import logging
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


def get_embedding_model(self):
    return SentenceTransformer("pritamdeka/S-PubMedBERT-MS-MARCO")


def connect_to_weaviate(
    host="149.165.151.58",
    port=8080,
    grpc_port=50051,
    api_key=WEAVIATE_API_KEY,
):
    """
    Establish a connection to Weaviate with error handling.

    Args:
        host (str): Weaviate server host
        port (int): HTTP port
        grpc_port (int): gRPC port
        api_key (str): Authentication API key

    Returns:
        weaviate.WeaviateClient: Connected Weaviate client
    """
    try:

        # Create connection parameters
        conn_params = ConnectionParams.from_params(
            http_host=host,
            http_port=port,
            http_secure=False,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=False,
        )

        # Connect to Weaviate
        client = weaviate.WeaviateClient(
            connection_params=conn_params, auth_client_secret=AuthApiKey(api_key)
        )

        client.connect()

        if not client.is_ready():
            raise ConnectionError("Weaviate client is not ready")

        return client
    except Exception as e:
        logger.error(f"Weaviate connection failed: {e}")
        raise


def get_embedding_model(model_name="pritamdeka/S-PubMedBERT-MS-MARCO"):
    """
    Load embedding model for vector search.

    Args:
        model_name (str): Name of the SentenceTransformer model

    Returns:
        SentenceTransformer: Loaded embedding model
    """
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


def perform_search(
    client,
    collection,
    embedding_model,
    query,
    search_type="hybrid",
    limit=5,
    alpha=0.5,  # Added alpha parameter
):
    """
    Perform different types of search in Weaviate.

    Args:
        client (weaviate.WeaviateClient): Connected Weaviate client
        collection (str): Collection to search
        embedding_model (SentenceTransformer): Embedding model
        query (str): Search query
        search_type (str): Type of search ('bm25', 'vector', or 'hybrid')
        limit (int): Number of results to return
        alpha (float): Weight factor for hybrid search (0.0 = BM25 only, 1.0 = vector only)

    Returns:
        list: Search results
    """
    try:
        # Get the collection
        coll = client.collections.get(collection)

        # Generate query vector
        query_vector = embedding_model.encode(query).tolist()

        # Perform search based on type
        if search_type == "bm25":
            # BM25 keyword search
            results = coll.query.bm25(query=query, limit=limit)
        elif search_type == "vector":
            # Vector similarity search
            results = coll.query.near_vector(near_vector=query_vector, limit=limit)
        else:
            # Hybrid search (default)
            results = coll.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=alpha,  # Using the alpha parameter passed to the function
                limit=limit,
            )

        # Process results
        processed_results = []
        for obj in results.objects:
            result = {
                "pmid": obj.properties.get("pmid", "N/A"),
                "title": obj.properties.get("title", "N/A"),
                "journal": obj.properties.get("journal", "N/A"),
                "text_preview": obj.properties.get("text", "")[:200] + "...",
                "metadata": obj.metadata,
            }
            processed_results.append(result)

        return processed_results

    except Exception as e:
        logger.error(f"{search_type.upper()} search failed: {e}")
        return []


def close_weaviate_client(client):
    """
    Close the Weaviate client connection.

    Args:
        client (weaviate.WeaviateClient): Connected Weaviate client
    """
    try:
        client.close()
        logger.info("Weaviate client connection closed successfully.")
    except Exception as e:
        logger.error(f"Failed to close Weaviate client: {e}")
