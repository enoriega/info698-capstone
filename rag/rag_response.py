from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from typing import List, Dict
from dotenv import load_dotenv
import os
import weaviate
from weaviate.client import WeaviateClient
from weaviate.auth import AuthApiKey

load_dotenv()

LLM_APIKEY = os.getenv("LLM_APIKEY")
WEAVIATE_URL = "http://149.165.151.58:8080"  # Full URL with protocol and port
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

def create_weaviate_client():
    """Create and return a Weaviate client."""
    try:
        # Initialize client with the correct connection setup
        client = weaviate.connect_to_custom(
            http_host="149.165.151.58",
            http_port=8080,
            http_secure=False,  # Set to True if using HTTPS
            grpc_host="149.165.151.58",
            grpc_port=50051,    # Default gRPC port
            grpc_secure=False,  # Set to True if using secure gRPC
            auth_credentials=AuthApiKey(WEAVIATE_API_KEY)
        )
        return client
    except Exception as e:
        print(f"Error creating Weaviate client: {str(e)}")
        raise

def create_chat_client():
    """Create and return a ChatOpenAI client."""
    return ChatOpenAI(
        model="Qwen2.5-Coder-32B-Instruct",
        openai_api_key=LLM_APIKEY,
        openai_api_base="https://llm-api.cyverse.ai/v1",
        streaming=True
    )

def perform_vector_search(client: WeaviateClient, query: str, class_name="test_db"):
    """
    Perform a vector search using Weaviate v4 client.
    Returns relevant documents with all schema properties.
    """
    try:
        response = (
            client.collections
            .get(class_name)
            .query
            .near_text(
                query,
                limit=3,
                return_properties=[
                    "pmid",
                    "pmc",
                    "doi",
                    "title",
                    "journal",
                    "issn",
                    "volume",
                    "issue",
                    "publication_date",
                    "authors",
                    "keywords",
                    "text",
                    "chunk_id"
                ]
            )
            .do()
        )
        return response.objects
    except Exception as e:
        print(f"Error in vector search: {str(e)}")
        return []

def format_chat_history(messages: List[Dict[str, str]]) -> List:
    """Format the chat history into LangChain message format."""
    formatted_messages = [
        SystemMessage(content=(
            "You are a helpful AI assistant with access to a medical research database. "
            "Use the retrieved context to provide informative and relevant responses. "
            "If the information isn't in the retrieved context, say so clearly. "
            "When citing information, include the PMID and DOI when available."
        ))
    ]

    for message in messages:
        content = message.get("content", "")
        if message.get("role") == "user":
            formatted_messages.append(HumanMessage(content=content))
        elif message.get("role") == "assistant":
            formatted_messages.append(AIMessage(content=content))

    return formatted_messages

def format_retrieved_context(docs):
    """Format retrieved documents into a readable context string with comprehensive metadata."""
    context = "\nRelevant sources:\n"
    for i, doc in enumerate(docs, 1):
        properties = doc.properties
        context += f"\n{i}. Title: {properties.get('title', 'N/A')}\n"

        # Add comprehensive metadata
        context += f"   PMID: {properties.get('pmid', 'N/A')}\n"
        if properties.get('doi'):
            context += f"   DOI: {properties.get('doi', 'N/A')}\n"
        if properties.get('pmc'):
            context += f"   PMC: {properties.get('pmc', 'N/A')}\n"

        # Journal information
        journal_info = []
        if properties.get('journal'):
            journal_info.append(properties['journal'])
        if properties.get('volume'):
            journal_info.append(f"Vol. {properties['volume']}")
        if properties.get('issue'):
            journal_info.append(f"Issue {properties['issue']}")
        if journal_info:
            context += f"   Journal: {', '.join(journal_info)}\n"

        # Publication date
        if properties.get('publication_date'):
            context += f"   Published: {properties.get('publication_date')}\n"

        # Authors
        if properties.get('authors'):
            authors = properties['authors']
            if isinstance(authors, list):
                context += f"   Authors: {', '.join(authors[:3])}"
                if len(authors) > 3:
                    context += f" and {len(authors)-3} more\n"
                else:
                    context += "\n"

        # Keywords
        if properties.get('keywords'):
            keywords = properties['keywords']
            if isinstance(keywords, list) and keywords:
                context += f"   Keywords: {', '.join(keywords[:5])}"
                if len(keywords) > 5:
                    context += f" and {len(keywords)-5} more\n"
                else:
                    context += "\n"

        # Text excerpt
        context += f"   Excerpt: {properties.get('text', '')[:300]}...\n"

        # Chunk identifier
        if properties.get('chunk_id'):
            context += f"   Chunk ID: {properties.get('chunk_id')}\n"

        context += "\n"  # Add spacing between documents

    return context

def get_llm_response(user_input: str, chat_history: List[Dict[str, str]] = None):
    """
    Get streaming response from the LLM model using chat history and Weaviate retrieval.
    """
    try:
        chat = create_chat_client()
        weaviate_client = create_weaviate_client()

        if chat_history is None:
            chat_history = []

        # Get relevant documents using Weaviate v4 search
        docs = perform_vector_search(weaviate_client, user_input)

        # Format the context from retrieved documents
        context = format_retrieved_context(docs)

        # Format messages for the final response
        messages = format_chat_history(chat_history)
        messages.append(HumanMessage(content=f"Context: {context}\n\nQuestion: {user_input}"))

        # Get streaming response from LLM
        for chunk in chat.stream(messages):
            if chunk.content is not None:
                yield chunk.content

    except Exception as e:
        yield f"An error occurred: {str(e)}"

def insert_document(client: WeaviateClient, doc_data: Dict, vector: List[float], class_name="test_db"):
    """
    Insert a document with its vector embedding using Weaviate v4 client.
    """
    try:
        collection = client.collections.get(class_name)

        # Configure the document properties based on the schema
        properties = {
            "pmid": doc_data.get("pmid", ""),
            "pmc": doc_data.get("pmc", ""),
            "doi": doc_data.get("doi", ""),
            "title": doc_data.get("title", ""),
            "journal": doc_data.get("journal", ""),
            "issn": doc_data.get("issn", ""),
            "volume": doc_data.get("volume", ""),
            "issue": doc_data.get("issue", ""),
            "publication_date": doc_data.get("publication_date", ""),
            "authors": doc_data.get("authors", []),
            "keywords": doc_data.get("keywords", []),
            "text": doc_data.get("text", ""),
            "chunk_id": doc_data.get("chunk_id", "")
        }

        # Insert the document with its vector
        result = collection.data.insert(
            properties=properties,
            vector=vector
        )
        return result
    except Exception as e:
        print(f"Error inserting document: {str(e)}")
        return None