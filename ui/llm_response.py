from typing import List, Dict, Any
from dotenv import load_dotenv
import os

# Check
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import Client
from langchain.embeddings.base import Embeddings

from sentence_transformers import SentenceTransformer

from langchain.schema import SystemMessage, HumanMessage, AIMessage
import logging

# Configure logging - only essential logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

from rag.rag_pipeline import PubMedRAG

# Load environment variables
load_dotenv()

# Global RAG pipeline instance
_rag_instance = None


# Create an adapter class that provides the methods LangChain expects, this adapter is created to handle runtime errors.
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()


def get_rag_instance(supabase_client: Client) -> PubMedRAG:
    global _rag_instance

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    if _rag_instance is None:
        logger.debug("Initializing new RAG instance")

        ## TODO: Remove this is only for testing, Initialize the supabase client. only once and pass the retriver for langchain RAG.
        try:
            # Initialize the vectorstore and retriever
            vectorstore = SupabaseVectorStore(
                client=supabase_client,
                embedding=embedding_model,
                table_name="pubmed_documents",
                query_name="match_documents",
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        except Exception as e:
            logger.error(
                f"2.E Error loading vectorstore from Supabase: {str(e)}", exc_info=True
            )
            raise RuntimeError("Failed to initialize retriever") from e

        _rag_instance = PubMedRAG(retriever=retriever)
        _rag_instance.initialize()

    return _rag_instance


def format_chat_history(messages: List[Dict[str, str]]) -> List:
    formatted_messages = [
        SystemMessage(
            content=(
                "You are a helpful AI assistant that specializes in medical research. "
                "Provide informative and relevant responses based on PubMed research "
                "papers and the conversation context. Always cite your sources with "
                "PMID when possible."
            )
        )
    ]

    for message in messages:
        content = message.get("content", "")
        if message.get("role") == "user":
            formatted_messages.append(HumanMessage(content=content))
        elif message.get("role") == "assistant":
            formatted_messages.append(AIMessage(content=content))

    return formatted_messages


# RAG instance is sent as a parameter to this function, such that RAG is initialized only once.
def get_llm_response(
    user_input: str, rag_instance: PubMedRAG, chat_history: List[Dict[str, str]] = None
) -> str:
    try:
        logger.debug("Processing user input for LLM response")
        # Initialize chat history if needed
        if chat_history is None:
            chat_history = []

        response = rag_instance.query(user_input)

        # Check
        # formatted_messages = format_chat_history(chat_history)
        # formatted_messages.append(HumanMessage(content=user_input))
        # formatted_messages.append(AIMessage(content=response))
        return response

    except Exception as e:
        logger.error(f"Error in get_llm_response: {str(e)}", exc_info=True)
        error_msg = f"An error occurred: {str(e)}"
        return error_msg
