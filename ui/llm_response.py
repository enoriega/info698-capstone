from typing import List, Dict, Any
from dotenv import load_dotenv
import os
from supabase import Client
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import logging
from langchain.tools.retriever import create_retriever_tool
from rag.rag_pipeline import PubMedRAG

from logger_setup import logger

# Load environment variables
load_dotenv()

# Global RAG pipeline instance
_rag_instance = None
_current_model = None

def get_rag_instance(db_client, model_choice="gpt-4o") -> PubMedRAG:
    global _rag_instance, _current_model

    # Initialize a new instance if either:
    # 1. We don't have an instance yet
    # 2. The model choice has changed
    if _rag_instance is None or model_choice != _current_model:
        logger.debug(f"Initializing new RAG instance with model: {model_choice}")
        
        # Create new instance with the specified model
        _rag_instance = PubMedRAG(db_client, model_choice)
        _rag_instance.initialize()
        
        # Update the current model tracker
        _current_model = model_choice
        
        logger.debug(f"RAG instance initialized with {model_choice}")
    else:
        logger.debug(f"Using existing RAG instance with model: {_current_model}")
    
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
):
    try:
        logger.debug("Processing user input for LLM response")
        # Initialize chat history if needed
        if chat_history is None:
            chat_history = []

        for chunk in rag_instance.query_stream(user_input):
            yield chunk

        # Check
        # formatted_messages = format_chat_history(chat_history)
        # formatted_messages.append(HumanMessage(content=user_input))
        # formatted_messages.append(AIMessage(content=response))

    except Exception as e:
        logger.error(f"Error in get_llm_response: {str(e)}", exc_info=True)
        yield f"Error generating response: {str(e)}"
