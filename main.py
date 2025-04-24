import streamlit as st
import json
import os
import logging
from data_loader.data_loader import (
    load_pubmed_data,
)
import atexit
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from ui.llm_response import get_llm_response, get_rag_instance
from weaviate_db.client import connect_to_weaviate, close_weaviate_client
from logger_setup import logger

# Define the path for storing chat history
CHAT_HISTORY_FILE = "chat_history.json"


def load_chat_history():
    """Load chat history from file or initialize if not exists."""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, "r") as f:
                return json.load(f)
        else:
            return []
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")
        return []


def save_chat_history(messages):
    """Save chat history to file."""
    try:
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump(messages, f)
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")


def display_message(message):
    """Display a single message."""
    with st.chat_message(message["role"]):
        st.write(message["content"])


def initialize_session():
    """Initialize session state variables."""
    # Check if chat history exists in session state, if not, load it
    #
    if "chat_history" not in st.session_state:
        logger.debug("Loading chat history")
        st.session_state.chat_history = load_chat_history()
    if "rag_initialized" not in st.session_state:
        logger.debug("RAG system not initialized")
        st.session_state.rag_initialized = False
    if "rag_instance" not in st.session_state:
        logger.debug("Setting RAG instance to None initially")
        st.session_state.rag_instance = None
    if "db_client" not in st.session_state:
        db_client = connect_to_weaviate()
        st.session_state.db_client = db_client
    if "llm_model" not in st.session_state:
        logger.debug("Setting default LLM model")
        st.session_state.llm_model = "gpt-4o"

def main():
    # TODO : Add Streaming of text
    # TODO : Add thinking chain of RAG
    # TODO : Add LangSmith for Observability
    logger.debug("Starting main application")
    st.set_page_config(
        page_title="PubMed Chatbot", layout="wide", initial_sidebar_state="expanded"
    )
    st.title("PubMed Chatbot")

    # Initialize session
    # Entry point for the app
    # Load chat history and initialize session state of rag_initialized
    ## TODO: Enhancement : Rename this function to load_session_state_chat_history()
    initialize_session()

    # Initialize RAG instance once
    # Second hero function here, which initializes the RAG instance.
    db_instance = st.session_state.db_client
    if st.session_state.rag_instance is None:
        logger.debug(f"Initializing RAG with model: {st.session_state.llm_model}")
        with st.spinner(f"Initializing RAG system with {st.session_state.llm_model}... (this might take a minute)"):
            try:
                st.session_state.rag_instance = get_rag_instance(db_instance, model_choice=st.session_state.llm_model)
                if st.session_state.rag_instance:
                    st.session_state.rag_initialized = True
                    logger.debug("RAG system initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing RAG system: {str(e)}")
                logger.error(f"RAG initialization error: {str(e)}")

    # Add sidebar controls
    # Any streamlit command inside this block will be rendered in the sidebar
    with st.sidebar:
        st.header("Controls")

        # Add model selector
        model_choice = st.selectbox(
            "Select LLM Model",
            ["gpt-4o", "Llama-3.2-11B-Vision-Instruct"],
            index=0 if st.session_state.llm_model == "gpt-4o" else 1,
            key="model_selector"
        )

        # Update session state if model changed
        if model_choice != st.session_state.llm_model:
            st.session_state.llm_model = model_choice
            st.session_state.rag_instance = None  # Force RAG reinitialization
            st.session_state.rag_initialized = False
            st.rerun()

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            save_chat_history([])
            st.rerun()

        # Add a divider for better UI
        st.divider()

        # Display text in header formatting.
        st.header("RAG Settings")

        if st.session_state.rag_initialized:
            st.success(f"RAG system ready with {st.session_state.llm_model}!")
        else:
            st.warning("RAG system not initialized.")

    # Create a container for chat history
    chat_container = st.container()

    for message in st.session_state.chat_history:
        display_message(message)
        
    user_input = st.chat_input("What would you like to know?")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        display_message({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            # Get assistant response
            # get_llm_response is the hero function here, Which gets the response from the RAG pipeline.
            # Takes the RAG instance
            response_content = ""
            response_placeholder = st.empty()
            for chunk in get_llm_response(
                user_input,
                st.session_state.rag_instance,
                st.session_state.chat_history[:-1],
            ):

                if chunk:
                    response_content += str(chunk)
                    response_placeholder.markdown(response_content)

        # Add assistant response to chat history
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response_content}
        )

        # Save updated chat history to file
        save_chat_history(st.session_state.chat_history)

    # Display all messages in the chat container

    atexit.register(close_weaviate_client, st.session_state.db_client)


if __name__ == "__main__":
    main()

# SAMPLE QUESTIONS FOR TESTING:
# # What was the main objective of the study comparing lorazepam and pentobarbital? - giving Answer
# # Which drug provided greater sedation and antianxiety effects?
# # What were the dosages of lorazepam and pentobarbital used in the study?

# "What is required for the induction of tyrosine aminotransferase by Bt2cAMP in HTC hepatoma cells?" - Max Recursion reched
# "How does dexamethasone influence the effect of Bt2cAMP on tyrosine aminotransferase synthesis?"
# "What evidence suggests that dexamethasone acts beyond the activation of protein kinase by cAMP?"
