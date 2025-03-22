import os
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import Document

from data_loader.data_loader import load_pubmed_data
from .llm_integration import create_llama_chat_model, create_rag_chain

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class PubMedRAG:

    def __init__(self, retriever):

        # Initialize components as None
        self.retriever = retriever
        self.llm = None
        self.rag_chain = None

    def initialize(self):
        logger.debug("Initializing RAG pipeline")

        # Initialize LLM
        if self.llm is None:
            self._initialize_llm()

        # If check for self.rag_chain
        if self.rag_chain is None:
            self.rag_chain = create_rag_chain(self.retriever, self.llm)
            logger.debug("RAG pipeline initialization complete")

    def _initialize_llm(self):
        self.llm = create_llama_chat_model()

    def query(self, question: str) -> str:

        logger.debug(f"Processing RAG query: {question[:50]}...")

        if not self.rag_chain:
            raise ValueError("RAG chain not initialized. Call initialize() first.")

        try:
            input_dict = {"question": question}
            response = self.rag_chain.invoke(input_dict)
            return response
        except Exception as e:
            logger.error(f"3.E2 Error in RAG query: {str(e)}", exc_info=True)
            return f"Error generating response: {str(e)}"
