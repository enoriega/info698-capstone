import os
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.tools.retriever import create_retriever_tool
from data_loader.data_loader import load_pubmed_data
from .llm_integration import create_rag_chain
from langchain_community.retrievers import WikipediaRetriever
from langgraph.prebuilt import create_react_agent
from langchain.embeddings.base import Embeddings
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import SentenceTransformer

from typing import Optional
from langchain_community.vectorstores import Weaviate
from langchain_weaviate import WeaviateVectorStore

RECURSION_LIMIT = 100

from logger_setup import logger

# Global instance
vectorstore_as_retriever = None
db_client_global = None


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
class MyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> list[Document]:
        coll = db_client_global.collections.get("test_db")
        embedding_model = get_embedding_model()
        query_vector = embedding_model.encode(query).tolist()
        results = coll.query.near_vector(near_vector=query_vector, limit=3)
        documents = []
        for obj in results.objects:
            text_content = f"Title: {obj.properties.get('title', '')}\n"
            text_content += f"Text: {obj.properties.get('text', '')}\n"
            text_content += "\nMetadata below--\n"
            text_content += f"PMID: {obj.properties.get('pmid', '')}\n"
            text_content += f"Journal: {obj.properties.get('journal', '')}\n"
            text_content += f"Source: PubMed"

            metadata = {
                "pmid": obj.properties.get("pmid", ""),
                "journal": obj.properties.get("journal", ""),
                "source": "PubMed",
            }
            doc = Document(page_content=text_content, metadata=metadata)
            documents.append(doc)
        logger.info(f"Retrieved {len(documents)} documents")

        return documents


class PubMedRAG:

    def __init__(self, db_client, model_choice="gpt-4o"):
        global db_client_global
        # Initialize components as None
        self.llm = None
        self.rag_chain = None
        self.db_client = db_client
        db_client_global = db_client
        logger.debug("Model Choice RAG_PIPELINE: %s", model_choice)
        self.model_choice = model_choice

    def initialize(self):
        logger.debug("Initializing RAG pipeline with model choice: %s", self.model_choice)

        embedding_model = SentenceTransformer("pritamdeka/S-PubMedBERT-MS-MARCO")
        vectorstore = WeaviateVectorStore(
            client=self.db_client,
            index_name="test_db",
            text_key="text",
            embedding=embedding_model,
        )
        vectorstore_as_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        retriever = MyRetriever()

        retrieval_tool = create_retriever_tool(
            retriever=retriever,
            name="pubmed_retriever",
            description="A tool to retrieve documents from my knowledge base created from PubMedCentral Database.",
        )
        logger.debug("Initializing RAG pipeline")
        system_message =  """
        You are an intelligent assistant specialized in analyzing biomedical literature, particularly PubMed papers. 
        Provide comprehensive, detailed responses backed by scientific evidence.

        - **`pubmed_retriever`**: Retrieves documents from PubMed articles database. Each document contains:
        - Title and full text
        - PMID (PubMed ID)
        - Journal name
        - Source information
        - **`wikipedia_retriever`**: Provides background medical knowledge when needed

        ### Document Citation and Detail Rules:
        1. **Comprehensive Paper Analysis**:
        - Extract and present ALL relevant details from each paper
        - Include methodology details when relevant
        - Describe sample sizes and study designs
        - Report specific numbers, percentages, and statistics
        - Explain mechanisms and pathways when available

        2. **Citation Format**:
        - Group ALL findings from the same paper together
        - Format: `[PMID: xxxxx](https://pubmed.ncbi.nlm.nih.gov/xxxxx/)`
        - Include journal name on first citation
        - Never split findings from the same paper across different sections

        ### Response Organization:
        1. **Detailed Structure**:
        - Begin with a thorough background/context
        - Present findings paper by paper, with full details from each source
        - Include subsections for different aspects (methods, results, implications)
        - Provide detailed analysis and interpretation
        - End with comprehensive conclusions

        2. **Paper Grouping Format**:
        "According to [Authors] in [Journal] [PMID link], the study provided several key findings:
        - Detailed finding 1 with specific data
        - Detailed finding 2 with methodology
        - Detailed finding 3 with statistics
        - Study limitations and implications"

        ### Information Integration:
        - Synthesize findings across papers
        - Explain contradictions or conflicts in the literature
        - Provide context for technical terms
        - Connect findings to clinical applications

        Example of Detailed Citation:
        "A comprehensive study by Smith et al. in the Journal of Medicine [PMID: 38382828](https://pubmed.ncbi.nlm.nih.gov/38382828/) investigated 500 patients with advanced cancer and found:
        - 75% response rate to the new treatment (p<0.001)
        - Median survival increased by 8.5 months (95% CI: 6.8-10.2)
        - Reduced side effects in 60% of cases
        - Used a double-blind, randomized controlled trial design
        The authors also identified key molecular pathways involving..."

        Remember: 
        - Always provide maximum detail while keeping information from the same source together
        - Include specific data, numbers, and statistics whenever available
        - Explain mechanisms and implications thoroughly
        - Maintain scientific accuracy and proper citation structure
        """


        wiki_retriever = create_retriever_tool(
            retriever=WikipediaRetriever(),
            name="wikipedia_retriever",
            description="Wikipedia retriever to search for medical terms and helping for information which is not in the knowledge base.",
        )
        # Initialize LLM
        if self.llm is None:
            self._initialize_llm()

        # If check for self.rag_chain
        if self.rag_chain is None:
            self.rag_chain = create_react_agent(
                model=self.llm,
                tools=[retrieval_tool, wiki_retriever],
                prompt=system_message,
            )
            logger.debug("RAG pipeline initialization complete")

    def _initialize_llm(self):
        from .llm_integration import create_llm_model
        self.llm = create_llm_model(self.model_choice)

    def query(self, question: str) -> str:

        logger.debug(f"Processing RAG query: {question[:50]}...")

        if not self.rag_chain:
            raise ValueError("RAG chain not initialized. Call initialize() first.")

        try:
            messages = self.rag_chain.invoke({"messages": [("human", question)]})
            return messages["messages"][-1].content
        except Exception as e:
            logger.error(f"3.E2 Error in RAG query: {str(e)}", exc_info=True)
            return f"Error generating response: {str(e)}"

    def process_chunks(self, chunk):
        if "agent" not in chunk:
            return None

        for message in chunk["agent"]["messages"]:
            # Handle tool calls (searching actions)
            if "tool_calls" in message.additional_kwargs:
                for tool_call in message.additional_kwargs["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    tool_arguments = eval(tool_call["function"]["arguments"])
                    return f"🔍 Searching {tool_name}:: {tool_arguments['query']}\n\n"
            
            # Handle actual content
            elif message.content:
                content = message.content
                # If it's the start of the response, add the header
                if not hasattr(self, '_response_started'):
                    self._response_started = True
                    return f"\n### Research Findings\n\n{content}"
                return content

        return None

    def query_stream(self, question: str):
        """
        Stream responses with immediate output
        """
        logger.debug(f"Streaming RAG query: {question[:50]}...")

        if not self.rag_chain:
            raise ValueError("RAG chain not initialized. Call initialize() first.")

        try:
            # Reset the response started flag
            self._response_started = False
            
            # Stream chunks directly
            for chunk in self.rag_chain.stream(
                {"messages": [("human", question)]},
                {"recursion_limit": RECURSION_LIMIT},
            ):
                processed_chunk = self.process_chunks(chunk)
                if processed_chunk:
                    yield processed_chunk

        except Exception as e:
            logger.error(f"Error in RAG query stream: {str(e)}", exc_info=True)
            yield f"Error generating response: {str(e)}"
        finally:
            # Clean up
            if hasattr(self, '_response_started'):
                delattr(self, '_response_started')
