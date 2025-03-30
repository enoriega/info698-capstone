import os
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.tools.retriever import create_retriever_tool
from data_loader.data_loader import load_pubmed_data
from .llm_integration import create_llama_chat_model, create_rag_chain
from langchain_community.retrievers import WikipediaRetriever
from langgraph.prebuilt import create_react_agent
from langchain.embeddings.base import Embeddings
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import SupabaseVectorStore
from typing import Optional

RECURSION_LIMIT = 100

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

logger.warning("Warning Logger  from Rag Pipeline")
logger.error("Error Logger from Rag Pipeline")
logger.debug("DEBUG Logger from Rag Pipeline")
logger.info("INFO Logger from Rag Pipeline")
# Global instance
supabase_vs_retriever = None

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


class MyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> list[Document]:
        global supabase_vs_retriever
        return supabase_vs_retriever.get_relevant_documents(query)

class PubMedRAG:

    def __init__(self, supabase_client):
        # Initialize components as None
        self.llm = None
        self.rag_chain = None
        self.supabase_client = supabase_client

    def initialize(self):
        global supabase_vs_retriever
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize the vectorstore and retriever
        vectorstore = SupabaseVectorStore(
            client=self.supabase_client,
            embedding=embedding_model,
            table_name="pubmed_documents",
            query_name="match_documents",
        )
        supabase_vs_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        retriever = MyRetriever()

        retrieval_tool = create_retriever_tool(
            retriever=retriever,
            name="pubmed_retriever",  
            description="A tool to retrieve documents from my knowledge base created from PubMedCentral Database.",
        )
        logger.debug("Initializing RAG pipeline")
        system_message = "You are a medical research assistant with expertise in analyzing PubMed papers. Use the following pieces of context from research papers to answer the user's question. "

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
        self.llm = create_llama_chat_model()

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
        """
        Processes a chunk from the agent and returns formatted output strings
        with enhanced markdown formatting for better readability.
        """
        outputs = []

        if "agent" in chunk:
            for message in chunk["agent"]["messages"]:
                if "tool_calls" in message.additional_kwargs:
                    tool_calls = message.additional_kwargs["tool_calls"]

                    for tool_call in tool_calls:
                        tool_name = tool_call["function"]["name"]
                        tool_arguments = eval(tool_call["function"]["arguments"])
                        tool_query = tool_arguments["query"]

                        # Enhanced markdown formatting for tool calls
                        tool_call_msg = (
                            f"\n\n> 🔍 **Research in Progress**\n\n"
                            f"Searching through literature using `{tool_name}`\n\n"
                            f"**Query**: {tool_query}\n\n"
                            f"---"  # Horizontal line for visual separation
                        )
                        outputs.append(tool_call_msg)

                else:
                    agent_answer = message.content
                    if agent_answer:  # Only append if there's content
                        # Enhanced markdown formatting for agent responses
                        formatted_answer = (
                            f"\n\n### Research Findings\n\n"
                            f"{agent_answer}\n\n"
                            f"---\n\n"  # Horizontal line for visual separation
                        )
                        outputs.append(formatted_answer)

        # Join all outputs with double newlines for better spacing
        return "\n\n".join(outputs) if outputs else None
                    
    def query_stream(self, question: str):
        logger.debug(f"Streaming RAG query: {question[:50]}...")

        if not self.rag_chain:
            raise ValueError("RAG chain not initialized. Call initialize() first.")

        try:
            # First collect all information without streaming
            for chunk in self.rag_chain.stream(
                {"messages": [("human", question)]},
                {"recursion_limit": RECURSION_LIMIT}
            ):
                # Process each chunk and yield formatted outputs
                formatted_outputs = self.process_chunks(chunk)
                if formatted_outputs:  # Only yield if there's something to yield
                    yield f"{formatted_outputs}\n\n"
            
            # # Check if reached max iterations and summarize
            # collected_info = self.extract_messages_from_agent_result(agent_result)
            # print(f"Collected information: {collected_info}")
            # summarization_prompt = f"""
            # Based on the following information collected about the query: "{question}"
            
            # {collected_info}
            
            # Please provide a concise and accurate summary that directly answers the user's question.
            # """
            
            # # Stream the summarization
            # for chunk in self.llm.stream(summarization_prompt):
            #     yield chunk.content
                
        except Exception as e:
            logger.error(f"Error in RAG query stream: {str(e)}", exc_info=True)
            yield f"Error generating response: {str(e)}"
