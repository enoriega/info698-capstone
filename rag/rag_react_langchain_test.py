import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from rich.console import Console
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain import hub
from langgraph.errors import GraphRecursionError
from langchain_community.retrievers import WikipediaRetriever
import weaviate
from weaviate.auth import AuthApiKey
from sentence_transformers import SentenceTransformer
import logging
import ipywidgets as widgets
from IPython.display import display, clear_output
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores import Weaviate
from langchain_weaviate import WeaviateVectorStore
from weaviate.connect import ConnectionParams

# Constants
load_dotenv()
rich = Console()
RECURSION_LIMIT = 100
LLM_APIKEY = os.getenv("LLM_APIKEY")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://llm-api.cyverse.ai/v1")

collection_value = "PubMedArticle"
search_type_value = "vector"

# Configure logging - simplified
logging.basicConfig(
    filename="TestRAG.log", format="%(asctime)s %(message)s", filemode="w"
)


# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


def connect_to_weaviate(
    host="149.165.151.58",
    port=8080,
    grpc_port=50051,
    api_key="ZEwUMscMSntgpraNQGU0EcBGXX5JccIgoJ0yk+k1sGE=",
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


# Create an adapter class that provides the methods LangChain expects, this adapter is created to handle runtime errors.
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()


def create_llama_chat_model():
    return ChatOpenAI(
        model="Llama-3.2-11B-Vision-Instruct",
        openai_api_key=LLM_APIKEY,
        openai_api_base=LLM_API_BASE,
    )


# def process_chunks(chunk):
#     """
#     Processes a chunk from the agent and displays detailed information about tool calls,
#     including their outputs.
#     """

#     if "agent" in chunk:
#         for message in chunk["agent"]["messages"]:
#             if "tool_calls" in message.additional_kwargs:
#                 tool_calls = message.additional_kwargs["tool_calls"]

#                 for tool_call in tool_calls:
#                     tool_name = tool_call["function"]["name"]
#                     tool_arguments = eval(tool_call["function"]["arguments"])
#                     tool_query = tool_arguments["query"]

#                     rich.print(
#                         f"\nCalling [on deep_sky_blue1]{tool_name}[/on deep_sky_blue1] "
#                         f"\nQuery: [on deep_sky_blue1]{tool_query}[/on deep_sky_blue1].",
#                         style="deep_sky_blue1",
#                     )

#                     #Capture tool response
#                     tool_response = retrieval_tool.invoke({"query": tool_query})

#                     rich.print(
#                         f"\n[green]Tool Response:[/green]\n{tool_response}\n",
#                         style="green",
#                     )

#             else:
#                 agent_answer = message.content
#                 rich.print(f"\nAgent:\n{agent_answer}", style="black on white")


def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        print(f"Document metadata: {doc.metadata}")
        title = doc.metadata.get("title", "No title available")
        # Check both uppercase and lowercase versions of PMID
        pmid = doc.metadata.get("PMID", doc.metadata.get("pmid", "No PMID available"))
        formatted_docs.append(
            f"Document Title: {title}\nPMID: {pmid}\nContent: {doc.page_content}"
        )
    return "\n\n".join(formatted_docs)


class MyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> list[Document]:
        coll = client.collections.get("PubMedArticle")
        embedding_model = get_embedding_model()
        query_vector = embedding_model.encode(query).tolist()
        results = coll.query.near_vector(near_vector=query_vector, limit=2)
        documents = []
        for obj in results.objects:
            text_content = f"Title: {obj.properties.get('title', '')}\n"
            text_content += f"Abstract: {obj.properties.get('text', '')}"

            metadata = {
                "pmid": obj.properties.get("pmid", ""),
                "journal": obj.properties.get("journal", ""),
                "source": "PubMed",
            }
            doc = Document(page_content=text_content, metadata=metadata)
            documents.append(doc)
        logger.info(f"Retrieved {len(documents)} documents")

        return documents


# Global

client = None
llm = None
# Main Code Logic
try:
    client = connect_to_weaviate()
    print("Created Weaviate Client")
    llm = create_llama_chat_model()
    print("Created LLM Model instance")

    # messages = [
    #     (
    #         "system",
    #         "You are a helpful assistant that translates English to French. Translate the user sentence.",
    #     ),
    #     ("human", "I love programming."),
    # ]
    # ai_msg = llm.invoke(messages)
    # print(ai_msg)

    # st_model = SentenceTransformerEmbeddings()

    # vectorstore = SupabaseVectorStore(
    #     client=supabase_client,
    #     embedding=st_model,
    #     table_name="pubmed_documents",
    #     query_name="match_documents",
    # )

    # supabase_vs_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    embedding_model = SentenceTransformer("pritamdeka/S-PubMedBERT-MS-MARCO")
    vectorstore = WeaviateVectorStore(
        client=client,
        index_name="PubMedArticle",
        text_key="text",
        embedding=embedding_model,
    )
    vectorstore_as_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    retriever = MyRetriever()

    retrieval_tool = create_retriever_tool(
        retriever=retriever,
        name="pubmed_retriever",
        description="A tool to retrieve documents from my knowledge base created from PubMedCentral Database.",
    )

    # print(retrieval_tool.invoke("Lysosomal hydrolases of the epidermis , why is it used?"))

    system_message = "You are a medical research assistant with expertise in analyzing PubMed papers."

    wiki_retriever = create_retriever_tool(
        retriever=WikipediaRetriever(),
        name="wikipedia_retriever",
        description="Wikipedia retriever to search for medical terms and helping for information which is not in the knowledge base.",
    )

    langgraph_agent = create_react_agent(
        model=llm, tools=[retrieval_tool, wiki_retriever], prompt=system_message
    )

    # query = (
    #     "What was the main objective of the study comparing lorazepam and pentobarbital?"
    # )
    # messages = langgraph_agent.invoke({"messages": [("human", query)]})
    # # {
    # #     "input": query,
    # #     "output": messages["messages"][-1].content,
    # # }
    # print(messages)

    # try:
    #     for chunk in langgraph_agent.stream(
    #         {"messages": [("human", query)]},
    #         {"recursion_limit": RECURSION_LIMIT},
    #         stream_mode="values",
    #     ):
    #         print(chunk["messages"][-1])
    # except GraphRecursionError:
    #     print({"input": query, "output": "Agent stopped due to max iterations."})

    # Loop until the user chooses to quit the chat
    while True:
        # Get the user's question and display it in the terminal
        user_question = input("\nUser:\n")

        # Check if the user wants to quit the chat
        if user_question.lower() == "quit":
            rich.print("\nAgent:\nHave a nice day! :wave:\n", style="black on white")
            break

        # Use the stream method of the LangGraph agent to get the agent's answer
        for chunk in langgraph_agent.stream(
            {"messages": [HumanMessage(content=user_question)]}
        ):
            # Process the chunks from the agent
            print(chunk, end="", flush=True)

finally:
    if client:
        client.close()

# # What was the main objective of the study comparing lorazepam and pentobarbital?
# # Which drug provided greater sedation and antianxiety effects?
# # What were the dosages of lorazepam and pentobarbital used in the study?
# hypnotic drugs, including barbiturates, quinazolinones, and benzodiazepines,
