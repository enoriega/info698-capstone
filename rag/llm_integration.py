import os
from typing import List, Dict, Any, Optional
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
LLM_APIKEY = os.getenv("LLM_APIKEY")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://llm-api.cyverse.ai/v1")


def create_llama_chat_model():
    return ChatOpenAI(
        model="Llama-3.2-11B-Vision-Instruct",
        openai_api_key=LLM_APIKEY,
        openai_api_base=LLM_API_BASE,
    )


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


def create_retrieval_prompt() -> ChatPromptTemplate:
    template = """
    You are a medical research assistant with expertise in analyzing PubMed papers.
    
    Use the following pieces of context from research papers to answer the user's question.
    If you don't know the answer, just say you don't know. Don't try to make up an answer.
    Always cite your sources using the PMID when available.
    
    CONTEXT:
    {context}
    
    USER QUESTION: {question}
    
    YOUR ANSWER:
    """
    return ChatPromptTemplate.from_template(template)


def create_standalone_query_prompt() -> PromptTemplate:
    template = """
    You are a medical research expert. Convert the user's question into the most effective search query 
    for finding relevant PubMed articles. Extract specific medical terms, conditions, treatments, or 
    research areas. Don't add explanations, just provide the optimized search query.
    
    Original question: {question}
    
    Optimized search query:
    """
    return PromptTemplate.from_template(template)


def create_query_transformer(llm):
    query_prompt = create_standalone_query_prompt()
    return query_prompt | llm | StrOutputParser()

# Depreciated Remove this function as it is not used in the main application. This used Query Translation.
def create_rag_chain(retriever, llm):
    # Create the retrieval prompt
    prompt = create_retrieval_prompt()

    # Create a query transformer for better retrieval
    query_transformer = create_query_transformer(llm)

    def format_docs(docs):
        formatted_docs = []
        for doc in docs:
            print(f"Document metadata: {doc.metadata}")
            title = doc.metadata.get("title", "No title available")
            # Check both uppercase and lowercase versions of PMID
            pmid = doc.metadata.get(
                "PMID", doc.metadata.get("pmid", "No PMID available")
            )
            formatted_docs.append(
                f"Document Title: {title}\nPMID: {pmid}\nContent: {doc.page_content}"
            )
        return "\n\n".join(formatted_docs)

    rag_chain = (
        RunnablePassthrough.assign(transformed_query=query_transformer)
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(
                retriever.get_relevant_documents(x["transformed_query"])
            )
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
