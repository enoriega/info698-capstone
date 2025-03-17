from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def get_llm_response(user_input: str) -> str:
    """
    Get response from the LLM model for the given user input.
    
    Args:
        user_input (str): The user's message
        
    Returns:
        str: The LLM's response
    """
    # Initialize the ChatOpenAI client
    chat = ChatOpenAI(
        model="Llama-3.2-11B-Vision-Instruct",
        openai_api_key="sk-OMIAMjsYE-00DdubObVWLg",
        openai_api_base="https://llm-api.cyverse.ai/v1"
    )

    # Create a HumanMessage object with the user input
    message = HumanMessage(content=user_input)

    # Send the message to the model using the `invoke` method
    response = chat.invoke([message])

    return response.content