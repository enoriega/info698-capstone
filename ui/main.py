import streamlit as st
import json
import os
from llm_response import get_llm_response

# Define the path for storing chat history
CHAT_HISTORY_FILE = "chat_history.json"

def load_chat_history():
    """Load chat history from file or initialize if not exists."""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r') as f:
                return json.load(f)
        else:
            return []
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")
        return []

def save_chat_history(messages):
    """Save chat history to file."""
    try:
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump(messages, f)
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")

def display_message(message):
    """Display a single message."""
    with st.chat_message(message["role"]):
        st.write(message["content"])

def initialize_session():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history()

def main():
    st.set_page_config(
        page_title="PubMed Chatbot",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("PubMed Chatbot")
    
    # Initialize session
    initialize_session()
    
    # Add a clear chat button in the sidebar
    with st.sidebar:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            save_chat_history([])
            st.rerun()
    
    # Create a container for chat history
    chat_container = st.container()
    
    # Handle user input
    if user_input := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get assistant response
        with st.spinner("Thinking..."):
            response = get_llm_response(user_input, st.session_state.chat_history[:-1])
            
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Save updated chat history to file
        save_chat_history(st.session_state.chat_history)
    
    # Display all messages in the chat container
    with chat_container:
        for message in st.session_state.chat_history:
            display_message(message)

if __name__ == "__main__":
    main()