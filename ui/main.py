import streamlit as st
from llm_response import get_llm_response

def main():
    # Set page title and configure layout
    st.set_page_config(page_title="PubMed Chatbot", layout="wide")

    # Add a title
    st.title("PubMed Chatbot")

    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to say?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from LLM
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_llm_response(prompt)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()