import streamlit as st

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

        # Echo back the user's input as the bot response
        with st.chat_message("assistant"):
            response = prompt  # Simply echo back the input
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()