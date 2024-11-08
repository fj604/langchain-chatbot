import streamlit as st
from langchain_aws.chat_models import ChatBedrockConverse
from langchain.schema import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the page title and icon
st.set_page_config(page_title="🦜🔗 Chatbot App", page_icon="🤖")

# Initialize ConversationBufferMemory in session state
if "memory" not in st.session_state:
    # Limiting memory to manage token usage and maintain efficient conversation history
    st.session_state.memory = ConversationBufferMemory(max_token_limit=1000, return_messages=True)

# Display existing chat messages from memory
for message in st.session_state.memory.chat_memory.messages:
    with st.chat_message(message.role):
        st.markdown(message.content)

# Function to generate and display AI response
def generate_response(prompt):
    model = ChatBedrockConverse(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    response = ""

    # Get the conversation history from memory
    conversation_history = st.session_state.memory.chat_memory.messages

    # Add the current prompt to the conversation history
    conversation_history.append(HumanMessage(content=prompt))

    # Create the chain with StrOutputParser for streaming
    chain = model | StrOutputParser()

    # Create a placeholder for the assistant's response
    assistant_message_placeholder = st.chat_message("assistant")
    response_placeholder = assistant_message_placeholder.markdown("...")  # Initial placeholder for response

    # Loop through chunks and update the placeholder
    try:
        for chunk in chain.stream(conversation_history):
            if isinstance(chunk, str):
                response += chunk
                response_placeholder.markdown(response + "▌")  # Update only the content in the placeholder
            else:
                logger.warning("Received unexpected chunk format: %s", type(chunk))
                st.warning("Received unexpected chunk format. Please check model output.")
    except Exception as e:
        logger.error("Error during response generation: %s", str(e))
        st.error("An error occurred while generating a response. Please try again later.")

    # Finalize the message
    response_placeholder.markdown(response)

    # Store both the user prompt and AI response in memory
    st.session_state.memory.chat_memory.add_user_message(prompt)
    st.session_state.memory.chat_memory.add_ai_message(response)

    return response

# Input field for user message
if prompt := st.chat_input("Enter your message here..."):
    # Strip any leading/trailing whitespace from user input
    prompt = prompt.strip()

    # Check if prompt is not empty after stripping
    if prompt:
        # Display user message
        st.chat_message("user").markdown(prompt)
        # Generate AI response
        response = generate_response(prompt)
    else:
        st.warning("Please enter a valid message before submitting.")

