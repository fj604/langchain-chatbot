import streamlit as st
from langchain_aws.chat_models import ChatBedrockConverse
from langchain.schema import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Set the page title and icon
st.set_page_config(page_title="🦜🔗 Chatbot App", page_icon="🤖")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to generate and display AI response
def generate_response(prompt):
    model = ChatBedrockConverse(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    response = ""
    
    # Initialize memory with existing conversation history
    memory = ConversationBufferMemory()
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            memory.add_message(HumanMessage(content=msg["content"]))
        else:
            memory.add_message(AIMessage(content=msg["content"]))
    
    # Create the chain with StrOutputParser for streaming
    chain = ConversationChain(llm=model, memory=memory, output_parser=StrOutputParser())
    
    # Create a placeholder for the assistant's response
    assistant_message_placeholder = st.chat_message("assistant")
    response_placeholder = assistant_message_placeholder.markdown("...")  # Initial placeholder for response

    # Generate response and update the placeholder
    for chunk in chain.stream(prompt):
        if isinstance(chunk, str):
            response += chunk
            response_placeholder.markdown(response + "▌")  # Update only the content in the placeholder
        else:
            st.warning("Received unexpected chunk format. Please check model output.")

    # Finalize the message
    response_placeholder.markdown(response)
    return response

# Input field for user message
if prompt := st.chat_input("Enter your message here..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Generate AI response
    response = generate_response(prompt)
    # Append AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
