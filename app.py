import streamlit as st
from langchain_aws.chat_models import ChatBedrockConverse
from langchain.schema import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import boto3

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
def generate_response(prompt, model_id, region):
    model = ChatBedrockConverse(model_id=model_id, region_name=region)
    response = ""

    # Prepare the input history to maintain context
    conversation_history = [
        (
            HumanMessage(content=msg["content"])
            if msg["role"] == "user"
            else AIMessage(content=msg["content"])
        )
        for msg in st.session_state.messages
    ]

    # Add the current prompt to the conversation history
    conversation_history.append(HumanMessage(content=prompt))

    # Create the chain with StrOutputParser for streaming
    chain = model | StrOutputParser()

    # Create a placeholder for the assistant's response
    assistant_message_placeholder = st.chat_message("assistant")
    with st.spinner("Assistant is typing..."):
        response_placeholder = assistant_message_placeholder.markdown(
            "..."
        )  # Initial placeholder for response

    # Loop through chunks and update the placeholder
    for chunk in chain.stream(conversation_history):
        if isinstance(chunk, str):
            response += chunk
            response_placeholder.markdown(
                response + "▌"
            )  # Update only the content in the placeholder
        else:
            st.warning("Received unexpected chunk format. Please check model output.")

    # Finalize the message
    response_placeholder.markdown(response)
    return response


# Get a list of all available models in Bedrock in the given region
def get_bedrock_models(region):
    client = boto3.client("bedrock", region_name=region)
    response = client.list_foundation_models(byOutputModality="TEXT")
    models = [model["modelId"] for model in response["modelSummaries"]]
    return models


def get_bedrock_inference_profiles(region):
    client = boto3.client("bedrock", region_name=region)
    response = client.list_inference_profiles()
    profiles = [
        profile["inferenceProfileId"]
        for profile in response["inferenceProfileSummaries"]
    ]
    return profiles


regions = [
    "us-east-1",
    "eu-west-2",
]

region = st.sidebar.selectbox("Select AWS Region", regions)

# Update model and inference profile options based on selected region
model_options = get_bedrock_models(region) + get_bedrock_inference_profiles(region)

# Sidebar for model selection
model_id = st.sidebar.selectbox(
    "Select Model or Inference Profile",
    model_options,
)

if st.sidebar.button("Clear chat history"):
    st.session_state.messages = []
    st.rerun()

# Input field for user message
if prompt := st.chat_input("Enter your message here..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Generate AI response
    response = generate_response(prompt, model_id, region)
    # Append AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
