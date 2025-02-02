import streamlit as st
from langchain_aws.chat_models import ChatBedrockConverse
from langchain.schema import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import initialize_agent, Tool

import boto3


@tool
def current_date_time() -> str:
    """Return the current date and time in the format 'YYYY-MM-DD HH:MM:SS TZ'."""
    from datetime import datetime

    print("Current date and time tool called")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


tools = [
    current_date_time,
]

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
    model_with_tools = model.bind_tools(tools)
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
    chain = model_with_tools | StrOutputParser()

    # Create a placeholder for the assistant's response
    assistant_message_placeholder = st.chat_message("assistant")
    with st.spinner("Assistant is typing..."):
        response_placeholder = assistant_message_placeholder.markdown(
            "..."
        )  # Initial placeholder for response

    # Loop through chunks and update the placeholder
    for chunk in chain.stream(conversation_history):
        print(chunk)
        if isinstance(chunk, str):
            response += chunk
            response_placeholder.markdown(
                response + "▌"
            )  # Update only the content in the placeholder
        else:
            st.warning("Received unexpected chunk format. Please check model output.")
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
    "us-west-2",
    "eu-west-2",
    "eu-west-1",
    "eu-west-3",
    "eu-central-1",
]

default_region = "us-east-1"

default_model_id = "anthropic.claude-3-haiku-20240307-v1:0"

if "regions" not in st.session_state:
    st.session_state.regions = regions

region = st.sidebar.selectbox(
    "Select AWS Region",
    st.session_state.regions,
    index=(
        st.session_state.regions.index(st.session_state.region)
        if "region" in st.session_state and st.session_state.region in st.session_state.regions
        else 0
    ),
)

if "region" not in st.session_state or st.session_state.region != region:
    st.session_state.region = region
    st.session_state.model_options = get_bedrock_models(
        region
    ) + get_bedrock_inference_profiles(region)

if "model_id" not in st.session_state:
    st.session_state.model_id = default_model_id
    print("Setting default model_id to", default_model_id)

# Sidebar for model selection
model_id = st.sidebar.selectbox(
    "Select Model or Inference Profile",
    st.session_state.model_options,
    index=(
        st.session_state.model_options.index(st.session_state.model_id)
        if st.session_state.model_id in st.session_state.model_options
        else 0
    ),
)

if st.session_state.model_id != model_id:
    st.session_state.model_id = model_id

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
    response = generate_response(prompt, st.session_state.model_id, st.session_state.region)
    # Append AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
