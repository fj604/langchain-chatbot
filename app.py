import streamlit as st
from langchain_aws.chat_models import ChatBedrockConverse
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.messages.utils import message_chunk_to_message

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSerperAPIWrapper
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun



import time
import boto3


@tool
def get_current_date_time() -> str:
    """Return the current date, time, day of the week, and time zone"""
    from datetime import datetime
    print("Current date and time tool called")
    now = datetime.now()
    return f"The current date and time is {now.strftime('%Y-%m-%d %H:%M:%S')} on a {now.strftime('%A')} in the {now.strftime('%Z')} time zone."

@tool
def no_such_tool() -> str:
    """Return a message indicating that the tool does not exist"""
    return "The tool you requested does not exist."

@tool
def search_web(query: str) -> str:
    """Search the web and return the result"""
    search_runner = DuckDuckGoSearchRun()
    try:
        search_results = search_runner.invoke(query)
        if search_results:
            return search_results
        else:
            return "No results found."
    except Exception as e:
        return f"An error occurred: {e}"

@tool
def search_google(query: str) -> str:
    """Search Google and return the result"""
    search = GoogleSerperAPIWrapper()
    try:
        return search.run(query)
    except Exception as e:
        return f"An error occurred: {e}"
    
@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return the result"""
    api_wrapper = WikipediaAPIWrapper()
    wiki_runner = WikipediaQueryRun(api_wrapper=api_wrapper, query=query)
    try:
        wiki_results = wiki_runner.invoke(query)
        if wiki_results:
            return wiki_results
        else:
            return "No results found."
    except Exception as e:
        return f"An error occurred: {e}"


@tool
def search_wikidata(query: str) -> str:
    """Search Wikidata and return the result"""
    try:
        wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())
        if wikidata:
            return wikidata
        else:
            return "No results found."
    except Exception as e:
        return f"An error occurred: {e}"


tools = [
    get_current_date_time,
    search_google,
    search_web,
    search_wikipedia,
    search_wikidata,
    no_such_tool,
]

tool_map = {
    "current_date_time": get_current_date_time,
    "search_google": search_google,
    "search_web": search_web,
    "search_wikipedia": search_wikipedia,
    "search_wikidata": search_wikidata,
    "no_such_tool": no_such_tool,
    }


# Set the page title and icon
st.set_page_config(page_title="🦜🔗 Chatbot App", page_icon="🤖")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]


def get_text(message_content):
    if not message_content:
        return
    elif isinstance(message_content, str):
        return message_content
    elif isinstance(message_content, list):
        for item in message_content:
            if isinstance(item, str):
                return item
            if item["type"] == "text":
                return item["text"]


def display_conversation_history():
    for message in st.session_state.messages:
        text = get_text(message.content)
        if text:
            if isinstance(message, HumanMessage):
                st.chat_message("user").markdown(text)
            elif isinstance(message, AIMessage):
                st.chat_message("assistant").markdown(text)
            elif isinstance(message, SystemMessage):
                pass

# Function to generate and display AI response
def generate_response(model_id, region):
    model = ChatBedrockConverse(model_id=model_id, region_name=region)
    model_with_tools = model.bind_tools(tools)

    # Create the chain with StrOutputParser for streaming
    chain = model_with_tools

    message_placeholder = st.empty()
    

    with message_placeholder:
        # Create a placeholder for the assistant's response
        assistant_message_placeholder = st.chat_message("assistant")

        response_placeholder = assistant_message_placeholder.markdown(
            "💭"
        )  # Initial placeholder for response

        # Loop through chunks and update the placeholder
        first = True
        text = ""

        for chunk in chain.stream(st.session_state.messages):
            if first:
                response = chunk
                first = False
            else:
                response += chunk
            if response.content:
                if isinstance(response.content, str):
                    text = response.content
                elif isinstance(response.content, list):
                    for item in response.content:
                        if item["type"] == "text":
                            text = item["text"]
                            break
                response_placeholder.markdown(text + "▌")
        response_placeholder.markdown(text)
    if not text:
        message_placeholder.empty()
    return message_chunk_to_message(response)


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
        if "region" in st.session_state
        and st.session_state.region in st.session_state.regions
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

# Display existing chat messages

display_conversation_history()

# Input field for user message
if prompt := st.chat_input("Enter your message here..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    # Generate AI response
    # Add the current prompt to the conversation history
    st.session_state.messages.append(HumanMessage(content=prompt))
    input_required = False
    while not input_required:
        response = generate_response(st.session_state.model_id, st.session_state.region)
        st.session_state.messages.append(response)
        if response.tool_calls:
                for tool_call in response.tool_calls:
                    print("Tool call:", tool_call)
                    tool_name = tool_call["name"]
                    selected_tool = tool_map.get(tool_name, None)
                    if not selected_tool:
                        selected_tool = no_such_tool
                        tool_name = "no_such_tool"
                        continue
                    print("Selected tool:", selected_tool)
                    with st.spinner(tool_name):
                        tool_message = selected_tool.invoke(tool_call)
                        print("Tool message:", tool_message)
                    st.session_state.messages.append(tool_message)
        else:
            input_required = True
