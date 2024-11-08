import streamlit as st
from langchain_aws.chat_models import ChatBedrockConverse
from langchain.schema import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="🦜🔗 Chatbot App", page_icon="🤖")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def generate_response(prompt):
    model = ChatBedrockConverse(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    memory = ConversationBufferMemory()
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            memory.add_message(HumanMessage(content=msg["content"]))
        else:
            memory.add_message(AIMessage(content=msg["content"]))
    
    chain = ConversationChain(llm=model, memory=memory, output_parser=StrOutputParser())
    assistant_message_placeholder = st.chat_message("assistant")
    response_placeholder = assistant_message_placeholder.markdown("...")
    response = ""

    for chunk in chain.stream(prompt):
        if isinstance(chunk, str):
            response += chunk
            response_placeholder.markdown(response + "▌")
        else:
            st.warning("Received unexpected chunk format. Please check model output.")

    response_placeholder.markdown(response)
    return response

if prompt := st.chat_input("Enter your message here..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
