from backend.core import run_llm_chain
import streamlit as st
from typing import Set

from langchain_core.messages import HumanMessage, AIMessage

st.header("LangChain Documentation Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_answers_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_string = "Sources:\n"
    for i, source in enumerate(sorted(source_urls)):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response..."):
        response = run_llm_chain(
            question=prompt,
            chat_history=st.session_state["chat_history"],
        )

        sources = {
            doc.metadata.get("source")
            for doc in response["context"]
            if doc.metadata.get("source")
        }

        formatted_response = (
            f"{response['answer']}\n\n{create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

        # Correct LangChain message types
        st.session_state["chat_history"].append(HumanMessage(content=prompt))
        st.session_state["chat_history"].append(AIMessage(content=response["answer"]))


# Render chat history
for answer, question in zip(
    st.session_state["chat_answers_history"],
    st.session_state["user_prompt_history"],
):
    st.chat_message("user").write(question)
    st.chat_message("assistant").write(answer)
