import asyncio
import os
import re
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from runnable import get_runnable

THREAD_ID = str(uuid.uuid4())






SYSTEM_PROMPT = """
You are a meeting notes assistant. Your primary function is to answer questions by retrieving and synthesizing information from the meeting notes database.

RULES:
1. You MUST call the query_documents tool for every user question before providing an answer. Do not answer from memory or prior conversation context.
2. Base your answer strictly on the retrieved documents. Do not introduce, infer, or fabricate information that is not explicitly stated in the source material.
3. When the retrieved documents contain relevant information, synthesize a comprehensive answer drawing from ALL returned documents. A single query may return documents from different meetings -- combine them into a unified response.
4. Reference specific meeting names when available in the source metadata (e.g., "In the Database Optimization Discussion meeting...").
5. List all relevant decisions, action items, and participants mentioned in the documents.
6. Only say "I don't know" or "I couldn't find that information" if the retrieved documents genuinely contain no relevant information after the search.
"""


# ---------------------------------------------------------------------------
# Chatbot instance
# ---------------------------------------------------------------------------

@st.cache_resource
def chatbot():
    """Create and cache the LangGraph chatbot instance."""
    return get_runnable()


app = chatbot()



async def ask_llm(prompt: str) -> tuple[str, list[str]]:
    """
    Send the user's prompt to the LangGraph agent.

    Returns:
        (response, tool_outputs) — the assistant text and raw tool-output
        strings (used for source extraction).
    """
    config = {"configurable": {"thread_id": THREAD_ID}}

    result = await app.ainvoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        },
        config=config,
    )

    # Extract ToolMessage contents for source display
    tool_outputs = [
        msg.content
        for msg in result["messages"]
        if hasattr(msg, "tool_call_id") and msg.tool_call_id
    ]

    return result["messages"][-1].content, tool_outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    st.set_page_config(
        page_title="Meeting Notes Assistant",
        page_icon="🤖",
        layout="centered",
    )

    st.title("🤖 Meeting Notes Assistant")
    st.caption("Ask questions about your meeting notes.")

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Empty state
    if not st.session_state.history:
        st.info(
            "👋 Welcome! Ask me anything about your meeting notes, "
            "action items, decisions, or participants."
        )

    # Display previous messages
    for role, message in st.session_state.history:
        with st.chat_message(role):
            st.markdown(message)

    # Chat input
    if prompt := st.chat_input("Ask about a meeting, decision, or action item..."):

        # User message
        st.session_state.history.append(("user", prompt))

        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching meeting notes..."):
                response, tool_outputs = await ask_llm(prompt)

            response = response.replace("\\n", "\n").strip()

            st.markdown(response)

            if tool_outputs:
                with st.expander("📚 Retrieved Sources", expanded=False):
                    for i, output in enumerate(tool_outputs, start=1):
                        st.markdown(f"**Source {i}**")
                        st.markdown(output)
                        if i != len(tool_outputs):
                            st.divider()

        # Save assistant response
        st.session_state.history.append(("assistant", response))


if __name__ == "__main__":
    asyncio.run(main())
