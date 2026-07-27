import asyncio
import os
import re
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from runnable import get_runnable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUGGESTIONS = [
    ("What key decisions were made?", "📋"),
    ("What are the action items?", "✅"),
    ("Who participated in the meetings?", "👥"),
    ("Any project updates?", "📊"),
]

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

CUSTOM_CSS = """
<style>
/* ── Root & typography ─────────────────────────────────────────────── */
:root {
    --accent: #2563eb;
    --accent-light: #eff6ff;
    --bg-primary: #ffffff;
    --bg-secondary: #f9fafb;
    --text-primary: #111827;
    --text-secondary: #6b7280;
    --text-tertiary: #9ca3af;
    --border: #e5e7eb;
    --border-light: #f3f4f6;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.06);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 'Helvetica Neue', Arial, sans-serif;
}

/* ── Page layout ───────────────────────────────────────────────────── */
.stMain .block-container {
    max-width: 768px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 8rem !important;
}

/* ── Hide default Streamlit header decoration ──────────────────────── */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
}
header[data-testid="stHeader"] * {
    visibility: hidden;
}

/* ── Chat messages — universal ─────────────────────────────────────── */
div[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 1.25rem 0 !important;
    margin: 0 !important;
    border-bottom: 1px solid var(--border-light) !important;
    animation: fadeIn 0.3s ease-out;
}
div[data-testid="stChatMessage"]:last-of-type {
    border-bottom: none !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── User messages ─────────────────────────────────────────────────── */
div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-user"]
) {
    background: var(--accent-light) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.25rem !important;
    margin-left: 12% !important;
    max-width: 88% !important;
    border-bottom: none !important;
    border-left: 3px solid var(--accent) !important;
}
div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-user"]
) p {
    color: var(--text-primary) !important;
    font-size: 0.93rem !important;
    line-height: 1.6 !important;
}

/* ── Assistant messages ────────────────────────────────────────────── */
div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) {
    background: transparent !important;
    padding-left: 0.5rem !important;
    border-bottom: none !important;
}
div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) p {
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}

/* ── Avatar icons ──────────────────────────────────────────────────── */
div[data-testid="chatAvatarIcon-user"] {
    background: var(--accent) !important;
    border-radius: 50% !important;
}
div[data-testid="chatAvatarIcon-assistant"] {
    background: var(--text-primary) !important;
    border-radius: 50% !important;
}

/* ── Markdown inside assistant messages ─────────────────────────────── */
div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) h1,
div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) h2,
div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    margin-top: 1.25rem !important;
    margin-bottom: 0.5rem !important;
}
div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) h3 {
    font-size: 1rem !important;
}

div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) pre {
    background: #1e1e2e !important;
    color: #cdd6f4 !important;
    border-radius: var(--radius-sm) !important;
    padding: 1rem !important;
    margin: 0.75rem 0 !important;
    overflow-x: auto;
    border: 1px solid #313244 !important;
}
div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) code {
    background: #f3f4f6 !important;
    color: #d946a8 !important;
    padding: 0.15em 0.4em !important;
    border-radius: 4px !important;
    font-size: 0.875em !important;
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', Consolas, monospace !important;
}
div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) pre code {
    background: none !important;
    color: #cdd6f4 !important;
    padding: 0 !important;
    border-radius: 0 !important;
}

div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) ul,
div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) ol {
    padding-left: 1.5rem !important;
    margin: 0.5rem 0 !important;
}
div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) li {
    margin-bottom: 0.35rem !important;
    line-height: 1.6 !important;
    font-size: 0.95rem !important;
}

div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) blockquote {
    border-left: 3px solid var(--border) !important;
    padding: 0.5rem 0.75rem !important;
    color: var(--text-secondary) !important;
    margin: 0.75rem 0 !important;
    background: var(--bg-secondary) !important;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0 !important;
}

div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) table {
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 0.75rem 0 !important;
    font-size: 0.88rem !important;
}
div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) th {
    background: var(--bg-secondary) !important;
    font-weight: 600 !important;
    text-align: left !important;
    padding: 0.6rem 0.75rem !important;
    border-bottom: 2px solid var(--border) !important;
}
div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) td {
    padding: 0.5rem 0.75rem !important;
    border-bottom: 1px solid var(--border-light) !important;
}

div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) strong {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* ── Source expanders ──────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    margin-top: 0.75rem !important;
    background: var(--bg-secondary) !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] summary {
    font-size: 0.82rem !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: 0.6rem 0.8rem !important;
}
div[data-testid="stExpander"] summary:hover {
    color: var(--accent) !important;
}
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 0 0.8rem 0.6rem !important;
}

/* ── Typing indicator (three animated dots) ────────────────────────── */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 0.75rem 0;
}
.typing-indicator span {
    width: 8px;
    height: 8px;
    background: var(--text-tertiary);
    border-radius: 50%;
    animation: bounce 1.4s ease-in-out infinite;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30%           { transform: translateY(-6px); opacity: 1; }
}

/* ── Suggestion pills ──────────────────────────────────────────────── */
.suggestion-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.55rem 1rem;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 999px;
    font-size: 0.85rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s ease;
    white-space: nowrap;
}
.suggestion-pill:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--accent-light);
    box-shadow: var(--shadow-sm);
}

/* ── Chat input bar ────────────────────────────────────────────────── */
div[data-testid="stChatInput"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-md) !important;
    background: var(--bg-primary) !important;
    max-width: 768px !important;
    margin: 0 auto !important;
}
div[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

/* ── Spinner ───────────────────────────────────────────────────────── */
div[data-testid="stSpinner"] {
    padding: 0.5rem 0 !important;
}
div[data-testid="stSpinner"] p {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

/* ── Sidebar ───────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdown"] h2 {
    font-size: 1.15rem !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    padding-top: 0.25rem !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdown"] p {
    color: var(--text-secondary) !important;
    font-size: 0.88rem !important;
    line-height: 1.5 !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--text-tertiary) !important;
    font-weight: 600 !important;
}

/* ── Sidebar buttons ───────────────────────────────────────────────── */
section[data-testid="stSidebar"] button[kind="secondary"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    background: var(--bg-primary) !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}
section[data-testid="stSidebar"] button[kind="secondary"]:hover {
    border-color: #ef4444 !important;
    color: #ef4444 !important;
    background: #fef2f2 !important;
}

/* ── Clear any Streamlit default margins on main ───────────────────── */
section.main .block-container {
    padding-top: 1rem !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Chatbot instance
# ---------------------------------------------------------------------------

@st.cache_resource
def chatbot():
    """Create and cache the LangGraph chatbot instance."""
    return get_runnable()


app = chatbot()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_sources(text: str) -> list[str]:
    """Pull 'Source: ...' filenames from free-form text (fallback path)."""
    matches = re.findall(r"Source:\s*([^\n\]]+)", text, re.IGNORECASE)
    seen: set[str] = set()
    unique: list[str] = []
    for m in matches:
        s = m.strip().rstrip(".")
        if s and s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _extract_sources_from_tool_outputs(outputs: list[str]) -> list[str]:
    """Extract source filenames from the raw tool-output strings."""
    sources: list[str] = []
    for output in outputs:
        sources.extend(_extract_sources(output))
    return sources


async def ask_llm(prompt: str) -> tuple[str, list[str]]:
    """
    Send the user's prompt to the LangGraph agent.

    Returns:
        (response, tool_outputs) — the assistant text and raw tool-output
        strings (used for source extraction).
    """
    THREAD_ID = st.session_state.thread_id
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
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("## Meeting Notes AI")

        st.markdown(
            "Search through your meeting notes with AI. "
            "Get instant summaries of decisions, action items, "
            "participants, and key insights."
        )

        st.markdown("---")

        st.markdown("### Quick Start")
        st.markdown(
            "1. Ask a question in the chat\n"
            "2. The AI searches your meeting documents\n"
            "3. Get a synthesized answer with sources"
        )

        st.markdown("---")

        # Status
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:6px; '
            f'margin-bottom:6px;">'
            f'<span style="width:7px; height:7px; background:#22c55e; '
            f'border-radius:50%; display:inline-block;"></span>'
            f'<span style="font-size:0.82rem; color:var(--text-secondary);">'
            f'Model: <code>{model}</code></span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="display:flex; align-items:center; gap:6px; '
            'margin-bottom:6px;">'
            '<span style="width:7px; height:7px; background:#22c55e; '
            'border-radius:50%; display:inline-block;"></span>'
            '<span style="font-size:0.82rem; color:var(--text-secondary);">'
            'Documents: loaded</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        if st.button("New Chat", use_container_width=True, type="secondary"):
            st.session_state.history = []
            st.session_state.pending_prompt = None
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        st.markdown("---")

        st.markdown(
            '<p style="font-size:0.75rem; color:var(--text-tertiary); '
            'text-align:center; margin-top:1rem;">'
            'Powered by LangGraph · ChromaDB · Streamlit</p>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------

WELCOME_HTML = """
<div style="text-align:center; padding:6rem 1rem 2rem;">
    <div style="width:56px; height:56px; border-radius:16px;
                background:linear-gradient(135deg, #2563eb, #7c3aed);
                display:inline-flex; align-items:center; justify-content:center;
                margin-bottom:1.25rem; box-shadow:0 4px 16px rgba(37,99,235,0.25);">
        <span style="font-size:1.6rem;">📝</span>
    </div>
    <h1 style="font-size:1.75rem; font-weight:700; color:#111827;
               margin:0 0 0.4rem; letter-spacing:-0.02em;">
        Meeting Notes AI
    </h1>
    <p style="font-size:1rem; color:#6b7280; margin:0 0 0.25rem;
              font-weight:500;">
        What can I help you find?
    </p>
    <p style="font-size:0.88rem; color:#9ca3af; max-width:420px;
              margin:0 auto 2.5rem; line-height:1.5;">
        Ask anything about your meeting notes — decisions, action items,
        participants, and more.
    </p>
</div>
"""


def render_welcome_state():
    st.markdown(WELCOME_HTML, unsafe_allow_html=True)

    # Suggestion pills in a centered layout
    cols = st.columns([1, 2, 2, 1])
    for i, (text, icon) in enumerate(SUGGESTIONS):
        col = cols[i]
        with col:
            st.markdown(
                f'<div class="suggestion-pill">'
                f'<span>{icon}</span> {text}</div>',
                unsafe_allow_html=True,
            )
            if st.button(
                text, key=f"sug_{i}", use_container_width=True,
                help=f"Ask: {text}",
            ):
                st.session_state.pending_prompt = text
                st.rerun()


# ---------------------------------------------------------------------------
# Typing indicator
# ---------------------------------------------------------------------------

TYPING_HTML = """
<div class="typing-indicator">
    <span></span><span></span><span></span>
</div>
"""


# ---------------------------------------------------------------------------
# Message processing
# ---------------------------------------------------------------------------

async def process_message(prompt: str):
    """Display a user message, call the LLM, render the response + sources."""
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        # Show typing indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)

        response, tool_outputs = await ask_llm(prompt)

        # Clear typing indicator and show response
        typing_placeholder.empty()
        st.markdown(response)

        # --- Sources ---------------------------------------------------
        all_sources = _extract_sources_from_tool_outputs(tool_outputs)
        if not all_sources:
            all_sources = _extract_sources(response)

        # Deduplicate, preserving order
        seen: set[str] = set()
        unique_sources: list[str] = []
        for s in all_sources:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                unique_sources.append(s)

        if unique_sources:
            with st.expander(f"📎  {len(unique_sources)} source(s) referenced"):
                for src in unique_sources:
                    st.markdown(f"`{src}`")

    st.session_state.history.append({"role": "assistant", "content": response})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    st.set_page_config(
        page_title="Meeting Notes AI",
        page_icon="📝",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Session state ──────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # ── Sidebar ────────────────────────────────────────────────────────
    render_sidebar()

    # ── Process any pending suggestion before rendering the view ───────
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        await process_message(prompt)
        st.rerun()
        return

    # ── Render current view ────────────────────────────────────────────
    if not st.session_state.history:
        render_welcome_state()
    else:
        for msg in st.session_state.history:
            avatar = "🧑" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    # ── Chat input ─────────────────────────────────────────────────────
    if prompt := st.chat_input("Message Meeting Notes AI…"):
        await process_message(prompt)


if __name__ == "__main__":
    asyncio.run(main())
