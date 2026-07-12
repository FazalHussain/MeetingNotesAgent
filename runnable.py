import json
import os

"""
LangGraph workflow for the demo chatbot.
This file defines the state, tool execution, and the runnable graph.
"""

from typing import Annotated, Dict, Literal
from typing_extensions import TypedDict

from openai import OpenAIError

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from langchain_openai import ChatOpenAI
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

try:
    from tools import available_functions
except ImportError:
    available_functions = []

model = os.getenv("LLM_MODEL", "gpt-4o-mini")
print(f"Using model: {model}", flush=True)

tools = list(available_functions.values())

llm = ChatOpenAI(
    model=model,
    api_key=os.getenv("OMNIROUTER_API_KEY"),
    base_url=os.getenv("OMNIROUTER_BASE_URL"),
    streaming=True,
)

llm_with_tools = llm.bind_tools(tools)


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


async def call_model(
    state: GraphState,
    config: RunnableConfig,
) -> Dict[str, AnyMessage]:
    """
    Invoke the language model with the current conversation state.

    Args:
        state: The current graph state containing the conversation messages.
        config: The runnable configuration for the current execution.

    Returns:
        A dictionary containing the model response messages.
    """

    try:
        response = await llm_with_tools.ainvoke(
            state["messages"],
            config,
        )
    except OpenAIError as exc:
        raise RuntimeError(
            "Unable to reach the configured model provider. "
            "Check the API key, base URL, and model name in the environment settings."
        ) from exc

    return {"messages": response}


def tool_node(state: GraphState):
    """
    Execute tool calls requested by the model.

    Args:
        state: The current graph state containing the latest assistant message.

    Returns:
        A dictionary containing the tool output messages.
    """

    last_message = state["messages"][-1]

    outputs = []

    for call in last_message.tool_calls:

        tool = available_functions[call["name"]]

        result = tool.invoke(call["args"])

        outputs.append(
            ToolMessage(
                content=result if isinstance(result, str) else json.dumps(result),
                tool_call_id=call["id"],
            )
        )

    return {"messages": outputs}


def should_continue(
    state: GraphState,
) -> Literal["tools", "__end__"]:
    """
    Decide whether the workflow should continue to tool execution.

    Args:
        state: The current graph state containing the latest message.

    Returns:
        Either the tool node route or the end marker for the workflow.
    """

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


def get_runnable():
    """
    Build and compile the LangGraph workflow.

    Returns:
        A compiled LangGraph runnable object with the configured nodes and memory.
    """

    workflow = StateGraph(GraphState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
    )

    workflow.add_edge("tools", "agent")

    memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)