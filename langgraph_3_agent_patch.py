
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph.message import add_messages

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

class GoalAgent:
    def __call__(self, state):
        messages = state["messages"]
        messages.append(SystemMessage(content="Break the user's vague career goal into a specific learning objective."))
        return {"messages": messages}

class MemoAgent:
    def __call__(self, state):
        messages = state["messages"]
        messages.append(SystemMessage(content="Split the goal into weekly priorities. Estimate how long each takes."))
        return {"messages": messages}

class RoadmapAgent:
    def __call__(self, state):
        messages = state["messages"]
        messages.append(SystemMessage(content="For each weekly task, recommend 1 online resource."))
        return {"messages": messages}

builder = StateGraph()
builder.add_node("goal_agent", GoalAgent())
builder.add_node("memo_agent", MemoAgent())
builder.add_node("roadmap_agent", RoadmapAgent())

builder.set_entry_point("goal_agent")
builder.add_edge("goal_agent", "memo_agent")
builder.add_edge("memo_agent", "roadmap_agent")
builder.add_edge("roadmap_agent", END)

planner = builder.compile()

# Example run
if __name__ == "__main__":
    messages = [{"role": "user", "content": "I want to get into design roles"}]
    result = planner.invoke({"messages": messages})
    print("Final Roadmap Plan:")
    for m in result["messages"]:
        print(m["content"] if isinstance(m, dict) else m)
