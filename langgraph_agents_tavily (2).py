
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage
from tavily import TavilyClient

# State type
class AgentState(TypedDict):
    messages: List[BaseMessage]

# Load secrets
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)
client = TavilyClient(api_key=TAVILY_API_KEY)

class GoalAgent:
    def __call__(self, state: AgentState) -> AgentState:
        messages = state["messages"]
        messages.append(SystemMessage(content="Break the user's vague career goal into a clear, achievable learning objective."))
        return {"messages": messages}

class MemoAgent:
    def __call__(self, state: AgentState) -> AgentState:
        messages = state["messages"]
        messages.append(SystemMessage(content="Split the career goal into a weekly roadmap with clear priorities and task durations."))
        return {"messages": messages}

class RoadmapAgent:
    def __call__(self, state: AgentState) -> AgentState:
        messages = state["messages"]
        goal = next((m.content for m in messages if m.type == "human"), "career development")

        response = client.crawl(
            url="https://www.coursera.org/collections/free-courses-career-development",
            instructions=f"Search for beginner-friendly, free career development courses or learning resources that help someone achieve this goal: {goal}. Prioritize structured tutorials, certifications, and practical guides."
        )

        resources = response.get("results", [])
        resource_texts = [f"- {res['title']} ({res['url']})" for res in resources if 'title' in res and 'url' in res]
        resource_summary = "\n".join(resource_texts[:5]) if resource_texts else "No resources found."

        messages.append(SystemMessage(content=f"🔍 External Search Results:\n{resource_summary}"))
        return {"messages": messages}

# Define state graph
graph = StateGraph(AgentState)
graph.add_node("goal_agent", GoalAgent())
graph.add_node("memo_agent", MemoAgent())
graph.add_node("roadmap_agent", RoadmapAgent())

graph.set_entry_point("goal_agent")
graph.add_edge("goal_agent", "memo_agent")
graph.add_edge("memo_agent", "roadmap_agent")
graph.add_edge("roadmap_agent", END)

planner = graph.compile()

def run_langgraph_pipeline(user_input: str) -> str:
    result = planner.invoke({"messages": [{"type": "human", "content": user_input}]})
    return "\n".join([m.content for m in result["messages"]])
