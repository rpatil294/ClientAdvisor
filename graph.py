import os
from dotenv import load_dotenv
from state import GraphState
from langchain.chat_models import init_chat_model
from agents.advisor import AdvisorAgent
from agents.analyst import AnalystAgent
from agents.client import ClientAgent
from langgraph.graph import  START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph

load_dotenv()

def build_graph() -> CompiledStateGraph:
    llm = init_chat_model(
        os.getenv("LLM_MODEL"),
        model_provider=os.getenv("LLM_PROVIDER"),
        temperature=0.0,
        max_tokens=6000,
        max_retries=3,
    )

    advisor = AdvisorAgent(llm)
    client = ClientAgent(llm)
    analyst = AnalystAgent(llm)

    workflow = StateGraph(GraphState)

    workflow.add_node("advisor", advisor)
    workflow.add_node("client", client)
    workflow.add_node("analyst", analyst)

    workflow.add_edge(START, "advisor")

    def router(state: GraphState) -> str:
        if state.get("resolution_reached", False):
            return "end"
        return state["next_agent"]

    workflow.add_conditional_edges(
        "advisor", router,
        {"client": "client", 
         "analyst": "analyst", 
         "end": END},
    )

    workflow.add_edge("client", "advisor")
    workflow.add_edge("analyst", "advisor")

    return workflow.compile()
