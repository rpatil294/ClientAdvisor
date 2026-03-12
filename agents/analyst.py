from langchain.agents import create_agent
from tools.vectordb import knowledge_search
from tools.websearch import web_search
from state import GraphState, ResearchResults
from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import SystemMessage, HumanMessage

class AnalystAgent:
    def __init__(self, llm):
        env = Environment(loader=FileSystemLoader("prompts"))
        system_prompt = env.get_template("analyst.txt").render()

        self.agent = create_agent(
            model=llm,
            tools=[knowledge_search, web_search],
            system_prompt=SystemMessage(content=system_prompt),
        )
        self.structured_llm = llm.with_structured_output(
            ResearchResults, method="function_calling"
        )

    def __call__(self, state: GraphState) -> dict:
        research_briefs = state.get("research_briefs", [])
        if not research_briefs:
            return {"research_results": [], "research_briefs": []}

        brief_text = "Research the following:\n"
        for brief in research_briefs:
            brief_text += f"- Topic: {brief['topic']}\n  Context: {brief['summary']}\n"

        result = self.agent.invoke({
            "messages": [{"role": "user", "content": brief_text}]},
            {"recursion_limit":25})

        raw_findings = result["messages"][-1].content
        structured: ResearchResults = self.structured_llm.invoke(
            f"Structure these research findings:\n\n{raw_findings}"
        )
        print(f"Structured Analyst Results: {structured}")

        return {
            "research_results": [structured.model_dump()],
            "research_briefs": [],
            "messages": [HumanMessage(content=f"Research complete: {raw_findings}")],

        }
