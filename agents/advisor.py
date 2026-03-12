from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from state import GraphState, AdvisorDecision
from jinja2 import Environment, FileSystemLoader

class AdvisorAgent:
    def __init__(self, llm):
        env = Environment(loader=FileSystemLoader("prompts"))
        self.template = env.get_template("advisor.txt")
        self.structured_llm = llm.with_structured_output(
            AdvisorDecision, method="function_calling"
        )

    def __call__(self, state: GraphState) -> dict:
        client_profile = state.get("client_profile", {})
        phase = state.get("phase", "discovery")
        research_results = state.get("research_results", [])
        system_prompt = self.template.render(
            client_profile=client_profile,
            phase=phase,
            research_results=research_results,
        )

        conversation = list(state["messages"])

        new_messages = []
        
        if not conversation:
            conversation.append(HumanMessage(content="Hello, Greetings. How can I assist you today?"))
        elif isinstance(conversation[-1], AIMessage):
            conversation.append(HumanMessage(content="Continue."))

        messages = [SystemMessage(content=system_prompt)] + conversation
        decision: AdvisorDecision = self.structured_llm.invoke(messages)
        new_messages.append(AIMessage(content=decision.message))


        result = {
            "messages": new_messages,
            "next_agent": decision.next_agent,
            "phase": decision.phase,
            "resolution_reached": decision.resolution_reached,
        }

        if decision.next_agent == "analyst" and decision.research_briefs:
            result["research_briefs"] = [
                {"topic": brief.topic, "summary": brief.summary}
                for brief in decision.research_briefs
            ]

        print(f"Advisor Message: {decision.message}")

        return result

