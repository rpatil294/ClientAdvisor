from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from state import GraphState
from jinja2 import Environment, FileSystemLoader


class ClientAgent:
    def __init__(self, llm):
        env = Environment(loader=FileSystemLoader("prompts"))
        self.template = env.get_template("client.txt")
        self.llm = llm

    def __call__(self, state: GraphState) -> dict:
        client_profile = state.get("client_profile", {})
        system_prompt = self.template.render(client_profile=client_profile)
        conversation = []
        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                conversation.append(HumanMessage(content=msg.content))
            elif isinstance(msg, HumanMessage):
                conversation.append(AIMessage(content=msg.content))
            else:
                conversation.append(msg)

        messages = [SystemMessage(content=system_prompt)] + conversation
        response = self.llm.invoke(messages)
        
        print(f"Client Response: {response.content}")

        return {
            "messages": [HumanMessage(content=response.content)],
        }
