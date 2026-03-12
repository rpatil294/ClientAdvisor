import os
import sys
from dotenv import load_dotenv

from graph import build_graph
from langgraph.errors import GraphRecursionError

load_dotenv()

def main():
    required_vars = ["LLM_MODEL", "LLM_PROVIDER"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        sys.exit(1)
    graph = build_graph()

    client_profile = {
        "age": 45,
        "financial_goals": "Retire at 60, fund child's college in 5 years",
        "risk_tolerance": "moderate",
        "assets": "$500,000",
        "investments": "50% savings, 30% 401k index funds, 20% individual stocks",
        "annual_income": "$180,000",
    }

    initial_state = {
        "messages": [],
        "client_profile": client_profile,
        "research_briefs": [],
        "research_results": [],
        "recommendations": [],
        "phase": "discovery",
        "next_agent": "client",
        "resolution_reached": False,
    }

    try:
        result = graph.invoke(initial_state, {"recursion_limit": 25})
        print("\n--- Conversation ---")
        for msg in result["messages"]:
            role = msg.__class__.__name__
            print(f"\n[{role}]: {msg.content}")
            print("====" * 20)
            print("\n--- Final State ---")
            print(f"Phase: {result.get('phase')}")
            print(f"Resolution: {result.get('resolution_reached')}")
            print(f"Research results: {len(result.get('research_results', []))}")
    except GraphRecursionError:
        print("Conversation hit recursion limit")

if __name__ == "__main__":
    main()