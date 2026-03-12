**Multi-agent financial advisory system built on LangGraph. Three LLM-powered agents collaborate to provide personalized investment advice to a simulated client.**

**Advisor** -- orchestrates the conversation, asks questions, commissions research, delivers recommendations.
**Client** -- simulated client with a fixed profile. Responds in character, reveals information, pushes back on unsuitable advice. **Analyst** -- research agent with tool access (vector search + web search). Receives structured research briefs, returns findings. 

## Setup                                                                                                                      
```bash
cd ClientAdvisor
uv sync
cp .env.template .env
```

The system runs a fully automated conversation:
1. Advisor begins in discovery phase, asking the client about their financial situation
2. Client responds based on their profile (age 45, moderate risk, $500K assets, retire at 60)
3. Advisor commissions research by sending briefs to the analyst
4. Analyst uses knowledge store and web search to gather data
5. Advisor synthesizes findings into a recommendation
6. Client may push back refinement or accept
7. Conversation ends when the client accepts and the advisor sets `resolution_reached=True`
8. Recursion limit is set to 25 iterations to prevent infinite loops.

## How It Works                                                                                                 
                                                                                                              
### State Management                                                                                            
All agents share a single `GraphState` (TypedDict) with:
`messages` -- conversation history (append-only via `add_messages` reducer)
`client_profile` -- static client attributes
`research_briefs` -- tasks from advisor to analyst
`research_results` -- findings from analyst to advisor
`phase` -- discovery | research | recommendation | refinement | conclusion
`next_agent` -- client | analyst | end
`resolution_reached` -- termination flag

### Structured Output
                                                                                                              
The advisor uses `llm.with_structured_output(AdvisorDecision)` to return:
`message` -- what to say next
`next_agent` -- routing decision
`research_briefs` -- research tasks (when routing to analyst)
`phase` -- conversation phase
`resolution_reached` -- whether to terminate

### Tools  

**Knowledge Store** (`vectordb.py`):
Loads PDFs from `knowledge_store/` at startup
Chunks with `RecursiveCharacterTextSplitter` (600 chars, 100 overlap)
Embeds with HuggingFace `all-mpnet-base-v2`
In-memory ChromaDB with MMR retrieval (k=2)

**Web Search** (`websearch.py`):
Tavily Search (finance topic, advanced depth) if API key is set
DuckDuckGo fallback otherwise

### Message Role Handling                                                                                       
Since all agents share one message list but each agent is an LLM that produces `AIMessage`:                     
The client swaps message roles before calling the LLM (advisor's `AIMessage` becomes `HumanMessage` from the client's perspective)                                                                                        
The client stores its response as `HumanMessage` (from the advisor's perspective, the client is the human)
The analyst adds findings as `HumanMessage` so the advisor sees it as input                               

## Limitations                                                                                                  
**No persistence** -- conversation state and vector store are in-memory only
**Rate limits** -- with Anthropic's free/low tier, the analyst's tool-calling loop can hit token/minute limits, causing errors
**Single message list** -- all agents share one conversation history, requiring role-swapping logic to maintain correct message alternation   
**No Async Functionality** -- the system runs synchronously, which can lead to delays when the analyst is doing research. An async implementation would allow the advisor to continue processing while waiting for the analyst's findings.
**No User Context** -- the client is a static agent with a fixed profile and no memory of past interactions. 