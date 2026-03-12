from typing import TypedDict, Annotated, Literal
from typing_extensions import Required
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field


class ClientProfile(TypedDict):
    age: int
    financial_goals: str
    risk_tolerance: str
    assets: str
    investments: str
    annual_income: str

class ResearchBrief(BaseModel):
    topic: str = Field(
        description="The research topic"
    )
    summary: str = Field(
        description="Summary of findings on this topic"
    )

class ResearchResults(BaseModel):
    research_briefs: list[ResearchBrief] = Field(
        description="Briefs that were researched"
    )
    research_results: str = Field(
        description="Consolidated research findings"
    )

class AdvisorDecision(BaseModel):
    message: str = Field(
        description="Your message to the client or for the analyst"
    )
    next_agent: Literal["client", "analyst", "end"] = Field(
        description="Who to talk to next"
    )
    research_briefs: list[ResearchBrief] = Field(
        default=[],
        description="Research tasks for the analyst. Only populate when next_agent is analyst"
    )
    phase: Literal["discovery", "research", "recommendation", "refinement", "conclusion"] = Field(
        description="Current conversation phase"
    )
    resolution_reached: bool = Field(
        default=False, 
        description="True only when client has explicitly accepted therecommendation")


class GraphState(TypedDict, total=False):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    client_profile: ClientProfile

    research_briefs: list[dict]
    research_results: list[dict]

    recommendations: list[str]
    phase: Literal["discovery", "research", "recommendation", "refinement", "conclusion"]
    next_agent: Literal["client", "analyst", "end"]
    resolution_reached: bool

