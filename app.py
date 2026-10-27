"""A small, reliable multi-agent research brief generator.

The graph intentionally keeps agent responsibilities separate:
researcher -> analyst -> critic -> finalizer.
Each node receives and returns typed shared state, so intermediate results
remain inspectable and can be validated before the final answer is printed.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()


class ResearchNotes(BaseModel):
    """Evidence gathered by the researcher."""

    key_points: list[str] = Field(min_length=3, max_length=8)
    assumptions: list[str] = Field(default_factory=list, max_length=5)
    sources_to_verify: list[str] = Field(default_factory=list, max_length=5)


class Analysis(BaseModel):
    """Decision-oriented synthesis of the research notes."""

    summary: str = Field(min_length=80, max_length=1200)
    recommendations: list[str] = Field(min_length=2, max_length=6)
    risks: list[str] = Field(default_factory=list, max_length=5)


class Review(BaseModel):
    """Quality gate result produced before finalization."""

    approved: bool
    issues: list[str] = Field(default_factory=list, max_length=6)
    corrections: list[str] = Field(default_factory=list, max_length=6)


class BriefState(TypedDict, total=False):
    topic: str
    research: ResearchNotes
    analysis: Analysis
    review: Review
    final_answer: str
    error: str


@tool
def calculate_reading_time(text: str) -> str:
    """Estimate how many minutes it takes to read a piece of text."""
    words = len(text.split())
    return f"{words} words, approximately {max(1, round(words / 200))} minute(s) to read"


def build_model() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Copy .env.example to .env and add your Gemini API key."
        )
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        temperature=0.2,
        max_retries=0,
        timeout=45,
    )


def researcher(state: BriefState) -> BriefState:
    print("Researcher agent is working...", file=sys.stderr, flush=True)
    prompt = (
        f"Research the topic: {state['topic']}\n"
        "Give practical, current, balanced notes for a decision-maker. "
        "Separate facts from assumptions and list claims that should be verified. "
        "Do not invent citations."
    )
    try:
        model = build_model().with_structured_output(ResearchNotes)
        result = model.invoke(
            [
                SystemMessage(content="You are the research agent. Be concise and factual."),
                HumanMessage(content=prompt),
            ]
        )
        return {"research": result}
    except Exception as exc:
        return {"error": f"Research agent failed: {exc}"}


def analyst(state: BriefState) -> BriefState:
    if state.get("error"):
        return {}
    print("Analyst agent is working...", file=sys.stderr, flush=True)
    notes = state["research"].model_dump_json(indent=2)
    try:
        model = build_model().with_structured_output(Analysis)
        result = model.invoke(
            [
                SystemMessage(
                    content="You are the analysis agent. Turn research into actionable advice."
                ),
                HumanMessage(
                    content=f"Topic: {state['topic']}\nResearch notes:\n{notes}\n"
                    "Explain trade-offs and give recommendations grounded only in these notes."
                ),
            ]
        )
        return {"analysis": result}
    except Exception as exc:
        return {"error": f"Analysis agent failed: {exc}"}


def critic(state: BriefState) -> BriefState:
    if state.get("error"):
        return {}
    print("Critic agent is working...", file=sys.stderr, flush=True)
    try:
        model = build_model().with_structured_output(Review)
        result = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the critic agent and quality gate. Check unsupported claims, "
                        "missing trade-offs, and whether recommendations answer the topic."
                    )
                ),
                HumanMessage(
                    content=f"Topic: {state['topic']}\n"
                    f"Research: {state['research'].model_dump_json()}\n"
                    f"Analysis: {state['analysis'].model_dump_json()}\n"
                    "Approve only if the answer is useful and internally consistent."
                ),
            ]
        )
        return {"review": result}
    except Exception as exc:
        return {"error": f"Critic agent failed: {exc}"}


def finalizer(state: BriefState) -> BriefState:
    print("Finalizing the brief...", file=sys.stderr, flush=True)
    if state.get("error"):
        return {"final_answer": f"Workflow stopped safely: {state['error']}"}
    analysis = state["analysis"]
    review = state["review"]
    body = (
        f"# Brief: {state['topic']}\n\n"
        f"## Summary\n{analysis.summary}\n\n"
        "## Recommendations\n"
        + "\n".join(f"- {item}" for item in analysis.recommendations)
        + "\n\n## Risks and assumptions\n"
        + "\n".join(f"- {item}" for item in (analysis.risks or state["research"].assumptions))
    )
    if not review.approved:
        body += "\n\n## Review notes\n" + "\n".join(
            f"- {item}" for item in (review.issues + review.corrections)
        )
    return {"final_answer": body + f"\n\n_{calculate_reading_time.invoke(body)}_"}


def build_graph():
    graph = StateGraph(BriefState)
    graph.add_node("researcher", researcher)
    graph.add_node("analyst", analyst)
    graph.add_node("critic", critic)
    graph.add_node("finalizer", finalizer)
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "critic")
    graph.add_edge("critic", "finalizer")
    graph.add_edge("finalizer", END)
    return graph.compile()


def run(topic: str) -> str:
    if not topic.strip():
        raise ValueError("Topic cannot be empty.")
    result = build_graph().invoke({"topic": topic.strip()})
    return result["final_answer"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a reviewed multi-agent brief.")
    parser.add_argument(
        "topic",
        nargs="*",
        help="Question or topic to research; omit it to be prompted interactively",
    )
    args = parser.parse_args()
    topic = " ".join(args.topic).strip()
    if not topic:
        topic = input("Enter a question or topic to research: ").strip()
    if not topic:
        parser.error("a non-empty topic is required")
    print(run(topic))
