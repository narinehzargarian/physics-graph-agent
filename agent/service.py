"""Python API for the agent. Run with: uvicorn agent.service:app"""
import os

from fastapi import FastAPI
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from agent import data_pipeline
from agent.stores import AnswerCache, GraphStore, SearchStore
from agent.workflow import ask, build_workflow

app = FastAPI(title="Physics Graph Agent")
state = {}


def components():
    """Connects on first use, so the service can start before its databases are ready."""
    if not state:
        graph, search = GraphStore(), SearchStore()
        concepts, relations, _ = data_pipeline.load()
        graph.load(concepts, relations)
        search.load(concepts)
        llm = ChatOllama(
            model=os.environ.get("OLLAMA_MODEL", "gemma2:2b"),
            base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            temperature=0,
        )
        state["workflow"] = build_workflow(llm, search, graph)
        state["cache"] = AnswerCache()
    return state


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    session_id: str = Field(default="default", max_length=64)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_endpoint(body: Question):
    parts = components()
    # Follow-up questions depend on the conversation, so only a session's first question is cached.
    cached = parts["cache"].get(body.question) if body.session_id == "default" else None
    if cached:
        return {**cached, "cached": True}
    result = ask(parts["workflow"], body.question, thread_id=body.session_id)
    if body.session_id == "default" and result["grounded"]:
        parts["cache"].put(body.question, result)
    return {**result, "cached": False}
