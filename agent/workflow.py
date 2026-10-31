"""The multi-agent workflow, built with LangGraph.

    planner --> search agent --> graph agent --> writer --> checker --> done
                                                   ^           |
                                                   +-- retry --+

The planner decides which retrieval a question needs. The writer may only use retrieved facts and
must cite them. The checker verifies the citations in code rather than trusting the model, and
sends failures back to the writer with feedback.
"""
import json
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

MAX_ATTEMPTS = 3
MIN_SEARCH_SCORE = 5.0  # Off-topic questions peak near 3.3 and real concepts start near 12 on this data.
CANNOT_ANSWER = "I can't answer that from my knowledge base."
CITATION_RE = re.compile(r"\[(\d+)\]")

PLANNER_PROMPT = (
    "You route physics questions. Reply with only JSON like "
    '{"route": "search"} or {"route": "connect"}. Use "connect" when the question asks how two '
    'concepts relate or connect. Use "search" for everything else.'
)
WRITER_PROMPT = (
    "Answer the question in 1 to 3 sentences using only the numbered facts. Put the fact number in "
    "brackets after each claim, like [1]. Facts like 'A USES B' mean A uses B. "
    "If the facts don't answer the question, say so."
)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # Kept across turns by the checkpointer.
    question: str
    route: str
    hits: list[dict]
    facts: list[str]
    answer: str
    citations: list[int]
    feedback: str
    attempts: int
    trace: list[str]  # Steps of the current question only; ask() resets it.
    grounded: bool


def parse_route(text):
    """Small models often wrap JSON in other text, so find it and fall back safely."""
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if match:
        try:
            route = json.loads(match.group(0)).get("route")
            if route in ("search", "connect"):
                return route
        except (json.JSONDecodeError, AttributeError):
            pass
    return "search"


def check_citations(answer, fact_count):
    """Returns (cited numbers, problem or None). Runs in code, so the model can't talk its way past it."""
    cited = sorted({int(n) for n in CITATION_RE.findall(answer)})
    if not cited:
        return cited, "Your answer has no citations. Cite the fact numbers in brackets."
    invalid = [n for n in cited if n < 1 or n > fact_count]
    if invalid:
        return cited, f"Citations {invalid} don't exist. Only facts 1 to {fact_count} exist."
    return cited, None


def build_workflow(llm, search_store, graph_store, checkpointer=None):
    def planner(state):
        reply = llm.invoke([SystemMessage(PLANNER_PROMPT), HumanMessage(state["question"])])
        route = parse_route(reply.content)
        return {"route": route, "trace": state["trace"] + [f"planner:{route}"], "attempts": 0, "feedback": ""}

    def search_agent(state):
        hits = [h for h in search_store.search(state["question"]) if h["score"] >= MIN_SEARCH_SCORE]
        facts = [f"{h['name']}: {h['description']}" for h in hits]
        return {"hits": hits, "facts": facts, "trace": state["trace"] + [f"search:{len(hits)}"]}

    def graph_agent(state):
        hits = state.get("hits", [])
        links = []
        if state["route"] == "connect" and len(hits) >= 2:
            links = graph_store.shortest_path(hits[0]["key"], hits[1]["key"])
        elif hits:
            links = graph_store.neighbors(hits[0]["key"])
        return {"facts": state["facts"] + links, "trace": state["trace"] + [f"graph:{len(links)}"]}

    def writer(state):
        numbered = "\n".join(f"[{i}] {fact}" for i, fact in enumerate(state["facts"], start=1))
        messages = [SystemMessage(WRITER_PROMPT)]
        # Earlier turns let a follow-up question make sense, but facts come only from this turn.
        messages += state.get("messages", [])[-4:]
        request = f"Facts:\n{numbered}\n\nQuestion: {state['question']}"
        if state.get("feedback"):
            request += f"\n\nYour last answer was rejected: {state['feedback']}"
        reply = llm.invoke(messages + [HumanMessage(request)])
        return {"answer": reply.content.strip(), "attempts": state["attempts"] + 1, "trace": state["trace"] + ["writer"]}

    def checker(state):
        cited, problem = check_citations(state["answer"], len(state["facts"]))
        return {
            "citations": cited,
            "feedback": problem or "",
            "grounded": problem is None,
            "trace": state["trace"] + ["checker:" + ("ok" if problem is None else "rejected")],
        }

    def give_up(state):
        return {"answer": CANNOT_ANSWER, "citations": [], "grounded": False, "trace": state["trace"] + ["give_up"]}

    def finish(state):
        return {
            "messages": [HumanMessage(state["question"]), AIMessage(state["answer"])],
            "trace": state["trace"] + ["finish"],
        }

    def after_search(state):
        # With nothing retrieved, don't let the model answer from its own memory.
        return "graph_agent" if state.get("hits") else "give_up"

    def after_checker(state):
        if state["grounded"]:
            return "finish"
        return "writer" if state["attempts"] < MAX_ATTEMPTS else "give_up"

    graph = StateGraph(AgentState)
    for name, node in [
        ("planner", planner), ("search_agent", search_agent), ("graph_agent", graph_agent),
        ("writer", writer), ("checker", checker), ("give_up", give_up), ("finish", finish),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "search_agent")
    graph.add_conditional_edges("search_agent", after_search, ["graph_agent", "give_up"])
    graph.add_edge("graph_agent", "writer")
    graph.add_edge("writer", "checker")
    graph.add_conditional_edges("checker", after_checker, ["writer", "finish", "give_up"])
    graph.add_edge("give_up", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


def ask(workflow, question, thread_id="default"):
    state = workflow.invoke(
        {"question": question, "trace": []},
        config={"configurable": {"thread_id": thread_id}},
    )
    return {
        "question": question,
        "answer": state["answer"],
        "facts": state.get("facts", []),
        "citations": state["citations"],
        "grounded": state["grounded"],
        "route": state["route"],
        "trace": state["trace"],
    }
