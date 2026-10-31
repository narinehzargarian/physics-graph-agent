"""Runs labeled questions through the real system with a real local model and prints the scores."""
import json
import os
import sys
import time

from langchain_ollama import ChatOllama

from agent import data_pipeline
from agent.stores import GraphStore, SearchStore
from agent.workflow import CANNOT_ANSWER, ask, build_workflow

# (question, a concept that must be retrieved, expected route).
ANSWERABLE = [
    ("What is the Debye length?", "Debye length", "search"),
    ("What does the Particle-in-Cell method do?", "Particle-in-Cell method", "search"),
    ("Explain quantum tunneling.", "Quantum tunneling", "search"),
    ("What is a tokamak used for?", "Tokamak", "search"),
    ("What does Noether's theorem say?", "Noether's theorem", "search"),
    ("What is time dilation?", "Time dilation", "search"),
    ("What is the photoelectric effect?", "Photoelectric effect", "search"),
    ("What is entropy?", "Entropy", "search"),
    ("How does nuclear fusion connect to quantum tunneling?", "Nuclear fusion", "connect"),
    ("How are the tokamak and plasma related?", "Tokamak", "connect"),
    ("How is a black hole connected to general relativity?", "General relativity", "connect"),
    ("What is the relationship between the Debye length and the Boltzmann distribution?", "Debye length", "connect"),
]
UNANSWERABLE = [
    "What is the best pizza topping?",
    "Who won the 1998 World Cup?",
    "Which stocks should I buy?",
]


def main():
    model = os.environ.get("OLLAMA_MODEL", "gemma2:2b")
    graph, search = GraphStore(), SearchStore()
    concepts, relations, report = data_pipeline.load()
    graph.load(concepts, relations)
    search.load(concepts)
    llm = ChatOllama(model=model, base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"), temperature=0)
    workflow = build_workflow(llm, search, graph)
    print(json.dumps({"model": model, "data_cleaning": report}))

    retrieved = routed = grounded = retried = 0
    for i, (question, expected, route) in enumerate(ANSWERABLE):
        start = time.monotonic()
        result = ask(workflow, question, thread_id=f"eval-{i}")
        found = any(expected.casefold() in fact.casefold() for fact in result["facts"])
        retrieved += found
        routed += result["route"] == route
        grounded += result["grounded"]
        retried += sum(step == "checker:rejected" for step in result["trace"]) > 0
        print(json.dumps({"question": question, "answer": result["answer"], "retrieved": found,
                          "route": result["route"], "grounded": result["grounded"],
                          "seconds": round(time.monotonic() - start, 1)}))

    refused = 0
    for i, question in enumerate(UNANSWERABLE):
        result = ask(workflow, question, thread_id=f"eval-off-{i}")
        refused += result["answer"] == CANNOT_ANSWER
        print(json.dumps({"question": question, "answer": result["answer"], "trace": result["trace"]}))

    total = len(ANSWERABLE)
    summary = {
        "retrieval_hit": f"{retrieved}/{total}",
        "route_correct": f"{routed}/{total}",
        "grounded_answers": f"{grounded}/{total}",
        "needed_retry": f"{retried}/{total}",
        "off_topic_refused": f"{refused}/{len(UNANSWERABLE)}",
    }
    print("SUMMARY " + json.dumps(summary))
    graph.close()
    return 0 if retrieved == total else 1


if __name__ == "__main__":
    sys.exit(main())
