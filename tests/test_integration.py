"""Runs against real Neo4j, Elasticsearch, and Redis. Skipped unless NEO4J_PASSWORD is set."""
import os
import uuid

import pytest
from langchain_core.messages import AIMessage

from agent import data_pipeline
from agent.workflow import MIN_SEARCH_SCORE, ask, build_workflow

pytestmark = pytest.mark.skipif("NEO4J_PASSWORD" not in os.environ, reason="needs the databases")


@pytest.fixture(scope="module")
def stores():
    from agent.stores import AnswerCache, GraphStore, SearchStore

    graph, search = GraphStore(), SearchStore()
    concepts, relations, _ = data_pipeline.load()
    graph.load(concepts, relations)
    search.load(concepts)
    yield graph, search, AnswerCache(), concepts, relations
    graph.close()


def test_loading_twice_does_not_create_duplicates(stores):
    graph, _, _, concepts, relations = stores
    graph.load(concepts, relations)
    with graph.driver.session() as session:
        assert session.run("MATCH (c:Concept) RETURN count(c) AS n").single()["n"] == len(concepts)
        assert session.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"] == len(relations)


def test_neighbors_state_each_link_in_its_real_direction(stores):
    links = stores[0].neighbors("tokamak")
    assert "Tokamak CONFINES Plasma" in links
    assert "Tokamak USED_FOR Nuclear fusion" in links


def test_shortest_path_crosses_several_links(stores):
    path = stores[0].shortest_path("tokamak", "quantum tunneling")
    assert path == ["Tokamak USED_FOR Nuclear fusion", "Nuclear fusion USES Quantum tunneling"]


def test_unconnected_or_unknown_concepts_return_no_path(stores):
    assert stores[0].shortest_path("tokamak", "not a concept") == []


def test_search_finds_exact_and_misspelled_names(stores):
    search = stores[1]
    assert search.search("What is the Debye length?")[0]["name"] == "Debye length"
    assert search.search("tokomak")[0]["name"] == "Tokamak"


def test_search_finds_a_concept_by_its_description(stores):
    names = [hit["name"] for hit in stores[1].search("ionized gas of free electrons")]
    assert "Plasma" in names


def test_off_topic_questions_score_below_the_search_gate(stores):
    # Regression test: these used to get search hits from fuzzy matches on common words, so only
    # the model's reluctance to cite them kept a wrong answer out.
    for question in ["What is the best pizza topping?", "Who won the 1998 World Cup?", "Which stocks should I buy?"]:
        hits = stores[1].search(question)
        assert [h for h in hits if h["score"] >= MIN_SEARCH_SCORE] == [], question


def test_cache_round_trip_ignores_case_and_spacing(stores):
    cache = stores[2]
    question = f"What is {uuid.uuid4()}?"
    assert cache.get(question) is None
    cache.put(question, {"answer": "x"})
    assert cache.get("  " + question.upper() + " ") == {"answer": "x"}


def test_full_workflow_against_real_stores(stores):
    graph, search, *_ = stores

    class Model:
        replies = ['{"route": "connect"}', "Fusion uses tunneling [5]."]

        def invoke(self, _messages):
            return AIMessage(self.replies.pop(0))

    result = ask(build_workflow(Model(), search, graph), "How does nuclear fusion connect to quantum tunneling?", "it")
    assert result["route"] == "connect"
    assert "Nuclear fusion USES Quantum tunneling" in result["facts"]
