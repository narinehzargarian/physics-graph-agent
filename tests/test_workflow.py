"""Tests the agent's control flow with a scripted model and in-memory stores."""
from langchain_core.messages import AIMessage

from agent.workflow import CANNOT_ANSWER, MAX_ATTEMPTS, ask, build_workflow, check_citations, parse_route

PLASMA = {"key": "plasma", "name": "Plasma", "description": "An ionized gas.", "score": 9.0}
TOKAMAK = {"key": "tokamak", "name": "Tokamak", "description": "Confines plasma.", "score": 8.0}


class ScriptedModel:
    """Plays back replies in order, and records what it was sent."""

    def __init__(self, route, *answers):
        self.replies = [route, *answers]
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(self.replies.pop(0))


class FakeSearch:
    def __init__(self, hits):
        self.hits = hits

    def search(self, _text):
        return self.hits


class FakeGraph:
    def __init__(self):
        self.calls = []

    def neighbors(self, key):
        self.calls.append(("neighbors", key))
        return ["Tokamak CONFINES Plasma"]

    def shortest_path(self, a, b):
        self.calls.append(("path", a, b))
        return ["Tokamak CONFINES Plasma"]


def run(model, hits=(PLASMA, TOKAMAK), question="What is plasma?", thread="t"):
    graph = FakeGraph()
    workflow = build_workflow(model, FakeSearch(list(hits)), graph)
    return ask(workflow, question, thread), graph, workflow


def test_search_question_uses_neighbors_and_passes_checker():
    result, graph, _ = run(ScriptedModel('{"route": "search"}', "Plasma is an ionized gas [1]."))
    assert result["grounded"] is True
    assert result["citations"] == [1]
    assert graph.calls == [("neighbors", "plasma")]
    assert result["facts"][-1] == "Tokamak CONFINES Plasma"


def test_connect_question_uses_shortest_path():
    _, graph, _ = run(ScriptedModel('{"route": "connect"}', "They are linked [3]."))
    assert graph.calls == [("path", "plasma", "tokamak")]


def test_uncited_answer_is_rejected_then_fixed_with_feedback():
    model = ScriptedModel('{"route": "search"}', "Plasma is hot.", "Plasma is an ionized gas [1].")
    result, _, _ = run(model)
    assert result["grounded"] is True
    assert result["trace"].count("checker:rejected") == 1
    assert "no citations" in model.calls[-1][-1].content


def test_citation_of_a_fact_that_does_not_exist_is_rejected():
    model = ScriptedModel('{"route": "search"}', "Made up [99].", "Fixed [2].")
    result, _, _ = run(model)
    assert result["citations"] == [2]
    assert "[99]" in model.calls[-1][-1].content


def test_gives_up_after_max_attempts_rather_than_returning_an_unverified_answer():
    model = ScriptedModel('{"route": "search"}', *["No citation."] * MAX_ATTEMPTS)
    result, _, _ = run(model)
    assert result["answer"] == CANNOT_ANSWER
    assert result["grounded"] is False
    assert result["trace"].count("writer") == MAX_ATTEMPTS


def test_off_topic_question_never_reaches_the_writer():
    model = ScriptedModel('{"route": "search"}')
    result, graph, _ = run(model, hits=[], question="Best pizza?")
    assert result["answer"] == CANNOT_ANSWER
    assert len(model.calls) == 1
    assert graph.calls == []


def test_weak_search_matches_count_as_no_match():
    weak = {**PLASMA, "score": 0.5}
    model = ScriptedModel('{"route": "search"}')
    result, _, _ = run(model, hits=[weak])
    assert result["answer"] == CANNOT_ANSWER


def test_conversation_is_remembered_within_a_session_only():
    model = ScriptedModel(
        '{"route": "search"}', "Plasma is ionized [1].",
        '{"route": "search"}', "It is confined [1].",
        '{"route": "search"}', "Other session [1].",
    )
    workflow = build_workflow(model, FakeSearch([PLASMA]), FakeGraph())
    ask(workflow, "What is plasma?", "session-a")
    second = ask(workflow, "How is it confined?", "session-a")
    writer_call = model.calls[3]
    assert any("What is plasma?" in m.content for m in writer_call)
    assert second["trace"][0].startswith("planner")
    assert second["trace"].count("writer") == 1  # The trace covers this question only.

    ask(workflow, "New topic?", "session-b")
    assert not any("What is plasma?" in m.content for m in model.calls[5])


def test_parse_route_handles_messy_model_output():
    assert parse_route('{"route": "connect"}') == "connect"
    assert parse_route('Sure! Here it is:\n```json\n{"route": "connect"}\n```') == "connect"
    assert parse_route("I think search") == "search"
    assert parse_route('{"route": "delete everything"}') == "search"
    assert parse_route("{broken") == "search"


def test_check_citations():
    assert check_citations("A [1] and B [2].", 2) == ([1, 2], None)
    assert check_citations("A.", 2)[1] is not None
    assert check_citations("A [3].", 2)[1] is not None
    assert check_citations("A [0].", 2)[1] is not None
