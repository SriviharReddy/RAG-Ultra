import pytest
from langgraph.graph import END
from my_agent.agent import route_relevance, route_groundedness


def test_route_relevance_retrieve():
    state = {"route_decision": "retrieve"}
    assert route_relevance(state) == "retrieve"


def test_route_relevance_assemble():
    state = {"route_decision": "assemble"}
    assert route_relevance(state) == "assemble"


def test_route_relevance_default():
    # When route_decision is missing from state
    state = {}
    assert route_relevance(state) == "assemble"


def test_route_relevance_other_decisions():
    # Any other decision falls back to assemble
    for decision in ["end", "generate", "verify", "unknown"]:
        state = {"route_decision": decision}
        assert route_relevance(state) == "assemble"


def test_route_groundedness_retrieve():
    state = {"route_decision": "retrieve"}
    assert route_groundedness(state) == "retrieve"


def test_route_groundedness_end():
    state = {"route_decision": "end"}
    assert route_groundedness(state) == END
    assert route_groundedness(state) == "__end__"


def test_route_groundedness_default():
    # When route_decision is missing from state
    state = {}
    assert route_groundedness(state) == END
    assert route_groundedness(state) == "__end__"


def test_route_groundedness_other_decisions():
    # Any non-retrieve decision routes to END
    for decision in ["assemble", "generate", "verify", "unknown"]:
        state = {"route_decision": decision}
        assert route_groundedness(state) == END
