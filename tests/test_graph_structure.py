"""Graph compiles with the expected topology — no LLM/network required."""

from tbcg.graph.build import build_graph


def test_graph_compiles_with_hitl():
    graph = build_graph(human_in_the_loop=True)
    nodes = set(graph.get_graph().nodes)
    for expected in [
        "engagement",
        "research",
        "finance",
        "strategy",
        "risk",
        "critic",
        "revise",
        "human_review",
        "report",
    ]:
        assert expected in nodes


def test_graph_compiles_without_hitl():
    graph = build_graph(human_in_the_loop=False)
    nodes = set(graph.get_graph().nodes)
    assert "human_review" not in nodes
    assert "report" in nodes
