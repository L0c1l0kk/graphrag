import pytest
import polars as pl

from graphrag_src.GrahpGenerator import GraphGenerator


def test_generate_graph_builds_vertices_and_aggregates_edges(tmp_path):
    generator = GraphGenerator.__new__(GraphGenerator)

    entity_db_path = tmp_path / "entities.parquet"
    relation_db_path = tmp_path / "relations.parquet"

    pl.DataFrame(
        {
            "id": ["E0000000", "E0000001"],
            "canonical_name": ["Alice", "Paper"],
            "label": ["person", "work of art"],
            "description": ["A researcher.", "A paper."],
        }
    ).write_parquet(entity_db_path)

    pl.DataFrame(
        {
            "head_id": ["E0000000", "E0000000"],
            "tail_id": ["E0000001", "E0000001"],
            "relation": ["authored", "linked"],
            "score": [0.8, 0.6],
            "head_type": ["person", "person"],
            "tail_type": ["work of art", "work of art"],
        }
    ).write_parquet(relation_db_path)

    graph = generator._generate_graph(str(entity_db_path), str(relation_db_path))

    assert graph.vcount() == 2
    assert graph.vs["name"] == ["E0000000", "E0000001"]
    assert graph.vs.find(name="E0000000")["canonical_name"] == "Alice"
    assert "description" not in graph.vs.attributes()

    edge_id = graph.get_eid("E0000000", "E0000001")
    edge = graph.es[edge_id]
    assert edge["weight"] == 2
    assert edge["relations"] == "authored|linked"
    assert edge["relation_count"] == 2
    assert edge["score_mean"] == pytest.approx(0.7)