import pyarrow as pa
import pytest

from graphrag_src.EntityRelationExtractor import EntityRelationExtractor


class DummyParquetWriter:
    def __init__(self, path, schema, compression=None):
        self.path = path
        self.schema = schema
        self.compression = compression
        self.tables = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def write_table(self, table):
        self.tables.append(table)


class DummyChunks:
    def __init__(self, batches):
        self._batches = batches

    def iter(self, batch_size=256):
        for batch in self._batches:
            yield batch


class DummyModel:
    def inference(self, texts, labels, relations, threshold, relation_threshold, return_relations, flat_ner):
        entities = [
            [
                {"text": "Alice", "label": "person", "score": 0.95},
                {"text": "Bob", "label": "person", "score": 0.85},
            ]
            for _ in texts
        ]
        relations = [
            [
                {
                    "head": "Alice",
                    "head_type": "person",
                    "relation": "authored",
                    "tail": "Paper",
                    "tail_type": "work of art",
                    "score": 0.80,
                }
            ]
            for _ in texts
        ]
        return entities, relations


def test_chunk_text_splits_with_overlap():
    extractor = EntityRelationExtractor(ner_model=DummyModel(), embed_model=None, chunk_size=5, chunk_overlap=2)
    chunks = extractor._chunk_text("one two three four five six seven")

    assert chunks == ["one two three four five", "four five six seven", "seven"]


def test_format_relations_falls_back_to_head_label():
    extractor = EntityRelationExtractor(ner_model=DummyModel(), embed_model=None)
    relations = [
        {
            "head": "Alice",
            "head_label": "person",
            "relation": "authored",
            "tail": "Paper",
            "tail_label": "work of art",
            "score": 0.8,
        },
        {
            "head": "Bob",
            "head_type": "person",
            "relation": "collaborated",
            "tail": "Alice",
            "tail_type": "person",
            "score": 0.75,
        },
    ]

    formatted = extractor._format_relations(relations)

    assert formatted[0]["head_label"] == "person"
    assert formatted[1]["head_label"] == "person"
    assert formatted[1]["tail_label"] == "person"


def test_process_batch_outputs_entities_and_relations():
    extractor = EntityRelationExtractor(ner_model=DummyModel(), embed_model=None)
    batch = [{"id": 0, "doc_id": "doc1", "text": "Alice wrote a paper."}]

    results = extractor._process_batch(batch)

    assert len(results) == 1
    output = results[0]
    assert output["chunk_id"] == 0
    assert output["doc_id"] == "doc1"
    assert output["entities"][0]["text"] == "Alice"
    assert output["relations"][0]["relation"] == "authored"


def test_generate_runs_pipeline_and_writes_parquet(monkeypatch):
    monkeypatch.setattr(EntityRelationExtractor, "_generate_chunks", lambda self, path, out: None)

    batches = [
        {"id": [0], "doc_id": ["doc1"], "text": ["Alice wrote a paper."]},
        {"id": [1], "doc_id": ["doc2"], "text": ["Bob reviewed the paper."]},
    ]
    monkeypatch.setattr("graphrag_src.EntityRelationExtractor.load_from_disk", lambda path: DummyChunks(batches))
    monkeypatch.setattr("graphrag_src.EntityRelationExtractor.pq.ParquetWriter", DummyParquetWriter)

    def fake_merge(self, df):
        return ({"E0000000": {"id": "E0000000", "label": "person", "canonical_name": "Alice", "surface_forms": ["Alice"], "source_chunks": []}}, {"Alice": "E0000000"})

    monkeypatch.setattr(EntityRelationExtractor, "_get_unique_entity_strings", lambda self, path: None)
    monkeypatch.setattr(EntityRelationExtractor, "_merge_entities", fake_merge)

    extractor = EntityRelationExtractor(ner_model=DummyModel(), embed_model=None)
    entity_db, entity_map = extractor.generate("dummy_path")

    assert entity_db == {"E0000000": {"id": "E0000000", "label": "person", "canonical_name": "Alice", "surface_forms": ["Alice"], "source_chunks": []}}
    assert entity_map == {"Alice": "E0000000"}
