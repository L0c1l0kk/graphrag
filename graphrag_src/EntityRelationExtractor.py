from collections import defaultdict
import logging
import os

from datasets import IterableDataset, load_dataset, load_from_disk, Dataset, DatasetDict
from typing import List, Dict, Optional, Any, cast
from gliner import GLiNER
from gliner.model import BaseGLiNER
from FlagEmbedding import BGEM3FlagModel
import torch
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from tqdm import tqdm
import faiss
import numpy as np


class EntityRelationExtractor:
    """
    Joint entity and relation extraction via GLiNER-Relex.

    this was specifically built for GLiNER-Relex could be extended to use different models.
    """

    DEFAULT_ENTITY_LABELS: List[str] = [
        "person", "organization", "location", "event",
        "concept", "product", "work of art", "law", "disease", "date",
    ]
    # Plain-string relation types — GLiNER-Relex has no fixed taxonomy.
    DEFAULT_RELATION_LABELS: List[str] = [
        "developed", "discovered", "founded", "caused", "treated by",
        "located in", "part of", "authored", "affiliated with",
        "instance of", "preceded by", "influenced",
    ]
    
    ENTITIES_PATH="data/entities"
    RELATIONS_PATH="data/relations"
    
    _ENTITY_SCHEMA = pa.schema([
        ("chunk_id", pa.string()),
        ("text",     pa.string()),
        ("label",    pa.string()),
        ])

    _RELATION_SCHEMA = pa.schema([
        ("chunk_id",  pa.string()),
        ("head",      pa.string()),
        ("head_type", pa.string()),
        ("relation",  pa.string()),
        ("tail",      pa.string()),
        ("tail_type", pa.string()),
        ])

    def __init__(
        self,
        ner_model: Optional[BaseGLiNER] = None,
        embed_model: Any = None,
        chunk_size: int = 100,
        chunk_overlap: int = 20,
        chunks_path: str = "data/chunks_arrow",
        entity_labels: Optional[List[str]] = None,
        relation_labels: Optional[List[str]] = None,
        threshold: float = 0.4,
        relation_threshold: float = 0.5,
        max_entities_per_chunk: int = 30,
    ):
        """
        Args:
            model: Pre-loaded GLiNER-Relex model.  Defaults to
                knowledgator/gliner-relex-large-v1.0 from HF Hub.
            chunk_size: Words per chunk when splitting raw text.
            chunk_overlap: Overlap words between consecutive chunks.
            entity_labels: Zero-shot entity type labels.
            relation_labels: Zero-shot relation type labels (plain English).
            threshold: Entity extraction confidence cutoff (0.4–0.5
                recommended; lower floods the graph with noise entities).
            relation_threshold: Relation extraction confidence cutoff.
            max_entities_per_chunk: Hard cap on entities kept per chunk
                before RE to bound the O(n²) pair-enumeration cost.
                GLiNER returns entities sorted by confidence, so the cap
                drops the lowest-confidence spans first.
        """
        self.ner_model = ner_model
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks_path=chunks_path
        self.entity_labels = entity_labels or self.DEFAULT_ENTITY_LABELS
        self.relation_labels = relation_labels or self.DEFAULT_RELATION_LABELS
        self.threshold = threshold
        self.relation_threshold = relation_threshold
        self.max_entities_per_chunk = max_entities_per_chunk
        self._n_chunks=0
        # module-level logger
        self.logger = logging.getLogger(__name__)
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger.debug("Initialized EntityRelationExtractor: chunk_size=%s, chunk_overlap=%s, chunks_path=%s",
                          self.chunk_size, self.chunk_overlap, self.chunks_path)

    #TODO if needed a config file, and matching behavior to modify default settings
    # Chunking
    

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks, start = [], 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _chunk_generator(self, dataset: IterableDataset):
        chunk_id = 0
        for doc in tqdm(dataset,desc="Chunking"):
            for chunk in self._chunk_text(doc["text"]):
                yield {"id": str(chunk_id), "text": chunk}
                chunk_id += 1
        self._n_chunks = chunk_id+1

    def _generate_chunks(self, path: str, output_dir:str) -> None:
        """
        Generates chunks and saves them to disk into output_dir, load with datasets.load_from_disk()
        """
        if path=="wiki_dpr":
            self.logger.info("Dataset %s already chunked; skipping generation", path)
            return
        else:
            dataset=load_dataset(path, split="train", streaming=True)
        Dataset.from_generator(
            lambda: self._chunk_generator(dataset)
        ).save_to_disk(output_dir)
        
        
        if self.ner_model is None:
            self.logger.info("Loading default GLiNER model from hub")
            self.ner_model = GLiNER.from_pretrained("knowledgator/gliner-relex-large-v1.0")
        device= "cuda" if torch.cuda.is_available() else "cpu"
        self.ner_model.to(device)
        self.logger.info("Using device: %s", device)
        
        
        del self.ner_model
        self.logger.info("Finished chunk generation and released ner_model")

    def load_chunks(self) -> (Dataset | IterableDataset):
        self.logger.info("Loading chunks from %s", self.chunks_path)
        if self.chunks_path=="wiki_dpr":
            ds = load_dataset("facebook/wiki_dpr", name="psgs_w100.nq.no_index.no_embeddings", split="train")
            ds=ds.select(range(100)) # for testing
            self._n_chunks=len(ds)
        else:
            ds = load_from_disk(self.chunks_path)
            self._n_chunks=len(ds)
            #TODO make this work for IterableDataset
        self.logger.info("Loaded chunks: %s", getattr(ds, '__class__', type(ds)))
        return ds
    # Extraction
    

    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """
        Run joint NER + RE on a list of chunk dicts in a single forward pass.

        Input dicts must contain: id (int), text (str).

        Returns a list of output dicts with:
          chunk_id  – echoed from input id
          text      – passthrough
          entities  – list of {text, label, score}
          relations – list of {head, head_label, relation, tail, tail_label, score}
        """
        texts = [item["text"] for item in batch]
        self.logger.debug("Processing batch of %d texts", len(texts))

        entities_all, relations_all = self.ner_model.inference(
            texts=texts,
            labels=self.entity_labels,
            relations=self.relation_labels,
            threshold=self.threshold,
            relation_threshold=self.relation_threshold,
            return_relations=True,
            flat_ner=False,
        )

        results = []
        for item, entities, relations in zip(batch, entities_all, relations_all):
            # Cap to bound the quadratic RE pair-enumeration cost.
            entities = entities[: self.max_entities_per_chunk]

            results.append({
                "chunk_id":  item["id"],
                "text":      item["text"],
                "entities":  self._format_entities(entities),
                "relations": self._format_relations(relations),
            })
        return results


    # Formatting helpers


    @staticmethod
    def _format_entities(entities: List[Dict]) -> List[Dict]:
        return [
            {"text": e["text"], "label": e["label"], "score": round(e["score"], 4)}
            for e in entities
        ]

    @staticmethod
    def _format_relations(relations: List[Dict]) -> List[Dict]:
        """
        Normalise GLiNER-Relex relation dicts to a consistent schema.

        GLiNER-Relex v1.0 relation dict uses:
          head (str), head_type (str), relation (str),
          tail (str), tail_type (str), score (float)

        We expose head_label / tail_label for consistency with entity dicts
        and fall back to alternative key names across library releases.
        """
        return [
            {
                "head":       r["head"],
                "head_label": r.get("head_type") or r.get("head_label", ""),
                "relation":   r["relation"],
                "tail":       r["tail"],
                "tail_label": r.get("tail_type") or r.get("tail_label", ""),
                "score":      round(r["score"], 4),
            }
            for r in relations
        ]


# Deduplication and normalization using polars

    def _rows_to_table(self, rows: list[dict], schema: pa.Schema) -> pa.Table:
        def _coerce_scalar(value: Any) -> Any:
            if isinstance(value, dict):
                for key in ("text", "label", "name", "value", "id"):
                    nested = value.get(key)
                    if nested is not None:
                        return nested
                return str(value)
            return value

        return pa.table(
            {
                field.name: pa.array(
                    [_coerce_scalar(row[field.name]) for row in rows],
                    type=field.type,
                )
                for field in schema
            },
            schema=schema,
        )

    def _get_unique_entity_strings(self, entities_path: str) -> pl.DataFrame:
        return (
            pl.scan_parquet(entities_path)
            .with_columns(
                pl.col("text").str.strip_chars(),
                pl.col("text").str.strip_chars().str.to_lowercase().alias("text_norm"),
            )
            .group_by(["text_norm", "text", "label"])
            .agg(
                pl.len().alias("count"),
                pl.col("chunk_id").alias("chunk_ids"),
            )
            .group_by("text_norm")
            .agg(
                pl.col("text").sort_by("count", descending=True).first().alias("canonical_text"),
                pl.col("label").sort_by("count", descending=True).first().alias("label"),
                pl.col("text").alias("surface_forms"),          # List of all surface forms
                pl.col("chunk_ids").explode().alias("chunk_ids")  # Flattened list of all chunk_ids
            )
            .collect()
        )
    
    def _merge_entities(self, entities_df: pl.DataFrame, sim_threshold: float = 0.9) -> tuple[Dict[str, Dict], Dict[str, str]]:
        entity_map:     Dict[str, str]  = {}
        entity_db:      Dict[str, Dict] = {}
        entity_counter: int             = 0
        
        if self.embed_model is None:
            self.logger.info("Initializing embedding model for entity merge")
            self.embed_model=BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

        for label in entities_df["label"].unique().to_list():
            self.logger.debug("Merging entities for label=%s", label)
            rows = entities_df.filter(pl.col("label") == label).to_dicts()
            if not rows:
                continue

            embeddings = self.embed_model.encode(
                [r["canonical_text"] for r in rows],
                batch_size=64,
                max_length=128,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )["dense_vecs"].astype(np.float32) 

            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1]) 
            index.add(embeddings)

            k = min(32, len(rows))
            scores, indices = index.search(embeddings, k)

            # union-find
            uf = list(range(len(rows)))

            def find(x: int) -> int:
                while uf[x] != x:
                    uf[x] = uf[uf[x]]   # path compression
                    x = uf[x]
                return x

            for i, (row_scores, row_indices) in enumerate(zip(scores, indices)):
                for score, j in zip(row_scores, row_indices):
                    if i != j and score >= sim_threshold:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            uf[ri] = rj

            clusters: Dict[int, List[Dict]] = defaultdict(list)
            for idx, row in enumerate(rows):
                clusters[find(idx)].append(row)

            for members in clusters.values():
                canonical_name = max(members, key=lambda m: len(m["canonical_text"]))["canonical_text"]
                eid = f"E{entity_counter:07d}"
                entity_counter += 1
                surface_forms = [f for m in members for f in m["surface_forms"]]
                entity_db[eid] = {
                    "id":             eid,
                    "label":          label,
                    "canonical_name": canonical_name,
                    "source_chunks":  [chunk_id for m in members for chunk_id in m["chunk_ids"]],
                }
                for form in surface_forms:
                    entity_map[form] = eid

        del self.embed_model
        
        entity_db=[{**entity} for entity in entity_db.values()]
        self.logger.info("Finished merging entities; created %d entity ids", len(entity_db))
        
        return entity_db, entity_map
    
    def _save_inplace(self, df: pl.LazyFrame | pl.DataFrame, path: str) -> None:
        """Save a Polars DataFrame to a Parquet file, overwriting the original."""
        temp_path = f"{path}.tmp"
        try:
            if(df is pl.LazyFrame):
                df.sink_parquet(temp_path)
            elif (df is pl.DataFrame):
                df.write_parquet(temp_path)
            os.replace(temp_path, path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
    
    def _index_relations(self, relations_path: str, entity_map: Dict[str, str]) -> pl.LazyFrame:
        lazy_df = (
            pl.scan_parquet(relations_path)
            .with_columns(
                pl.col("head").str.strip_chars(),
                pl.col("tail").str.strip_chars(),
            )
            .with_columns(
                pl.col("head").replace_strict(entity_map, default=None).alias("head_id"),
                pl.col("tail").replace_strict(entity_map, default=None).alias("tail_id"),
            )
            .filter(pl.col("head_id").is_not_null() & pl.col("tail_id").is_not_null())
        )
        return lazy_df
    
    # generation logic
    def generate(self, dataset_path: str = "wiki_dpr", batch_size: int = 256):
        """
        Run extraction over a dataset and produce deduplicated entity and relation databases.

        This method performs a full extraction pipeline over documents available at
        `dataset_path` (defaults to the compressed DPR Wikipedia splits). The pipeline
        includes:
          1. Chunking the input documents and saving the chunk dataset to
             `self.CHUNKS_PATH` via ``_generate_chunks``.
          2. Running joint NER + RE inference on each chunk in batches and
             writing the raw entity and relation rows to parquet files at
             `self.ENTITIES_PATH` and `self.RELATIONS_PATH` respectively. The parquet
             files use the schemas defined by ``self._ENTITY_SCHEMA`` and
             ``self._RELATION_SCHEMA``.
          3. Post-processing the collected entity strings to compute canonical
             surface forms and cluster similar entity mentions using the
             configured `self.embed_model`. Finally, produces an entity database
             and a surface-form -> entity id mapping.

        Parameters
        ----------
        dataset_path : str, optional
            Path or identifier for the source dataset to extract from. When
            set to the special value ``"wiki_dpr"`` the method will
            stream the DPR Wikipedia passages dataset. Otherwise the string is
            passed to ``datasets.load_dataset``/streaming to obtain the input
            documents. Each input document must be a mapping with at least
            the keys ``"id"`` and ``"text"``.

        Side effects
        -----------
        - Writes chunk data to ``self.CHUNKS_PATH`` (callable by
          ``datasets.load_from_disk``).
        - Appends raw entity rows (one row per detected entity span) to the
          parquet file at ``self.ENTITIES_PATH`` using ``self._ENTITY_SCHEMA``.
        - Appends raw relation rows (one row per detected relation) to the
          parquet file at ``self.RELATIONS_PATH`` using ``self._RELATION_SCHEMA``.

        Notes on extraction
        -------------------
        - Extraction runs in streaming/batched mode; intermediate rows are
          flushed periodically to disk (controlled by the local ``FLUSH_AT``
          constant) to avoid excessive memory use.
        - The method uses ``self._process_batch`` to perform joint NER+RE and
          expects that function to return results with keys ``"entities"`` and
          ``"relations"``. The raw rows written to parquet mirror the
          dictionaries produced by that processing step.

        Returns
        -------

        - ``entity_db`` : Dict[str, Dict]
            Mapping from generated entity id (string) to a dictionary with
            the following keys:
                - ``id``: entity identifier (e.g. "E0000001")
                - ``label``: entity type label (string)
                - ``canonical_name``: chosen canonical surface form (string)
                - ``surface_forms``: list of observed surface form strings
                - ``source_chunks``: list of chunk ids where the entity was seen

        Example
        -------
        >>> entity_db = extractor.generate("wiki_dpr")
        >>> entity_db["E0000001"]["canonical_name"]
        'Barack Obama'

        """
        if dataset_path == "wiki_dpr":
            self.chunks_path = "wiki_dpr"
        
        self.logger.info("Starting extraction pipeline for dataset: %s", dataset_path)
        self._generate_chunks(dataset_path, self.chunks_path)
        chunks = self.load_chunks()

        entity_rows:   list[dict] = []
        relation_rows: list[dict] = []
        FLUSH_AT = 500_000
        self.logger.info("Beginning chunk extraction loop with FLUSH_AT=%d", FLUSH_AT)
    
        os.makedirs(os.path.dirname(self.ENTITIES_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(self.RELATIONS_PATH), exist_ok=True)
        
        with (
            pq.ParquetWriter(self.ENTITIES_PATH,  self._ENTITY_SCHEMA,   compression="zstd") as ew,
            pq.ParquetWriter(self.RELATIONS_PATH, self._RELATION_SCHEMA, compression="zstd") as rw,
        ):
            def flush():
                if entity_rows:
                    ew.write_table(self._rows_to_table(entity_rows, self._ENTITY_SCHEMA))
                    self.logger.info("Flushing %d entity rows to %s", len(entity_rows), self.ENTITIES_PATH)
                    entity_rows.clear()
                if relation_rows:
                    rw.write_table(self._rows_to_table(relation_rows, self._RELATION_SCHEMA))
                    self.logger.info("Flushing %d relation rows to %s", len(relation_rows), self.RELATIONS_PATH)
                    relation_rows.clear()

            total_batches = (self._n_chunks + batch_size - 1) // batch_size
            with tqdm(total=total_batches, desc="Extracting") as pbar:
                for batch in chunks.iter(batch_size=batch_size):
                    batch_dict = cast(Dict[str, List[Any]], batch)
                    records = [dict(zip(batch_dict.keys(), vals)) for vals in zip(*batch_dict.values())]
                    for result in self._process_batch(records):
                        cid = result["chunk_id"]
                        entity_rows.extend(
                            {"chunk_id": cid, "text": e["text"], "label": e["label"]}
                            for e in result["entities"]
                        )
                        relation_rows.extend(
                            {"chunk_id": cid, "head": r["head"],
                            "head_type": r.get("head_type") or r.get("head_label", ""), "relation": r["relation"],
                            "tail": r["tail"], "tail_type": r.get("tail_type") or r.get("tail_label", "")}
                            for r in result["relations"]
                        )
                    pbar.update(1)
                    if len(entity_rows) >= FLUSH_AT or len(relation_rows) >= FLUSH_AT:
                        flush()

            flush()

        unique = self._get_unique_entity_strings(self.ENTITIES_PATH)
        entity_db, entity_map = self._merge_entities(unique)
        relations_db = self._index_relations(self.RELATIONS_PATH, entity_map)
        
        self._save_inplace(relations_db, self.RELATIONS_PATH)
        self._save_inplace(pl.from_dict(entity_db), self.ENTITIES_PATH)
        self.logger.info("Extraction pipeline completed successfully")
        
        return entity_db
        
        
        
        