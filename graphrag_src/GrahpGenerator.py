import asyncio
import logging
from typing import Any, Dict, List, Optional, cast

import igraph as ig
import ollama
import pandas as pd
import polars as pl
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_from_disk
from tqdm import tqdm

try:
    from .EntityRelationExtractor import EntityRelationExtractor
except ImportError:
    from EntityRelationExtractor import EntityRelationExtractor


class GraphGenerator:
    
    CHUNKS_PATH="data/chunks_arrow"
    ENTITY_SCHEMA = pa.schema([
            pa.field("id",             pa.string()),
            pa.field("canonical_name", pa.string()),
            pa.field("label",          pa.string()),
            pa.field("description",    pa.string()),
        ])
    
    def __init__(
        self,
        ner_model_name=None,
        embed_model_name=None,
        description_model_name=None,
        #TODO implement different models
        
        ) -> None:
        self.extractor=EntityRelationExtractor(chunks_path=self.CHUNKS_PATH)
        self.desc_model=description_model_name
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _format_rss_mb() -> float:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    
    
    
    async def _generate_entity_descriptions(
        self,
        output_path: str,
        entity_db: Dict[str, Dict] | None = None,
        entity_db_path: str | None = None,
        max_concurrent: int = 8,
        chunk_batch_size: int = 256,
        flush_every: int = 500,
    ) -> str:
        """
        Returns output_path — entity descriptions are streamed to parquet,
        never fully materialized in RAM.
        """

        if entity_db is None:
            if entity_db_path is None:
                raise ValueError("Either entity_db or entity_db_path must be provided")
            entity_db = pq.read_table(entity_db_path).to_pydict()
        assert entity_db is not None

        model = self.desc_model or "llama3.1:8b-instruct-q4_K_M"

        PROMPT = (
            'You are an information extraction assistant. Based on the following text excerpts, '
            'write a concise 2-3 sentence description of the entity "{entity_name}" (type: {entity_type}).\n'
            'Focus only on factual information directly supported by the excerpts. '
            'Be specific and avoid generic statements.\n\n'
            'Excerpts:\n{excerpts}\n\n'
            'Description of "{entity_name}":'
        )

        client = ollama.AsyncClient()
        semaphore = asyncio.Semaphore(max_concurrent)
        ready: asyncio.Queue[tuple[str, list[str]] | None] = asyncio.Queue()

        # ── Shared write buffer — only the event loop thread touches this ──────
        write_lock = asyncio.Lock()
        buffer: list[dict] = []
        writer: pq.ParquetWriter | None = None

        async def _flush(force: bool = False) -> None:
            nonlocal writer, buffer
            if not buffer:
                return
            async with write_lock:
                if not force and len(buffer) < flush_every:
                    return
                batch = buffer
                buffer = []
            table = pa.table({
                "id":             [r["id"]             for r in batch],
                "canonical_name": [r["canonical_name"] for r in batch],
                "label":          [r["label"]          for r in batch],
                "description":    [r["description"]    for r in batch],
            }, schema=self.ENTITY_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(output_path, self.ENTITY_SCHEMA, compression="zstd")
            writer.write_table(table)

        # ── Producer ──────────────────────────────────────────────────────────
        async def _collect() -> None:
            loop = asyncio.get_running_loop()

            def _scan() -> None:
                context_chunks: dict[str, list[str]] = {eid: [] for eid in entity_db}
                needs_more: set[str] = set(entity_db)
                for batch in self.extractor.load_chunks().iter(chunk_batch_size):
                    if not needs_more:
                        break
                    batch_dict = cast(dict[str, list[Any]], batch)
                    records = [dict(zip(batch_dict.keys(), values)) for values in zip(*batch_dict.values())]
                    for record in records:
                        eid = record["id"]
                        if eid in needs_more:
                            bucket = context_chunks[eid]
                            bucket.append(record["text"])
                            if len(bucket) >= 3:
                                needs_more.discard(eid)
                                loop.call_soon_threadsafe(ready.put_nowait, (eid, list(bucket)))
                for eid in needs_more:
                    loop.call_soon_threadsafe(ready.put_nowait, (eid, list(context_chunks[eid])))
                loop.call_soon_threadsafe(ready.put_nowait, None)

            await asyncio.to_thread(_scan)

        # ── Consumer ──────────────────────────────────────────────────────────
        async def _describe(eid: str, excerpts: list[str], pbar: tqdm) -> None:
            nonlocal buffer
            entity = entity_db[eid]
            if not excerpts:
                description = f"A {entity['label']} named {entity['canonical_name']}."
            else:
                prompt = PROMPT.format(
                    entity_name=entity["canonical_name"],
                    entity_type=entity["label"],
                    excerpts="\n---\n".join(excerpts),
                )
                async with semaphore:
                    response = await client.generate(
                        model=model,
                        prompt=prompt,
                        options={"temperature": 0.1, "num_predict": 150},
                    )
                description = response["response"].strip()

            # Buffer the result — no mutation of entity_db at all
            async with write_lock:
                buffer.append({
                    "id":             eid,
                    "canonical_name": entity["canonical_name"],
                    "label":          entity["label"],
                    "description":    description,
                })
            await _flush()
            pbar.update(1)

        async def _consume(pbar: tqdm) -> None:
            tasks = []
            while True:
                item = await ready.get()
                if item is None:
                    break
                eid, excerpts = item
                tasks.append(asyncio.create_task(_describe(eid, excerpts, pbar)))
            await asyncio.gather(*tasks)
            await _flush(force=True)  # drain the last partial batch
            if writer:
                writer.close()

        with tqdm(total=len(entity_db), desc="Generating descriptions") as pbar:
            await asyncio.gather(_collect(), _consume(pbar))

        return output_path
    
    def _load_parquet_frame(self, path: str, columns: List[str]) -> pl.DataFrame:
        schema = set(pl.read_parquet_schema(path).keys())
        missing_columns = [column for column in columns if column not in schema]
        if missing_columns:
            raise KeyError(f"{path} is missing required columns: {missing_columns}")
        return pl.read_parquet(path, columns=columns)

    def _generate_graph(self, entity_db_path: str, relation_db_path: str) -> ig.Graph:
        self.logger.info(
            "Graph construction start: rss=%.1f MB entity_db=%s relation_db=%s",
            self._format_rss_mb(),
            entity_db_path,
            relation_db_path,
        )
        vertex_frame = self._load_parquet_frame(entity_db_path, ["id", "canonical_name", "label"])
        self.logger.info(
            "Loaded vertex frame: rows=%d cols=%d rss=%.1f MB",
            vertex_frame.height,
            vertex_frame.width,
            self._format_rss_mb(),
        )
        edge_frame = (
            self._load_parquet_frame(relation_db_path, ["head_id", "tail_id", "relation", "score"])
            .rename({"head_id": "source", "tail_id": "target"})
            .group_by(["source", "target"])
            .agg(
                pl.len().alias("weight"),
                pl.col("relation").drop_nulls().unique().sort().alias("relations"),
                pl.col("relation").drop_nulls().n_unique().alias("relation_count"),
                pl.col("score").mean().alias("score_mean"),
            )
            .with_columns(pl.col("relations").list.join("|"))
        )
        self.logger.info(
            "Prepared edge frame: rows=%d cols=%d rss=%.1f MB",
            edge_frame.height,
            edge_frame.width,
            self._format_rss_mb(),
        )

        self.logger.info("Converting frames to pandas for igraph: rss=%.1f MB", self._format_rss_mb())
        vertex_pdf = vertex_frame.rename({"id": "name"}).to_pandas()
        edge_pdf = edge_frame.to_pandas()
        self.logger.info(
            "Converted to pandas: vertex_rows=%d edge_rows=%d rss=%.1f MB",
            len(vertex_pdf),
            len(edge_pdf),
            self._format_rss_mb(),
        )

        self.logger.info("Building igraph Graph.DataFrame: rss=%.1f MB", self._format_rss_mb())
        return ig.Graph.DataFrame(edge_pdf, directed=False, vertices=vertex_pdf, use_vids=False)