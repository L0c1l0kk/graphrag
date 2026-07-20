import asyncio
import json
import logging
import os
from pathlib import Path
import random
import re
from abc import ABC, abstractmethod

import duckdb
import numpy as np
import ollama
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import polars as pl


class DescriptionGenerator(ABC):
    def __init__(self, input_path, entity_db_path, output_path, logger: logging.Logger, flush_every: int = 1000, max_concurrent: int = 8, model: str | None = None):
        self.logger = logger
        self.flush_every = flush_every
        self.output_path = output_path
        self.input_path = input_path
        self.entity_db_path = entity_db_path
        self.write_lock = asyncio.Lock()
        self.buffer: list[dict] = []
        self.writer: pq.ParquetWriter | None = None
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = ollama.AsyncClient()
        # Queue items are fully self-contained records (id + whatever fields
        # _describe needs), not just (id, excerpts) — this lets _scan stream
        # everything a subclass needs without any global in-memory DB lookup.
        self.ready: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=6000)
        self.model = model or "llama3.1:8b-instruct-q4_K_M"
        # Row count for the progress bar, set cheaply from Parquet metadata
        # (footer only — doesn't require reading/loading the actual rows).
        self._total: int | None = None
        self._index_path: str | None = None

    _SCHEMA = None
    _PROMPT = None

    async def _flush(self, force: bool = False) -> None:
        if not self.buffer:
            return
        async with self.write_lock:
            if not force and len(self.buffer) < self.flush_every:
                return
            batch = self.buffer
            buffer_len = len(batch)
            self.buffer = []
        table= pa.table({f.name : [r[f.name] for r in batch] for f in self._SCHEMA}, schema=self._SCHEMA)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.output_path, self._SCHEMA, compression="zstd")
        self.writer.write_table(table)
        self.logger.debug(f"Flush complete | Records: {buffer_len}")

    @abstractmethod
    def _scan(self, loop: asyncio.AbstractEventLoop) -> None:
        """Run in a worker thread and stream entity_db in small batches.

        Each batch performs one indexed lookup into chunk_db for the needed
        chunk ids, keeping the work bounded by entity_batch_size.
        """

    async def _collect(self) -> None:
        self.logger.info("Starting collect phase")
        loop = asyncio.get_running_loop()
        await asyncio.to_thread(self._scan, loop)
        self.logger.info(f"Collect phase complete | Queue size: {self.ready.qsize()}")

    @abstractmethod
    async def _describe(self, record: dict, pbar: tqdm) -> None:
        pass

    async def _worker(self, pbar: tqdm) -> None:
        while True:
            item = await self.ready.get()
            if item is None:
                await self.ready.put(None)  # re-broadcast so sibling workers also exit
                break

            item_chunks = len(item.get("source_chunks") or [])

            await self._describe(item, pbar)

    async def _consume(self, pbar: tqdm) -> None:
        self.logger.info(f"Starting consume phase with {self.max_concurrent} workers")
        workers = [asyncio.create_task(self._worker(pbar)) for _ in range(self.max_concurrent)]
        await asyncio.gather(*workers)
        self.logger.info(f"All workers finished | Buffer size: {len(self.buffer)}")
        await self._flush(force=True)
        if self.writer:
            self.writer.close()
        self.logger.info("Consume phase complete")

    async def generate_descriptions(self) -> str:
        if self._total is None:
            raise ValueError("Row count (self._total) not set — did the subclass set it before calling super()?")
        self.logger.info(f"Starting description generation for {self._total} items")
        with tqdm(total=self._total, desc="Generating descriptions") as pbar:
            await asyncio.gather(self._collect(), self._consume(pbar))
        self.logger.info(f"Description generation complete | Output: {self.output_path}")
        return self.output_path
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Cheap ~4-chars/token approximation — a safe overestimate for most
        # tokenizers on English text. Swap for a real tokenizer if you need
        # this to be exact rather than a conservative upper bound.
        return max(1, len(text) // 4)
    
    def _index_is_valid(self) -> bool:
        """Return True when chunk_index_path contains a queryable chunks table.

        This rejects stub files created by duckdb.connect() before a real
        build has finished.
        """
        try:
            if self._index_path is None:
                raise ValueError("Index path not set")
            con = duckdb.connect(self._index_path, read_only=True)
            try:
                con.execute("SELECT 1 FROM chunks LIMIT 1")
                return True
            finally:
                con.close()
        except duckdb.Error:
            return False


class EntityDescriptionGenerator(DescriptionGenerator):
    _SCHEMA = pa.schema([
            pa.field("id",             pa.string()),
            pa.field("label",          pa.string()),
            pa.field("canonical_name", pa.string()),
            pa.field("source_chunks",  pa.list_(pa.string())),
            pa.field("description",    pa.string()),
        ])
    _PROMPT = (
            'You are an information extraction assistant. Based on the following text excerpts, '
            'write a concise 2-3 sentence description of the entity "{entity_name}" (type: {entity_type}).\n'
            'Focus only on factual information directly supported by the excerpts. '
            'Your response must only contain the description text, without any additional commentary or formatting.'
            'Be specific and avoid generic statements.\n\n'
            'Excerpts:\n{excerpts}\n\n'
            'Description of "{entity_name}":'
        )

    # --- evidence-selection knobs -------------------------------------------------
    _TOKEN_BUDGET = 4000                  # excerpts-block token budget per prompt
    _MMR_LAMBDA = 0.7                     # 1.0 = pure relevance, 0.0 = pure diversity
    _MAX_CANDIDATES_FOR_MMR = 200         # cap MMR's O(k*n) cost for hub entities
    _HARD_CAP_BEFORE_CLUSTERING = 5000    # Amount of source_chunks allowed at once so no OOM happens on 16gb ram

    def __init__(self, *args, chunk_index_path: str | None = None, entity_batch_size: int = 500,
                 input_format: str | None = None, chunk_id_column: str = "id",
                 text_column: str = "text", **kwargs):
        super().__init__(*args, **kwargs)
        self._index_path = chunk_index_path or f"{self.input_path}.chunks.duckdb"
        self.entity_batch_size = entity_batch_size
        # "parquet" (default, unchanged behavior) or "arrow" — a single
        # memory-mapped .arrow file or a directory of HF-datasets-style
        # .arrow shards (e.g. wiki_dpr). Pass explicitly to skip sniffing.
        self.input_format = input_format or self._detect_input_format(self.input_path)
        self.chunk_id_column = chunk_id_column
        self.text_column = text_column

    async def _describe(self, record: dict, pbar: tqdm) -> None:
        excerpts = record["excerpts"]
        if not excerpts:
            description = f"A {record['label']} named {record['canonical_name']}."
        else:
            prompt = self._PROMPT.format(
                entity_name=record["canonical_name"],
                entity_type=record["label"],
                excerpts="\n---\n".join(excerpts),
            )
            async with self.semaphore:
                response = await self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={"temperature": 0.1, "num_predict": 150},
                )
            description = response["response"].strip()

        # Buffer the result — no mutation of entity_db at all
        async with self.write_lock:
            self.buffer.append({
                "id":             record["id"],
                "canonical_name": record["canonical_name"],
                "label":          record["label"],
                "description":    description,
                "source_chunks":  record["source_chunks"],
            })
            buffer_len = len(self.buffer)
        if buffer_len % 50 == 0:  # Log every 50 buffered items
            self.logger.debug(f"Record {record['id'][:20]}... buffered | Buffer: {buffer_len}")
        await self._flush()
        pbar.update(1)

    # -------------------------------------------------------------------------
    # Evidence selection (MMR under a token budget, over an entity's mentions)
    # tf-idf MMR

    @staticmethod
    def _tfidf(texts: list[str]):
        # max_features keeps the matrix bounded even if some entity's mentions
        # span a huge and varied vocabulary (hub entities). Rows are L2-normalized
        # by default, so row dot-products below are already cosine similarities.
        vectorizer = TfidfVectorizer(max_features=20_000, stop_words="english")
        return vectorizer.fit_transform(texts)

    @staticmethod
    def _tfidf_cluster_representatives(tfidf, k: int, seed: int = 0) -> list[int]:
        """Cluster candidates in TF-IDF space and return one real representative per cluster.

        The representative is the candidate closest to each cluster center.
        """
        n = tfidf.shape[0]
        k = min(k, n)
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=min(256, n))
        labels = km.fit_predict(tfidf)
        centers = km.cluster_centers_  # dense (k, vocab)

        representatives = []
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            sims = np.asarray(tfidf[members] @ centers[c]).ravel()
            representatives.append(int(members[int(sims.argmax())]))
        return representatives

    def _cluster_downsample(self, candidates: list[dict], k: int) -> list[dict]:
        """Downsample very large candidate sets with TF-IDF clustering.

        This keeps one representative per cluster so MMR stays bounded on hub
        entities with thousands of near-duplicate mentions.
        """
        if len(candidates) > self._HARD_CAP_BEFORE_CLUSTERING:
            rng = random.Random(0)
            candidates = rng.sample(candidates, self._HARD_CAP_BEFORE_CLUSTERING)

        tfidf = self._tfidf([c["text"] for c in candidates])
        rep_indices = self._tfidf_cluster_representatives(tfidf, k)
        return [candidates[i] for i in rep_indices]

    def _select_excerpts(self, candidates: list[dict]) -> list[str]:
        """Select excerpts with MMR under a token budget in TF-IDF space.

        Relevance is similarity to the centroid of the entity's own texts,
        and the score then diversifies away from already selected excerpts.
        """
        if not candidates:
            return []
        if len(candidates) == 1:
            return [candidates[0]["text"]]

        if len(candidates) > self._MAX_CANDIDATES_FOR_MMR:
            candidates = self._cluster_downsample(candidates, self._MAX_CANDIDATES_FOR_MMR)

        tfidf = self._tfidf([c["text"] for c in candidates])
        # Bounded to <= _MAX_CANDIDATES_FOR_MMR rows, so a dense pairwise
        # similarity matrix is cheap and lets the MMR loop below run as plain
        # numpy indexing instead of resparsifying every iteration.
        sim_matrix = np.asarray((tfidf @ tfidf.T).todense())

        centroid = np.asarray(tfidf.mean(axis=0)).ravel()
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        relevance = np.asarray(tfidf @ centroid).ravel()

        n = len(candidates)
        remaining = set(range(n))
        selected: list[int] = []
        budget = self._TOKEN_BUDGET

        while remaining and budget > 0:
            if not selected:
                scores = relevance
            else:
                max_sim_to_selected = sim_matrix[:, selected].max(axis=1)
                scores = self._MMR_LAMBDA * relevance - (1 - self._MMR_LAMBDA) * max_sim_to_selected

            best = max(remaining, key=lambda i: scores[i])
            cost = self._estimate_tokens(candidates[best]["text"])

            if selected and cost > budget:
                remaining.discard(best)
                continue

            selected.append(best)
            remaining.discard(best)
            budget -= cost

        return [candidates[i]["text"] for i in selected]

    # -------------------------------------------------------------------------
    # chunk_db as an indexed store (one-time build) + batched entity_db scan
    # -------------------------------------------------------------------------

    @staticmethod
    def _detect_input_format(input_path: str) -> str:
        """Best-effort sniff whether input_path is Parquet or Arrow.

        Pass input_format explicitly to skip guessing.
        """
        p = Path(input_path)
        if p.is_dir():
            return "arrow" if any(p.glob("*.arrow")) else "parquet"
        return "arrow" if p.suffix == ".arrow" else "parquet"

    @staticmethod
    def _read_arrow_file(path: Path) -> pa.Table:
        """Read a single memory-mapped .arrow file without copying buffers.

        Try the IPC file format first, then fall back to the IPC stream format.
        """
        mm = pa.memory_map(str(path), "r")
        try:
            return pa.ipc.open_file(mm).read_all()
        except pa.ArrowInvalid:
            mm.seek(0)
            return pa.ipc.open_stream(mm).read_all()

    def _load_arrow_source(self, input_path: str) -> pa.Table:
        """Load one .arrow file or a shard directory into a single Table.

        Prefer state.json when available, and otherwise skip shard files that
        do not contain the required chunk_id_column and text_column fields.
        """
        p = Path(input_path)
        if not p.is_dir():
            return self._read_arrow_file(p)

        files: list[Path] = []
        state_path = p / "state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                candidate = [p / f["filename"] for f in state.get("_data_files", [])]
                files = [f for f in candidate if f.exists()]
            except (json.JSONDecodeError, KeyError, OSError) as exc:
                self.logger.warning("Could not parse %s (%s); falling back to glob", state_path, exc)
                files = []

        if not files:
            files = sorted(p.glob("*.arrow"))
        if not files:
            raise FileNotFoundError(f"No .arrow files found under {input_path}")

        tables = [self._read_arrow_file(f) for f in files]

        if not state_path.exists():
            wanted = {self.chunk_id_column, self.text_column}
            kept, skipped = [], []
            for f, t in zip(files, tables):
                (kept if wanted.issubset(t.schema.names) else skipped).append((f, t))
            if skipped:
                self.logger.warning(
                    "Skipping %d .arrow file(s) under %s missing columns %s "
                    "(likely a stray datasets cache/map file, not the source table): %s",
                    len(skipped), input_path, sorted(wanted), [str(f) for f, _ in skipped],
                )
            tables = [t for _, t in kept]
            if not tables:
                raise ValueError(
                    f"No .arrow files under {input_path} contain required columns {sorted(wanted)}"
                )

        return tables[0] if len(tables) == 1 else pa.concat_tables(tables)

    def _ensure_chunk_index(self) -> str:
        """Materialize chunk_db into a persistent indexed DuckDB table.

        This is built atomically, reused across runs, and supports either
        Parquet or Arrow inputs without needing an embedding column.
        """
        if os.path.exists(self._index_path) and not self._index_is_valid():
            self.logger.warning(
                "%s exists but has no usable 'chunks' table (likely left over "
                "from a crashed/interrupted build); rebuilding it",
                self._index_path,
            )
            os.remove(self._index_path)

        if not os.path.exists(self._index_path):
            tmp_path = f"{self._index_path}.building"
            if os.path.exists(tmp_path):
                os.remove(tmp_path)  # leftover from a previous crashed build
            con = duckdb.connect(tmp_path)
            try:
                con.execute("PRAGMA memory_limit='6GB'")
                if self.input_format == "arrow":
                    table = self._load_arrow_source(self.input_path)
                    con.register("chunks_src", table)
                    try:
                        con.execute(
                            f'CREATE TABLE chunks AS SELECT '
                            f'CAST("{self.chunk_id_column}" AS VARCHAR) AS id, '
                            f'CAST("{self.text_column}" AS VARCHAR) AS text '
                            f'FROM chunks_src'
                        )
                    finally:
                        con.unregister("chunks_src")
                else:
                    con.execute(
                        "CREATE TABLE chunks AS "
                        "SELECT id, text FROM read_parquet(?)",
                        [self.input_path],
                    )
                con.execute("CREATE UNIQUE INDEX chunks_pk ON chunks(id)")
            except Exception:
                con.close()
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)  # don't leave a stub for next run to trust
                raise
            else:
                con.close()
                os.replace(tmp_path, self._index_path)  # atomic on POSIX
        return self._index_path

    def _scan(self, loop: asyncio.AbstractEventLoop) -> None:
        """Run in a worker thread and stream entity_db in small batches.

        Each batch performs one indexed lookup into chunk_db for the needed
        chunk ids, keeping the work bounded by entity_batch_size.
        """
        chunk_db = self._ensure_chunk_index()
        con = duckdb.connect(chunk_db, read_only=True)
        con.execute("PRAGMA memory_limit='6GB'")
        self.logger.info(f"_scan started | Chunk index: {chunk_db}")

        try:
            pf = pq.ParquetFile(self.entity_db_path)
            batch_num = 0
            total_chunks_queued = 0
            total_entities_queued = 0
            for batch in pf.iter_batches(
                batch_size=self.entity_batch_size,
                columns=["id", "canonical_name", "label", "source_chunks"],
            ):
                batch_num += 1
                entities = batch.to_pylist()

                for e in entities:
                    sc = e["source_chunks"] or []
                    if len(sc) > self._HARD_CAP_BEFORE_CLUSTERING:
                        e["source_chunks"] = random.Random(0).sample(sc, self._HARD_CAP_BEFORE_CLUSTERING)
                
                # Track chunk statistics per batch
                chunk_counts = [len(e["source_chunks"] or []) for e in entities]
                chunk_count_stats = {
                    "min": min(chunk_counts) if chunk_counts else 0,
                    "max": max(chunk_counts) if chunk_counts else 0,
                    "avg": sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0,
                    "total": sum(chunk_counts),
                }

                for entity_idx, e in enumerate(entities):
                    source_chunks = e["source_chunks"] or []
                    num_chunks = len(source_chunks)
                    candidates = []

                    wanted = sorted({cid for cid in source_chunks if cid is not None})

                    chunk_lookup: dict[str, str] = {}
                    if wanted:
                        con.register("wanted_ids", pa.table({"id": wanted}))
                        try:
                            rows = con.execute(
                                "SELECT c.id, c.text "
                                "FROM chunks c JOIN wanted_ids w USING (id)"
                            ).fetchall()
                        finally:
                            con.unregister("wanted_ids")
                        chunk_lookup = {chunk_id: text for chunk_id, text in rows}

                    for cid in source_chunks:
                        text = chunk_lookup.get(cid)
                        if text is not None:
                            candidates.append({"id": cid, "text": text})

                    excerpts = self._select_excerpts(candidates)
                    record = {
                        "id": e["id"],
                        "canonical_name": e["canonical_name"],
                        "label": e["label"],
                        "source_chunks": e["source_chunks"],
                        "excerpts": excerpts,
                    }

                    if num_chunks > 100:  # Log only for entities with many chunks
                        self.logger.debug(
                            f"Entity {e['id'][:20]}... | chunks: {num_chunks} | "
                            f"candidates: {len(candidates)}"
                        )

                    asyncio.run_coroutine_threadsafe(self.ready.put(record), loop).result()
                    total_chunks_queued += num_chunks
                    total_entities_queued += 1

                if batch_num % 10 == 0:
                    queue_size = self.ready.qsize()
                    self.logger.info(
                        f"Batch {batch_num}: {len(entities)} entities | "
                        f"Chunks/entity: min={chunk_count_stats['min']}, max={chunk_count_stats['max']}, "
                        f"avg={chunk_count_stats['avg']:.1f}, total={chunk_count_stats['total']} | "
                        f"Queue: {queue_size} items | "
                        f"Cumulative: {total_entities_queued} entities, {total_chunks_queued} chunks"
                    )
        finally:
            con.close()
        
        self.logger.info(
            f"_scan phase complete | Total batches: {batch_num} | "
            f"Total queued: {total_entities_queued} entities, {total_chunks_queued} chunks | "
            f"Final queue size: {self.ready.qsize()} items"
        )
        asyncio.run_coroutine_threadsafe(self.ready.put(None), loop).result()

    async def generate_descriptions(self) -> str:
        # Row count from the Parquet footer only — no data actually read yet.
        self._total = pq.ParquetFile(self.entity_db_path).metadata.num_rows
        return await super().generate_descriptions()



class CommunityDescriptionGenerator(DescriptionGenerator):
    _SCHEMA = pa.schema([
            pa.field("cluster_id",      pa.string()),
            pa.field("entities",        pa.list_(pa.string())),
            pa.field("description",     pa.string()),
        ])
    _PROMPT = (
            'You are an information extraction assistant. Your task is to write descriptions for entity clusters, based on the entities it contains and their descriptions, and the relationships between them. '
            'Focus only on factual information directly supported by the excerpts. '
            'Your response must only contain the description text, without any additional commentary or formatting.'
            'Be specific and avoid generic statements.\n\n'
            'The relationships and entity descriptions you are given are all from the cluster "{cluster_id}".'
            'Write a description for the entity cluster "{cluster_id}".\n'
            'Relationships with entity descriptions:\n{relationships}\n\n'
            'Description of "{cluster_id}":'
        )
    
    def __init__(self, *args, entity_index_path: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._index_path = entity_index_path or f"{self.input_path}.entities.duckdb"
        self.ready: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=6000)
        self._community_ids: list[int] | None = None
    
    async def _describe(self, record: dict, pbar: tqdm) -> None:
        prompt = self._PROMPT.format(
            cluster_id=record["cluster_id"],
            relationships="\n---\n".join(
                f"{r['head_desc']} ({r['head']}) --[{', '.join(r['relations'])}]--> {r['tail']} ({r['tail_desc']})"
                for r in record["relationships"]
            ),
            )
        async with self.semaphore:
            response = await self.client.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 1000},
            )
        description = response["response"].strip()

        async with self.write_lock:
            self.buffer.append({
                "cluster_id":     record["cluster_id"],
                "entities":       record["entities"],
                "description":    description,
            })
        await self._flush()
        pbar.update(1)
        
        
    def _ensure_entity_index(self) -> str:
        """Materialize entity_db into a persistent indexed DuckDB table.

        This is a one-time build for later batched lookups.
        """
        if os.path.exists(self._index_path) and not self._index_is_valid():
            self.logger.warning(
                "%s exists but has no usable 'chunks' table (likely left over "
                "from a crashed/interrupted build); rebuilding it",
                self._index_path,
            )
            os.remove(self._index_path)
        
        if not os.path.exists(self._index_path):
            con = duckdb.connect(self._index_path)
            try:
                con.execute("PRAGMA memory_limit='6GB'")
                con.execute(
                    "CREATE TABLE entities AS "
                    "SELECT id, canonical_name, label, description FROM read_parquet(?)",
                    [self.entity_db_path],
                )
                con.execute("CREATE UNIQUE INDEX entities_pk ON entities(id)")
            finally:
                con.close()
        return self._index_path
    
    def _get_top_k_by_degree(self, cluster_id: str, input_path: str, k: int) -> pl.DataFrame:
        """Compute the top-k relationships for a community by weighted degree."""
        in_degree_df = pl.scan_parquet(input_path+f"_community_{cluster_id}_relations.parquet") \
            .group_by("tail") \
            .agg((pl.col("weight").sum()/pl.count("head")).alias("in_degree")) \
            .rename({"tail": "id"})
        
        degree_df = pl.scan_parquet(input_path+f"_community_{cluster_id}_relations.parquet") \
            .group_by("head") \
            .agg((pl.col("weight").sum()/pl.count("tail")).alias("out_degree")) \
            .rename({"head": "id"}) \
            .join(in_degree_df, on="id", how="full", coalesce=True) \
            .fill_null(0) \
            .with_columns((pl.col("in_degree") + pl.col("out_degree")).alias("degree")) \
        
        df = pl.scan_parquet(input_path+f"_community_{cluster_id}_relations.parquet") \
            .join(degree_df.select("id", "degree"), left_on="head", right_on="id", how="inner") \
            .join(degree_df.select("id", "degree"), left_on="tail", right_on="id", how="inner", suffix="_tail") \
            .with_columns((pl.col("degree") + pl.col("degree_tail")).alias("total_degree")) \
            .top_k(k, by="total_degree") \
            .select(["head", "relations", "tail"]) \
            .collect()
        
        return df
    
    def _discover_community_ids(self) -> list[int]:
        """Find cluster ids from the community relation Parquet files on disk.

        This globs the filename pattern in the parent directory and extracts
        the ids with a regex because self.input_path is a file prefix.
        """
        base = Path(self.input_path)
        name_pattern = re.compile(r"^_community_(\d+)_relations\.parquet$")
        ids = []
        for f in base.glob("_community_*_relations.parquet"):
            m = name_pattern.match(f.name)
            if m and int(m.group(1)) !=- 1:
                ids.append(int(m.group(1)))
        return sorted(ids)

    def _scan(self, loop: asyncio.AbstractEventLoop) -> None:
        
        self._ensure_entity_index()
        con=duckdb.connect(self._index_path, read_only=True)
        con.execute("PRAGMA memory_limit='6GB'")
        
        community_ids = self._community_ids if self._community_ids is not None else self._discover_community_ids()
        
        for cluster_id in community_ids:
            top_k_relations = self._get_top_k_by_degree(str(cluster_id), self.input_path, k=10)
            wanted=sorted(set(top_k_relations.select("head").to_series().to_list() + top_k_relations.select("tail").to_series().to_list()))
            entity_lookup: dict[str, dict[str, str]] = {}
            if wanted:
                con.register("wanted_ids", pa.table({"id": wanted}))
                try:
                    rows = con.execute(
                        "SELECT id, canonical_name, label, description FROM entities JOIN wanted_ids USING (id)"
                    ).fetchall()
                    entity_lookup = {
                        id_: {"canonical_name": canonical_name, "label": label, "description": description}
                        for (id_, canonical_name, label, description) in rows
                    }
                finally:
                    con.unregister("wanted_ids")

            relationships: list[dict[str, str]] = []
            for row in top_k_relations.iter_rows(named=True):
                head_info = entity_lookup.get(row["head"])
                tail_info = entity_lookup.get(row["tail"])
                if head_info is None or tail_info is None:
                    continue  # entity description not available yet; skip this edge
                relationships.append({
                    "head": head_info["canonical_name"],
                    "head_desc": head_info["description"],
                    "relations": row["relations"],
                    "tail": tail_info["canonical_name"],
                    "tail_desc": tail_info["description"],
                })

            record = {
                "cluster_id": str(cluster_id),
                "entities": wanted,
                "relationships": relationships,
            }
            asyncio.run_coroutine_threadsafe(self.ready.put(record), loop).result()
        
        con.close()
        asyncio.run_coroutine_threadsafe(self.ready.put(None), loop).result()

    async def generate_descriptions(self) -> str:
        # Discover communities once up front so both the progress-bar total
        # and the _scan loop use the same list (also avoids re-globbing).
        self._community_ids = self._discover_community_ids()
        self._total = len(self._community_ids)
        return await super().generate_descriptions()