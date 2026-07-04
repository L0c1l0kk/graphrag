import asyncio
import logging
import os
import random
from abc import ABC, abstractmethod

import duckdb
import numpy as np
import ollama
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


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

    _SCHEMA = None
    _PROMPT = None

    async def _flush(self, force: bool = False) -> None:
        if not self.buffer:
            return
        async with self.write_lock:
            if not force and len(self.buffer) < self.flush_every:
                return
            batch = self.buffer
            self.buffer = []
        table= pa.table({f.name : [r[f.name] for r in batch] for f in self._SCHEMA}, schema=self._SCHEMA)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.output_path, self._SCHEMA, compression="zstd")
        self.writer.write_table(table)

    @abstractmethod
    def _scan(self, loop: asyncio.AbstractEventLoop) -> None:
        """Runs in a worker thread (see _collect). Must push fully-formed
        record dicts onto self.ready via loop.call_soon_threadsafe, then
        finish with a single None sentinel."""

    async def _collect(self) -> None:
        loop = asyncio.get_running_loop()
        await asyncio.to_thread(self._scan, loop)

    @abstractmethod
    async def _describe(self, record: dict, pbar: tqdm) -> None:
        pass

    async def _consume(self, pbar: tqdm) -> None:
        tasks = []
        while True:
            item = await self.ready.get()
            if item is None:
                break
            tasks.append(asyncio.create_task(self._describe(item, pbar)))
        await asyncio.gather(*tasks)
        await self._flush(force=True)  # drain the last partial batch
        if self.writer:
            self.writer.close()

    async def generate_descriptions(self) -> str:
        if self._total is None:
            raise ValueError("Row count (self._total) not set — did the subclass set it before calling super()?")
        with tqdm(total=self._total, desc="Generating descriptions") as pbar:
            await asyncio.gather(self._collect(), self._consume(pbar))

        return self.output_path


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
            'Be specific and avoid generic statements.\n\n'
            'Excerpts:\n{excerpts}\n\n'
            'Description of "{entity_name}":'
        )

    # --- evidence-selection knobs -------------------------------------------------
    _TOKEN_BUDGET = 2000                  # excerpts-block token budget per prompt
    _MMR_LAMBDA = 0.7                     # 1.0 = pure relevance, 0.0 = pure diversity
    _MAX_CANDIDATES_FOR_MMR = 200         # cap MMR's O(k*n) cost for hub entities
    _HARD_CAP_BEFORE_CLUSTERING = 5000    # safety valve before even clustering

    def __init__(self, *args, chunk_index_path: str | None = None, entity_batch_size: int = 2000, **kwargs):
        super().__init__(*args, **kwargs)
        # Persistent, indexed DuckDB table built once from chunk_db (self.input_path).
        # Reused across runs — delete this file if chunk_db is regenerated.
        self.chunk_index_path = chunk_index_path or f"{self.input_path}.chunks.duckdb"
        # How many entities' worth of source_chunks we resolve per lookup batch.
        # This is the read-side analogue of flush_every: it bounds how much
        # chunk text/embedding data is resident in memory at once, independent
        # of how large entity_db or chunk_db are in total.
        self.entity_batch_size = entity_batch_size

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
        await self._flush()
        pbar.update(1)

    # -------------------------------------------------------------------------
    # Evidence selection (MMR under a token budget, over an entity's mentions)
    #
    # No chunk-level embeddings exist (and none are being generated), so this
    # uses the *original* MMR formulation (Carbonell & Goldstein, 1998), which
    # was TF-IDF-based to begin with — not a fallback approximation. A fresh
    # TF-IDF space is fit per entity, over just that entity's own candidate
    # excerpts (at most a few hundred short texts after downsampling), so this
    # is cheap, local, and needs no embedding model or GPU at all.
    # -------------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Cheap ~4-chars/token approximation — a safe overestimate for most
        # tokenizers on English text. Swap for a real tokenizer if you need
        # this to be exact rather than a conservative upper bound.
        return max(1, len(text) // 4)

    @staticmethod
    def _tfidf(texts: list[str]):
        # max_features keeps the matrix bounded even if some entity's mentions
        # span a huge and varied vocabulary (hub entities). Rows are L2-normalized
        # by default, so row dot-products below are already cosine similarities.
        vectorizer = TfidfVectorizer(max_features=20_000, stop_words="english")
        return vectorizer.fit_transform(texts)

    @staticmethod
    def _tfidf_cluster_representatives(tfidf, k: int, seed: int = 0) -> list[int]:
        """Cluster candidates in TF-IDF space, return the index of the real
        candidate closest to each cluster center (a real excerpt, not a
        synthetic average)."""
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
        """For hub entities with far more mentions than could ever fit in a
        prompt, cluster candidate texts (TF-IDF space) and keep one
        representative per cluster, instead of letting MMR degrade into
        O(k * n) over thousands of near-duplicate mentions."""
        if len(candidates) > self._HARD_CAP_BEFORE_CLUSTERING:
            rng = random.Random(0)
            candidates = rng.sample(candidates, self._HARD_CAP_BEFORE_CLUSTERING)

        tfidf = self._tfidf([c["text"] for c in candidates])
        rep_indices = self._tfidf_cluster_representatives(tfidf, k)
        return [candidates[i] for i in rep_indices]

    def _select_excerpts(self, candidates: list[dict]) -> list[str]:
        """MMR selection under a token budget, in TF-IDF space.

        candidates: [{"chunk_id", "text"}, ...] for every chunk that mentions
        this entity.

        "Relevance" is similarity to the centroid of the entity's own
        candidate texts — prefer excerpts representative of how the entity is
        typically discussed, then diversify away from what's already picked.
        Every candidate already mentions the entity by construction (that's
        why it's in source_chunks), so this is really doing redundancy
        reduction more than topical filtering.
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
                remaining.discard(best)  # doesn't fit this round; let a cheaper one win
                continue

            selected.append(best)
            remaining.discard(best)
            budget -= cost

        return [candidates[i]["text"] for i in selected]

    # -------------------------------------------------------------------------
    # chunk_db as an indexed store (one-time build) + batched entity_db scan
    # -------------------------------------------------------------------------

    def _ensure_chunk_index(self) -> str:
        """One-time cost: materialize chunk_db into a persistent, indexed
        DuckDB table so later batched lookups are index probes against a
        ~21M-row table, not a full join+sort of it. Reused across runs.

        Assumes chunk_db (self.input_path) has columns: chunk_id, text.
        (No embedding column needed — evidence selection runs on TF-IDF
        vectors fit locally per entity, not on precomputed chunk embeddings.)
        """
        if not os.path.exists(self.chunk_index_path):
            con = duckdb.connect(self.chunk_index_path)
            try:
                con.execute("PRAGMA memory_limit='6GB'")
                con.execute(
                    "CREATE TABLE chunks AS "
                    "SELECT chunk_id, text FROM read_parquet(?)",
                    [self.input_path],
                )
                con.execute("CREATE UNIQUE INDEX chunks_pk ON chunks(chunk_id)")
            finally:
                con.close()
        return self.chunk_index_path

    def _scan(self, loop: asyncio.AbstractEventLoop) -> None:
        """Runs in a worker thread (see DescriptionGenerator._collect).

        Streams entity_db in small batches via pyarrow (bounded memory — never
        materializes the full entity_db as Python objects), and for each batch
        does one indexed lookup into chunk_db for just the chunk_ids that batch
        needs. This replaces a global (entity, chunk) join: instead of sorting
        the whole edge list with heavy text/embedding payloads attached (which
        wouldn't fit comfortably in 16GB), we do many small, index-assisted
        lookups, each bounded by entity_batch_size.

        Assumes entity_db (self.entity_db_path) has columns:
        id, canonical_name, label, source_chunks (list<string>).
        """
        chunk_db = self._ensure_chunk_index()
        con = duckdb.connect(chunk_db, read_only=True)
        con.execute("PRAGMA memory_limit='6GB'")
        con.execute("PRAGMA threads=4")

        try:
            pf = pq.ParquetFile(self.entity_db_path)
            for batch in pf.iter_batches(
                batch_size=self.entity_batch_size,
                columns=["id", "canonical_name", "label", "source_chunks"],
            ):
                entities = batch.to_pylist()

                wanted = sorted({cid for e in entities for cid in (e["source_chunks"] or [])})
                chunk_lookup: dict[str, str] = {}
                if wanted:
                    con.register("wanted_ids", pa.table({"chunk_id": wanted}))
                    try:
                        rows = con.execute(
                            "SELECT c.chunk_id, c.text "
                            "FROM chunks c JOIN wanted_ids w USING (chunk_id)"
                        ).fetchall()
                    finally:
                        con.unregister("wanted_ids")
                    chunk_lookup = {chunk_id: text for chunk_id, text in rows}

                for e in entities:
                    candidates = []
                    for cid in (e["source_chunks"] or []):
                        text = chunk_lookup.get(cid)
                        if text is not None:
                            candidates.append({"chunk_id": cid, "text": text})

                    excerpts = self._select_excerpts(candidates)
                    record = {
                        "id": e["id"],
                        "canonical_name": e["canonical_name"],
                        "label": e["label"],
                        "source_chunks": e["source_chunks"],
                        "excerpts": excerpts,
                    }
                    asyncio.run_coroutine_threadsafe(self.ready.put(record), loop).result()
        finally:
            con.close()

        asyncio.run_coroutine_threadsafe(self.ready.put(record), loop).result()

    async def generate_descriptions(self) -> str:
        # Row count from the Parquet footer only — no data actually read yet.
        self._total = pq.ParquetFile(self.entity_db_path).metadata.num_rows
        return await super().generate_descriptions()
