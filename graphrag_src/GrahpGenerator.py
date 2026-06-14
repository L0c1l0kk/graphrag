import ollama
from datasets import load_from_disk, Dataset

from tqdm import tqdm
from EntityRelationExtractor import EntityRelationExtractor
from typing import List, Dict, Optional, Any
import asyncio
import pyarrow as pa
import pyarrow.parquet as pq


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
            if not force and len(buffer) < flush_every:
                return
            async with write_lock:
                batch = buffer
                buffer = []
            table = pa.table({
                "id":             [r["id"]             for r in batch],
                "canonical_name": [r["canonical_name"] for r in batch],
                "label":          [r["label"]          for r in batch],
                "description":    [r["description"]    for r in batch],
            }, schema=self.ENTITY_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(output_path, self.ENTITY_SCHEMA, compression="snappy")
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
                    for record in batch:
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
    
    def _generate_graph(self,entity_db):
        