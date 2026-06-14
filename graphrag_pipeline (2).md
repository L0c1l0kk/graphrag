# Lightweight GraphRAG Pipeline — Design & Implementation Draft

A local, low-cost approximation of Microsoft's GraphRAG pipeline, running on the NVIDIA A40 (48GB VRAM). Uses GLiNER for NER, a quantized local LLM (via Ollama) for entity description and community summarization, and NetworkX + leidenalg for graph construction and clustering.

## Model Selection Summary

| Stage | Model | Justification |
|---|---|---|
| **Stage 2 — NER + RE** | `knowledgator/gliner-relex-large-v1.0` | True joint NER+RE in a single forward pass with a dedicated RE module; zero-shot relation types via plain-string labels (no fixed taxonomy); state-of-the-art on DocRED (31.3% Micro-F1, vs 18.6% for GPT-4o-mini) for cross-sentence reasoning; ~70× throughput over GPT-based RE (0.9 s/doc vs 64 s/doc) |
| **Stage 3 — Dedup embeddings** | `BAAI/bge-m3` | Matches the retrieval embedding space exactly, so dedup similarity scores and retrieval similarity scores are on the same scale; 8192-token context window handles long entity surface forms gracefully |
| **Stage 4/7 — LLM** | `qwen2.5:32b-instruct-q4_K_M` | Best open-weight instruction-following model that fits fully in GPU memory on the A40 at Q4_K_M (~20GB weights + ~3GB KV cache); dramatically better structured output quality vs. 8B models, especially for JSON-mode entity descriptions and community summaries; 128K context window |
| **Stage 8 — Retrieval embeddings** | `BAAI/bge-m3` | State-of-the-art MTEB retrieval scores; supports all three retrieval modes (dense, sparse/lexical, multi-vector/ColBERT); enables hybrid ranking that fuses all three signals at query time, which is superior to cosine-only retrieval |

---

## Architecture Overview

```
Raw text corpus
      │
      ▼
[Stage 1] Chunking
      │
      ▼
[Stage 2] Entity & Relation Extraction (GLiNER)
      │
      ▼
[Stage 3] Entity Deduplication & Normalization
      │
      ▼
[Stage 4] Entity Description Generation (local LLM via Ollama)
      │
      ▼
[Stage 5] Graph Construction (NetworkX)
      │
      ▼
[Stage 6] Leiden Community Detection (leidenalg)
      │
      ▼
[Stage 7] Community Summarization (local LLM via Ollama)
      │
      ▼
[Stage 8] Embedding & Indexing (sentence-transformers + ChromaDB)
      │
      ▼
[Stage 9] Query — Local Search / Global Search
```

Each stage is designed to be run independently and checkpoint to disk, so the pipeline is resumable.

---

## Dependencies

### System Prerequisites

- **CUDA** ≥ 12.1 (for GPU acceleration of GLiNER and embeddings)
- **Ollama** installed separately: https://ollama.com/download
  - Pull model before running: `ollama pull qwen2.5:32b-instruct-q4_K_M`
  - This uses ~20GB VRAM at Q4_K_M; GLiNER uses ~1–2GB; the A40's 48GB means **both can run simultaneously**. No sequential GPU scheduling is required.

### `environment.yaml` (conda)

```yaml
name: graphrag-local
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pytorch>=2.2.0
  - torchvision
  - pytorch-cuda=12.1
  - pip
  - pip:
    # NER
    - gliner>=0.2.13

    # Graph construction & clustering
    - networkx>=3.3
    - leidenalg>=0.10.2
    - igraph>=0.11.6         # leidenalg depends on python-igraph

    # LLM inference (local, via Ollama REST API)
    - ollama>=0.2.0          # Python client for the Ollama server

    # Embeddings & vector store
    - sentence-transformers>=3.0.0
    - chromadb>=0.5.0

    # Data handling & utilities
    - datasets>=2.20.0       # for loading HuggingFace datasets (e.g. Wikipedia)
    - pandas>=2.2.0
    - tqdm>=4.66.0
    - numpy>=1.26.0
    - scikit-learn>=1.5.0    # cosine similarity for deduplication

    # Serialization / checkpointing
    - orjson>=3.10.0         # fast JSON; used for checkpoint files
```

### Alternative: `requirements.txt`

```
torch>=2.2.0
gliner>=0.2.13
networkx>=3.3
leidenalg>=0.10.2
igraph>=0.11.6
ollama>=0.2.0
sentence-transformers>=3.0.0
chromadb>=0.5.0
datasets>=2.20.0
pandas>=2.2.0
tqdm>=4.66.0
numpy>=1.26.0
scikit-learn>=1.5.0
orjson>=3.10.0
```

> ⚠️ **Note on `leidenalg`**: it requires `python-igraph` as a backend. The conda install handles this cleanly. With pip only, you may need `pip install leidenalg python-igraph` and verify C extension compilation. On Windows, pre-built wheels are available on PyPI; on Linux, build from source if wheels are not available for your Python version.

> ⚠️ **Note on `gliner` GPU**: GLiNER uses `onnxruntime` by default (CPU). For GPU acceleration install `onnxruntime-gpu` and set `use_onnx=True`, or call `model.to("cuda")` to use the native PyTorch CUDA path. On the A40, run GLiNER on GPU — there is ample VRAM headroom alongside the LLM.

---

## Stage 1 — Chunking

The corpus is split into fixed-size chunks with overlap. For the Wikipedia/DPR corpus this is already done (100-word passages); for raw text use the approach below.

```python
# stage1_chunking.py
from datasets import load_dataset
import orjson, pathlib
from tqdm import tqdm

CHUNK_SIZE = 150        # words
CHUNK_OVERLAP = 20      # words
OUTPUT_FILE = "data/chunks.jsonl"

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks

# Example: Wikipedia via HuggingFace datasets
dataset = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)

pathlib.Path("data").mkdir(exist_ok=True)
with open(OUTPUT_FILE, "wb") as f:
    chunk_id = 0
    for doc in tqdm(dataset, desc="Chunking"):
        for chunk in chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP):
            record = {"id": chunk_id, "doc_id": doc["id"], "text": chunk}
            f.write(orjson.dumps(record) + b"\n")
            chunk_id += 1
```

**Notes:**
- The full English Wikipedia dataset is ~20M passages at 100 words. Loading via HuggingFace streams it from disk if you use `streaming=True`, avoiding RAM issues.
- Adjust `CHUNK_SIZE` to stay within GLiNER's recommended input length (~512 tokens max; 150 words ≈ 200 tokens, well within bounds).

---

## Stage 2 — Entity & Relation Extraction (GLiNER-Relex)

`knowledgator/gliner-relex-large-v1.0` extends the GLiNER framework with a dedicated relation extraction module, performing joint NER and RE in a **single forward pass** with shared representations.  Unlike the earlier multitask model (which encoded relations implicitly through label concatenation), Relex explicitly models entity pairs, making it a true joint model.  Relation types are specified at inference time as plain English strings — no fixed taxonomy — so "Erwin Schrödinger — developed — wave mechanics" is extracted directly rather than inferred from co-occurrence alone.

```python
# stage2_ner.py
from gliner import GLiNER
import orjson, pathlib
from tqdm import tqdm

ENTITY_LABELS = [
    "person", "organization", "location", "event",
    "concept", "product", "work of art", "law", "disease", "date"
]
RELATION_LABELS = [
    "developed", "discovered", "founded", "caused", "treated by",
    "located in", "part of", "authored", "affiliated with",
    "instance of", "preceded by", "influenced"
]

INPUT_FILE  = "data/chunks.jsonl"
OUTPUT_FILE = "data/extractions.jsonl"
BATCH_SIZE  = 16         # Relex builds all entity pairs per text; 16 is safe on A40
THRESHOLD          = 0.4  # entity confidence cutoff
RELATION_THRESHOLD = 0.5  # relation confidence cutoff
MAX_ENTITIES_PER_CHUNK = 30  # cap to bound O(n²) pair-enumeration cost

model = GLiNER.from_pretrained("knowledgator/gliner-relex-large-v1.0")
model = model.to("cuda")

def process_batch(batch: list[dict]) -> list[dict]:
    texts = [item["text"] for item in batch]

    # Single forward pass — joint NER + RE
    entities_all, relations_all = model.inference(
        texts=texts,
        labels=ENTITY_LABELS,
        relations=RELATION_LABELS,
        threshold=THRESHOLD,
        relation_threshold=RELATION_THRESHOLD,
        return_relations=True,
        flat_ner=False,
    )

    results = []
    for item, entities, relations in zip(batch, entities_all, relations_all):
        entities = entities[:MAX_ENTITIES_PER_CHUNK]   # cap before pair enumeration
        results.append({
            "chunk_id":  item["id"],
            "doc_id":    item["doc_id"],
            "text":      item["text"],
            "entities":  [
                {"text": e["text"], "label": e["label"], "score": round(e["score"], 4)}
                for e in entities
            ],
            "relations": [
                {
                    "head":       r["head"],
                    "head_label": r.get("head_type", ""),
                    "relation":   r["relation"],
                    "tail":       r["tail"],
                    "tail_label": r.get("tail_type", ""),
                    "score":      round(r["score"], 4),
                }
                for r in relations
            ],
        })
    return results

pathlib.Path("data").mkdir(exist_ok=True)

with open(INPUT_FILE, "rb") as fin, open(OUTPUT_FILE, "wb") as fout:
    batch = []
    for line in tqdm(fin, desc="NER"):
        record = orjson.loads(line)
        batch.append(record)
        if len(batch) == BATCH_SIZE:
            for r in process_batch(batch):
                fout.write(orjson.dumps(r) + b"\n")
            batch = []
    if batch:
        for r in process_batch(batch):
            fout.write(orjson.dumps(r) + b"\n")
```

**Notes:**
- **Why Relex over multitask-GLiNER?** The multitask model encoded relations implicitly through label concatenation (NER dressed up as RE — no explicit entity-pair modelling). Relex introduces a dedicated RE module in the same encoder, giving true joint extraction. On DocRED (document-level cross-sentence reasoning, the closest benchmark to Wikipedia text) it scores 31.3% Micro-F1 vs GPT-4o-mini's 18.6% and the multitask variant's weaker performance; throughput is ~70× faster than GPT-based approaches (~0.9 s/doc vs 64 s/doc). For REBEL: it often adds world-knowledge triples not present in the source text, which is undesirable for a grounded extraction pipeline.
- **Quadratic RE scaling**: Relex builds all entity-pair candidates per chunk, so cost is O(n²) in entity count. With 150-word chunks entity density is typically low, but `MAX_ENTITIES_PER_CHUNK` provides a hard cap for pathological passages.
- `BATCH_SIZE = 16` is conservative; tune upward if VRAM headroom remains. On the A40, expect ~2,000–4,000 chunks/min.
- Keep `THRESHOLD` at 0.4–0.5 for entities; `RELATION_THRESHOLD` at 0.5–0.6 for relations. Lower relation thresholds increase recall but introduce spurious edges.

> ⚠️ **Known issue**: GLiNER's documentation warns that performance degrades with >30 entity types due to joint encoding overhead. Keep entity label count under 20 where possible; the same limit applies to relation labels.

---

## Stage 3 — Entity Deduplication & Normalization

This is the most important correctness step. Duplicate entity nodes ("Schrödinger", "E. Schrödinger", "Erwin Schrödinger") fragment the graph and corrupt community structure.

```python
# stage3_dedup.py
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import orjson, pathlib
from collections import defaultdict
from tqdm import tqdm

INPUT_FILE     = "data/extractions.jsonl"
ENTITY_MAP_OUT = "data/entity_map.json"   # raw_text -> canonical_id
ENTITY_DB_OUT  = "data/entities.json"
SIM_THRESHOLD  = 0.92   # cosine similarity above which two entities are merged

# BGE-M3: same model used in Stage 8 for retrieval — dedup and retrieval share
# the same embedding space, so cosine similarity scores are directly comparable.
# Dense retrieval head produces normalized CLS embeddings; cosine == inner product.
embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

# Step 1: collect all unique entity strings
raw_entities: dict[str, dict] = {}   # text -> {label, count}
with open(INPUT_FILE, "rb") as f:
    for line in tqdm(f, desc="Collecting entities"):
        record = orjson.loads(line)
        for ent in record["entities"]:
            key = ent["text"].strip().lower()
            if key not in raw_entities:
                raw_entities[key] = {"label": ent["label"], "count": 0, "surface_forms": set()}
            raw_entities[key]["count"] += 1
            raw_entities[key]["surface_forms"].add(ent["text"])

# Step 2: string normalization (cheap first pass)
# Group by lowercased, punctuation-stripped form
normalized: dict[str, str] = {}   # normalized_form -> canonical (most frequent) form
for raw, info in raw_entities.items():
    canonical = max(info["surface_forms"], key=lambda s: raw_entities[s.strip().lower()]["count"])
    normalized[raw] = canonical.strip()

# Step 3: embedding-based merge (within same label type, batched)
# Group entity strings by label to limit quadratic comparison cost
by_label: dict[str, list[str]] = defaultdict(list)
canonical_set = set(normalized.values())
for entity_text in canonical_set:
    label = raw_entities[entity_text.strip().lower()]["label"]
    by_label[label].append(entity_text)

entity_map: dict[str, str] = {}  # surface form -> final canonical id
entity_db:  dict[str, dict] = {}  # canonical_id -> {label, surface_forms, source_chunks: []}
entity_counter = 0

for label, entities in by_label.items():
    if len(entities) == 0:
        continue
    # BGE-M3 encode returns a dict with 'dense_vecs', 'lexical_weights', 'colbert_vecs'
    output = embed_model.encode(
        entities, batch_size=256,
        max_length=128,             # entity strings are short; cap for efficiency
        return_dense=True,
        return_sparse=False,        # not needed for dedup
        return_colbert_vecs=False
    )
    embeddings = output["dense_vecs"]  # shape: (n, 1024), already L2-normalized
    sim_matrix = cosine_similarity(embeddings)
    merged = [-1] * len(entities)
    cluster_id = 0
    for i in range(len(entities)):
        if merged[i] == -1:
            merged[i] = cluster_id
            for j in range(i + 1, len(entities)):
                if merged[j] == -1 and sim_matrix[i, j] >= SIM_THRESHOLD:
                    merged[j] = cluster_id
            cluster_id += 1
    # Build canonical entries
    clusters: dict[int, list[str]] = defaultdict(list)
    for idx, cid in enumerate(merged):
        clusters[cid].append(entities[idx])
    for cid, members in clusters.items():
        canonical_name = max(members, key=len)
        eid = f"E{entity_counter:07d}"
        entity_counter += 1
        entity_db[eid] = {
            "id": eid, "label": label,
            "canonical_name": canonical_name,
            "surface_forms": members,
            "source_chunks": []
        }
        for m in members:
            entity_map[m] = eid

# Save
with open(ENTITY_MAP_OUT, "wb") as f:
    f.write(orjson.dumps(entity_map))
with open(ENTITY_DB_OUT, "wb") as f:
    # Convert sets to lists for serialization
    for v in entity_db.values():
        v["surface_forms"] = list(v["surface_forms"])
    f.write(orjson.dumps(entity_db))

print(f"Reduced to {entity_counter} canonical entities from {len(raw_entities)} surface forms")
```

**Notes:**
- **Why BGE-M3 instead of MiniLM or mpnet?** Using the same model for dedup and retrieval means the semantic space is identical. When Stage 8 indexes entity descriptions and Stage 9 queries against them, the cosine similarity scores are calibrated to the same scale as the dedup threshold. Mixing models (e.g. MiniLM for dedup, mpnet for retrieval) means a 0.92 threshold in dedup space is meaningless relative to retrieval space. BGE-M3's dense retrieval head produces L2-normalized CLS embeddings, so cosine and inner product are equivalent — no ambiguity about the similarity function.
- `use_fp16=True` halves VRAM usage for the embedding model with negligible quality loss; recommended on A40.
- The embedding-based merge is O(n²) per label class. For large corpora, use approximate nearest neighbor search (e.g. `faiss`) instead of `cosine_similarity` for label groups with >10,000 members.
- `SIM_THRESHOLD = 0.92` is conservative; tune downward carefully — false merges (conflating distinct entities) are more harmful than false splits.
- After this stage, re-read `extractions.jsonl` and replace all entity surface forms with their canonical IDs.

---

## Stage 4 — Entity Description Generation (Local LLM via Ollama)

**VRAM scheduling**: GLiNER should be fully done before starting this stage. Ollama will use the GPU. Verify with `nvidia-smi` that VRAM is free before starting Ollama.

```python
# stage4_descriptions.py
import ollama
import orjson
from tqdm import tqdm

ENTITY_DB_FILE  = "data/entities.json"
EXTRACTIONS_FILE = "data/extractions.jsonl"
OUTPUT_FILE     = "data/entities_with_desc.json"

MODEL = "llama3.1:8b-instruct-q4_K_M"  # must be pulled via: ollama pull llama3.1:8b-instruct-q4_K_M

# Load entity DB and build entity -> source chunks map
with open(ENTITY_DB_FILE, "rb") as f:
    entity_db: dict = orjson.loads(f.read())
with open("data/entity_map.json", "rb") as f:
    entity_map: dict = orjson.loads(f.read())

# Map entity IDs to their source text chunks (sample up to 3 per entity)
entity_chunks: dict[str, list[str]] = {eid: [] for eid in entity_db}
with open(EXTRACTIONS_FILE, "rb") as f:
    for line in f:
        record = orjson.loads(line)
        for ent in record["entities"]:
            surface = ent["text"].strip()
            eid = entity_map.get(surface.lower())
            if eid and len(entity_chunks[eid]) < 3:
                entity_chunks[eid].append(record["text"])

DESCRIPTION_PROMPT = """\
You are an information extraction assistant. Based on the following text excerpts, write a concise 2-3 sentence description of the entity "{entity_name}" (type: {entity_type}).
Focus only on factual information directly supported by the excerpts. Be specific and avoid generic statements.

Excerpts:
{excerpts}

Description of "{entity_name}":"""

client = ollama.Client()

with open(ENTITY_DB_FILE, "rb") as f:
    entity_db = orjson.loads(f.read())

for eid, entity in tqdm(entity_db.items(), desc="Generating descriptions"):
    chunks = entity_chunks.get(eid, [])
    if not chunks:
        entity["description"] = f"A {entity['label']} named {entity['canonical_name']}."
        continue
    prompt = DESCRIPTION_PROMPT.format(
        entity_name=entity["canonical_name"],
        entity_type=entity["label"],
        excerpts="\n---\n".join(chunks[:3])
    )
    response = client.generate(
        model=MODEL,
        prompt=prompt,
        options={"temperature": 0.1, "num_predict": 150}
    )
    entity["description"] = response["response"].strip()

with open(OUTPUT_FILE, "wb") as f:
    f.write(orjson.dumps(entity_db))
```

**Notes:**
- At ~20 tokens/sec on a 4060 with 150-token outputs: 1M entities × 150 tokens / 20 tok/s ≈ ~87 days. **This is the real bottleneck for Wikipedia-scale.**
- For a tractable experiment: subsample to a domain (e.g. one Wikipedia category), or skip description generation for entities with fewer than N source chunks (they are likely noise anyway) and use a template description as fallback.
- The Ollama Python client (`ollama>=0.2.0`) exposes a clean `client.generate()` / `client.chat()` API wrapping the local REST server at `localhost:11434`.

> ⚠️ **Known issue**: Models below ~7B parameters frequently produce malformed or off-topic descriptions when given noisy input chunks. If using a smaller model (e.g. 3B), add an explicit JSON output schema and validate responses.

---

## Stage 5 — Graph Construction (NetworkX)

Build the entity co-occurrence graph. Two entities share an edge if they co-occur in the same chunk; edge weight = number of co-occurrences.

```python
# stage5_graph.py
import networkx as nx
import orjson
from itertools import combinations
from tqdm import tqdm

EXTRACTIONS_FILE = "data/extractions.jsonl"
ENTITY_MAP_FILE  = "data/entity_map.json"
ENTITY_DB_FILE   = "data/entities_with_desc.json"
GRAPH_FILE       = "data/graph.graphml"

with open(ENTITY_MAP_FILE, "rb") as f:
    entity_map = orjson.loads(f.read())
with open(ENTITY_DB_FILE, "rb") as f:
    entity_db = orjson.loads(f.read())

G = nx.Graph()

# Add nodes
for eid, entity in entity_db.items():
    G.add_node(eid,
               canonical_name=entity["canonical_name"],
               label=entity["label"],
               description=entity.get("description", ""))

# Add co-occurrence edges; enrich with typed relation labels from Stage 2
with open(EXTRACTIONS_FILE, "rb") as f:
    for line in tqdm(f, desc="Building graph"):
        record = orjson.loads(line)

        # --- co-occurrence edges ---
        chunk_entities = set()
        for ent in record["entities"]:
            eid = entity_map.get(ent["text"].strip().lower())
            if eid:
                chunk_entities.add(eid)
        for e1, e2 in combinations(chunk_entities, 2):
            if G.has_edge(e1, e2):
                G[e1][e2]["weight"] += 1
            else:
                G.add_edge(e1, e2, weight=1, relations=set())

        # --- typed relation labels from GLiNER-Relex ---
        for rel in record.get("relations", []):
            src = entity_map.get(rel["head"].strip().lower())
            dst = entity_map.get(rel["tail"].strip().lower())
            if src and dst and G.has_edge(src, dst):
                G[src][dst]["relations"].add(rel["relation"])

# Prune weak edges (co-occurrence of 1 is often noise)
edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < 2]
G.remove_edges_from(edges_to_remove)

# GraphML doesn't support sets — serialise relation labels as pipe-delimited string
for _, _, data in G.edges(data=True):
    data["relations"] = "|".join(sorted(data["relations"]))

nx.write_graphml(G, GRAPH_FILE)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
```

**Notes:**
- `graphml` is a portable format readable by Gephi, Cytoscape, and igraph. For very large graphs, use `nx.write_gpickle` for speed, or convert to igraph directly in Stage 6.
- Each edge now carries a `relations` attribute — a pipe-delimited string of all GLiNER-Relex relation types observed between that entity pair across chunks (e.g. `"developed|part of"`). Stage 7 uses this to build relational triples for the community summarization prompt.

---

## Stage 6 — Leiden Community Detection

```python
# stage6_leiden.py
import networkx as nx
import igraph as ig
import leidenalg as la
import orjson

GRAPH_FILE      = "data/graph.graphml"
COMMUNITY_FILE  = "data/communities.json"
N_ITERATIONS    = 10    # run multiple times, keep best modularity

G_nx = nx.read_graphml(GRAPH_FILE)

# Convert NetworkX -> igraph
# leidenalg works natively on igraph objects
G_ig = ig.Graph.from_networkx(G_nx)
weights = [G_nx[u][v]["weight"] for u, v in G_nx.edges()]

best_partition = None
best_modularity = -1.0
for i in range(N_ITERATIONS):
    partition = la.find_partition(
        G_ig,
        la.ModularityVertexPartition,
        weights=weights,
        seed=i
    )
    mod = partition.modularity
    if mod > best_modularity:
        best_modularity = mod
        best_partition = partition

print(f"Best modularity: {best_modularity:.4f}, Communities: {len(best_partition)}")

# Map back to NetworkX node IDs
node_ids = list(G_nx.nodes())
community_map: dict[str, int] = {
    node_ids[i]: best_partition.membership[i]
    for i in range(len(node_ids))
}

# Group by community
from collections import defaultdict
communities: dict[int, list[str]] = defaultdict(list)
for node_id, comm_id in community_map.items():
    communities[comm_id].append(node_id)

with open(COMMUNITY_FILE, "wb") as f:
    f.write(orjson.dumps({
        "membership": community_map,
        "communities": {str(k): v for k, v in communities.items()},
        "modularity": best_modularity
    }))
```

**Notes:**
- `leidenalg` requires `igraph` (C-backed). The `ig.Graph.from_networkx()` conversion is straightforward but may be slow for very large graphs (>1M nodes); prefer building igraph directly from edge lists.
- Running `N_ITERATIONS=10` and keeping the best modularity is standard practice since Leiden is non-deterministic.
- For hierarchical communities (as in the original GraphRAG), run Leiden at multiple resolutions using `la.RBConfigurationVertexPartition` with varying `resolution_parameter` values.

---

## Stage 7 — Community Summarization (Local LLM)

```python
# stage7_summarize.py
import ollama
import networkx as nx
import orjson
from tqdm import tqdm

COMMUNITY_FILE  = "data/communities.json"
ENTITY_DB_FILE  = "data/entities_with_desc.json"
GRAPH_FILE      = "data/graph.graphml"
SUMMARY_FILE    = "data/community_summaries.json"

MODEL = "qwen2.5:32b-instruct-q4_K_M"
MAX_ENTITIES_PER_SUMMARY  = 20
MAX_RELATIONS_PER_SUMMARY = 20   # cap relation triples to stay within context

with open(COMMUNITY_FILE, "rb") as f:
    community_data = orjson.loads(f.read())
with open(ENTITY_DB_FILE, "rb") as f:
    entity_db = orjson.loads(f.read())
G = nx.read_graphml(GRAPH_FILE)

SUMMARY_PROMPT = """\
You are a knowledge graph analyst. Below are entities and key relationships from a coherent thematic community in a knowledge graph.

Write a 3-5 sentence summary of what this community represents — its central theme, the key entities, and how they relate.

Entities:
{entity_list}

Key relations:
{relation_list}

Community summary:"""

client = ollama.Client()
summaries: dict[str, str] = {}

communities = community_data["communities"]
for comm_id, member_ids in tqdm(communities.items(), desc="Summarizing communities"):
    sample = member_ids[:MAX_ENTITIES_PER_SUMMARY]

    # Build entity descriptions
    entity_lines = []
    for eid in sample:
        entity = entity_db.get(eid)
        if entity:
            entity_lines.append(
                f"- {entity['canonical_name']} ({entity['label']}): {entity.get('description', 'No description.')}"
            )
    if not entity_lines:
        summaries[comm_id] = "Empty community."
        continue

    # Build typed relation triples between sampled members from graph edges
    relation_lines = []
    for i, eid1 in enumerate(sample):
        for eid2 in sample[i + 1:]:
            if G.has_edge(eid1, eid2):
                rel_str = G[eid1][eid2].get("relations", "")
                if rel_str:
                    n1 = entity_db.get(eid1, {}).get("canonical_name", eid1)
                    n2 = entity_db.get(eid2, {}).get("canonical_name", eid2)
                    for rel in rel_str.split("|"):
                        relation_lines.append(f"- {n1} — {rel} — {n2}")
                        if len(relation_lines) >= MAX_RELATIONS_PER_SUMMARY:
                            break
            if len(relation_lines) >= MAX_RELATIONS_PER_SUMMARY:
                break

    prompt = SUMMARY_PROMPT.format(
        entity_list="\n".join(entity_lines),
        relation_list="\n".join(relation_lines) or "No explicit relations extracted.",
    )
    response = client.generate(
        model=MODEL,
        prompt=prompt,
        options={"temperature": 0.2, "num_predict": 200}
    )
    summaries[comm_id] = response["response"].strip()

with open(SUMMARY_FILE, "wb") as f:
    f.write(orjson.dumps(summaries))
```

---

## Stage 8 — Embedding & Indexing (ChromaDB)

Two collections: one for entity descriptions (local search), one for community summaries (global search).

```python
# stage8_index.py
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import orjson
from tqdm import tqdm

ENTITY_DB_FILE = "data/entities_with_desc.json"
SUMMARY_FILE   = "data/community_summaries.json"
CHROMA_PATH    = "data/chroma_db"

embed_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2",   # 768-dim, better quality than MiniLM for retrieval
    device="cuda"                      # use GPU for embedding
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

# --- Entity collection (local search) ---
entity_col = client.get_or_create_collection(
    name="entities",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)

with open(ENTITY_DB_FILE, "rb") as f:
    entity_db = orjson.loads(f.read())

BATCH = 256
items = list(entity_db.items())
for i in tqdm(range(0, len(items), BATCH), desc="Indexing entities"):
    batch = items[i:i+BATCH]
    ids, docs, metas = [], [], []
    for eid, entity in batch:
        desc = entity.get("description", entity["canonical_name"])
        ids.append(eid)
        docs.append(desc)
        metas.append({"label": entity["label"], "name": entity["canonical_name"]})
    entity_col.upsert(ids=ids, documents=docs, metadatas=metas)

# --- Community collection (global search) ---
comm_col = client.get_or_create_collection(
    name="communities",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)

with open(SUMMARY_FILE, "rb") as f:
    summaries = orjson.loads(f.read())

comm_items = list(summaries.items())
for i in tqdm(range(0, len(comm_items), BATCH), desc="Indexing communities"):
    batch = comm_items[i:i+BATCH]
    ids = [f"comm_{cid}" for cid, _ in batch]
    docs = [summary for _, summary in batch]
    comm_col.upsert(ids=ids, documents=docs)

print("Indexing complete.")
```

---

## Stage 9 — Query

### Local Search

```python
# query_local.py
import chromadb
import networkx as nx
import orjson
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH    = "data/chroma_db"
ENTITY_DB_FILE = "data/entities_with_desc.json"
GRAPH_FILE     = "data/graph.graphml"

embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2", device="cuda")
chroma   = chromadb.PersistentClient(path=CHROMA_PATH)
entity_col = chroma.get_collection("entities", embedding_function=embed_fn)

with open(ENTITY_DB_FILE, "rb") as f:
    entity_db = orjson.loads(f.read())
G = nx.read_graphml(GRAPH_FILE)

def local_search(query: str, top_k_entities: int = 5, hop: int = 1) -> dict:
    # Step 1: find seed entities via vector similarity
    results = entity_col.query(query_texts=[query], n_results=top_k_entities)
    seed_ids = results["ids"][0]

    # Step 2: expand via graph neighborhood
    context_entities = set(seed_ids)
    for eid in seed_ids:
        if eid in G:
            neighbors = list(G.neighbors(eid))
            # Weight neighbors by edge weight, take top N
            neighbors_weighted = sorted(
                neighbors, key=lambda n: G[eid][n].get("weight", 1), reverse=True
            )
            context_entities.update(neighbors_weighted[:10])

    # Step 3: assemble context
    context = []
    for eid in context_entities:
        entity = entity_db.get(eid)
        if entity:
            context.append(f"{entity['canonical_name']} ({entity['label']}): {entity.get('description', '')}")

    return {"query": query, "context_entities": list(context_entities), "context": "\n".join(context)}
```

### Global Search

```python
# query_global.py
import chromadb
import ollama
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH = "data/chroma_db"
MODEL       = "llama3.1:8b-instruct-q4_K_M"

embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2", device="cuda")
chroma   = chromadb.PersistentClient(path=CHROMA_PATH)
comm_col = chroma.get_collection("communities", embedding_function=embed_fn)
ollama_client = ollama.Client()

def global_search(query: str, top_k_communities: int = 10) -> str:
    results = comm_col.query(query_texts=[query], n_results=top_k_communities)
    community_summaries = results["documents"][0]

    synthesis_prompt = f"""You are answering a question using summaries from a knowledge graph.

Question: {query}

Relevant community summaries:
{chr(10).join(f"- {s}" for s in community_summaries)}

Answer the question based on the information in the summaries above. Be concise and factual."""

    response = ollama_client.generate(
        model=MODEL, prompt=synthesis_prompt,
        options={"temperature": 0.1, "num_predict": 400}
    )
    return response["response"].strip()
```

---

## Known Problems & Mitigations

| Problem | Severity | Mitigation |
|---|---|---|
| **VRAM contention** between GLiNER and Ollama | High | Run stages sequentially; never run both on GPU simultaneously. GLiNER on CPU is fine. |
| **Entity description generation is extremely slow** at Wikipedia scale | High | Subsample corpus to a domain. Skip descriptions for entities with <3 source chunks (use template fallback). |
| **Coreference & entity merging quality** with 7–8B LLM is worse than GPT-4 | Medium | Use embedding dedup (Stage 3) + string normalization aggressively before involving LLM. |
| **leidenalg C extension build failures** on some Linux distros | Medium | Use conda (`conda install -c conda-forge leidenalg`) rather than pip; pre-built wheels available. |
| **GLiNER performance degradation** with >30 entity or relation types | Low-Medium | Keep each label list under 20 types. Use the bi-encoder variant for larger type sets. |
| **Quadratic RE pair enumeration** with GLiNER-Relex on entity-dense chunks | Low-Medium | Set `MAX_ENTITIES_PER_CHUNK`; at 150 words/chunk, entity density is typically low enough that the cap is never hit in practice. |
| **ChromaDB HNSW index size** for millions of entities | Medium | ChromaDB is fine up to ~5M vectors locally; beyond that consider switching to Qdrant or Weaviate with on-disk indexes. |
| **Models <7B produce malformed outputs** | Medium | Add JSON output schema and response validation; retry with temperature=0 on failure. |
| **Graph size** for full Wikipedia (potentially 100M+ edges) | High | Prune edges with weight < 2 (Stage 5). Further: restrict to top-k neighbors per node, or process Wikipedia in domain-specific subsets. |

---

## Recommended Experiment Scope

For a first working prototype on the 4060, don't use all of Wikipedia. A tractable scope:

- **One Wikipedia category** (e.g. "Infectious diseases", ~5,000 articles) → ~50,000 chunks
- GLiNER extraction: ~1–2 hours on CPU
- Description generation: ~12–24 hours on GPU (Ollama)
- Community summarization: ~1–2 hours
- Full pipeline end-to-end: feasible in 1–2 days

This is enough to validate the pipeline and tune thresholds before scaling.
