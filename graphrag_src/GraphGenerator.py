import logging
from pathlib import Path
from typing import Dict, Iterator, List

import igraph as ig
import pyarrow.parquet as pq
import polars as pl
import chromadb
from FlagEmbedding import BGEM3FlagModel 
import psutil
from datasets import load_dataset
import os
from tqdm import tqdm



try:
    from .EntityRelationExtractor import EntityRelationExtractor
    from  . import DescriptionGenerator as dg
except ImportError:
    from EntityRelationExtractor import EntityRelationExtractor
    import DescriptionGenerator as dg


class GraphGenerator:
    
    ENTITIES_PATH="data/entities_with_descriptions"
    RELATIONS_PATH="data/relations"
    COMMUNITIES_PATH="data/communities/"
    COMMUNITIES_DESC_PATH="data/communities_with_descriptions"
    CHROMA_PATH="data/chroma_db"
    ENTITIY_CHROMA_NAME="entity_embeddings"
    COMMUNITIES_CHROMA_NAME="community_embeddings"
    
    CHUNKS_PATH="data/chunks_arrow"
    
    def __init__(
        self,
        ner_model_name=None,
        embed_model_name=None,
        description_model_name=None,
        dataset_path="wiki_dpr",
        max_concurrent: int = 8,
        #TODO implement different models
        
        ) -> None:
        self.extractor=EntityRelationExtractor(chunks_path=self.CHUNKS_PATH)
        self.desc_model=description_model_name or "llama3.1:8b-instruct-q4_K_M"
        self.embed_model=embed_model_name or "BAAI/bge-m3"
        self.logger = logging.getLogger(__name__)
        if dataset_path=="wiki_dpr":
            ds = load_dataset("facebook/wiki_dpr", name="psgs_w100.nq.no_index.no_embeddings", split="train")
            self.CHUNKS_PATH=str(Path(ds.cache_files[0]["filename"]).parent)
        self.max_concurrent=max_concurrent


    @staticmethod
    def _format_rss_mb() -> float:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    
    
    
    def _load_parquet_frame(self, path: str, columns: List[str]) -> pl.DataFrame:
        schema = set(pl.read_parquet_schema(path).keys())
        missing_columns = [column for column in columns if column not in schema]
        if missing_columns:
            raise KeyError(f"{path} is missing required columns: {missing_columns}")
        return pl.read_parquet(path, columns=columns)

    def _save_inplace(self, df: pl.DataFrame, path: str) -> None:
        """Save a Polars DataFrame to a Parquet file, overwriting the original."""
        temp_path = f"{path}.tmp"
        try:
            df.write_parquet(temp_path, compression="zstd")
            os.replace(temp_path, path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
    

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
            self._load_parquet_frame(relation_db_path, ["head_id", "tail_id", "relation"])
            .rename({"head_id": "head", "tail_id": "tail"})
            .group_by(["head", "tail"])
            .agg(
                pl.len().alias("weight"),
                pl.col("relation").drop_nulls().unique().sort().alias("relations"),
                pl.col("relation").drop_nulls().n_unique().alias("relation_count"),
            )
            .with_columns(pl.col("relations").list.join("|"))
        )
        self.logger.info(
            "Prepared edge frame: rows=%d cols=%d rss=%.1f MB",
            edge_frame.height,
            edge_frame.width,
            self._format_rss_mb(),
        )
        #save grouped relations with weights
        self._save_inplace(edge_frame, relation_db_path)
        
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
    
    def _save_joined_column(
        self,
        path: str,
        join_df: pl.DataFrame,
        left_on: str,
        right_on: str,
        value_col: str,
        new_column_name: str,
    ) -> None:
        """Join one column from ``join_df`` into the parquet file at ``path``.

        New schema after clustering:

            pa.schema([
                pa.field("id",             pa.string()),
                pa.field("canonical_name", pa.string()),
                pa.field("label",          pa.string()),
                pa.field("description",    pa.string()),
                pa.field("cluster_id",     pa.int64()),
            ])

        The file is rewritten atomically by writing to a temporary parquet file
        first and then replacing the original on success.
        """
        lazy_df = pl.scan_parquet(path).join(
            join_df.lazy().select([right_on, value_col]).rename({value_col: new_column_name}),
            left_on=left_on,
            right_on=right_on,
            how="left",
        )
        temp_path = f"{path}.tmp"
        try:
            lazy_df.sink_parquet(temp_path)
            os.replace(temp_path, path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
    
    def _compute_clusters(self, graph: ig.Graph) -> None:
        self.logger.info("Computing clusters: rss=%.1f MB", self._format_rss_mb())
        
        clustering = graph.community_leiden(
            objective_function="modularity", # switch to CPM
            n_iterations=2,
            weights=graph.es["weight"] if "weight" in graph.es.attributes() else None,
        )
        
        entity_cluster_df = pl.DataFrame({
            "id": graph.vs["name"],
            "cluster_id": clustering.membership,
        })
        

        self._save_joined_column(
            path=self.ENTITIES_PATH,
            join_df=entity_cluster_df,
            left_on="id",
            right_on="id",
            value_col="cluster_id",
            new_column_name="cluster_id",
        )
        self.logger.info("Clusters computed and saved in %s", self.ENTITIES_PATH)
        
    def _relations_for_community(self, entity_path: str, relation_path: str, communities_path: str):
        for cluster_id in pl.scan_parquet(entity_path).select("cluster_id").unique().collect().to_series():
            members = (
                pl.scan_parquet(entity_path)
                .filter(pl.col("cluster_id") == cluster_id)
                .select("id")
            )
            relations_for_community = (
                pl.scan_parquet(relation_path)
                .join(members.rename({"id": "head"}), on="head", how="inner")
                .join(members.rename({"id": "tail"}), on="tail", how="inner")
                )
            os.makedirs(communities_path, exist_ok=True)
            relations_for_community.sink_parquet(f"{communities_path}_community_{cluster_id}_relations.parquet")
        self.logger.info("Graph generation completed successfully.")
    
    
    def _iter_parquet_batches(self, path: str, columns: List[str], batch_size: int) -> Iterator[Dict[str, list]]:
        pf = pq.ParquetFile(path)
        schema_columns = set(pf.schema_arrow.names)
        missing = [c for c in columns if c not in schema_columns]
        if missing:
            raise KeyError(f"{path} is missing required columns: {missing}")
        for record_batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            yield record_batch.to_pydict()
        
    def _embed_entities(self, path: str):
        client = chromadb.PersistentClient(self.CHROMA_PATH)
        collection = client.get_or_create_collection(name="entity_embeddings", metadata={"hnsw:space": "cosine"},)
        model=BGEM3FlagModel(self.embed_model, use_fp16=True)

        total_rows = pq.ParquetFile(path).metadata.num_rows
        total_batches = (total_rows + 256 - 1) // 256
        self.logger.info(
            "Starting entity embedding: path=%s rows=%d batches=%d collection=%s",
            path,
            total_rows,
            total_batches,
            self.ENTITIY_CHROMA_NAME,
        )

        total_embedded = 0
        with tqdm(total=total_batches, desc="Embedding entities") as pbar:
            for batch in self._iter_parquet_batches(path, columns=["id", "canonical_name", "description", "cluster_id"], batch_size=256):
                ids = batch["id"]
                names = batch["canonical_name"]
                descriptions = batch["description"]
                cluster_ids = batch["cluster_id"]

                texts = [
                    f"{name}: {description}" if description else name
                    for name, description in zip(names, descriptions)
                ]

                valid_idx = [i for i, text in enumerate(texts) if text]
                if valid_idx:
                    valid_ids = [ids[i] for i in valid_idx]
                    valid_texts = [texts[i] for i in valid_idx]
                    valid_metadata = [{"cluster_id": cluster_ids[i] if cluster_ids[i] is not None else -1} for i in valid_idx]

                    embeddings = model.encode(
                        valid_texts,
                        batch_size=len(valid_texts),
                        max_length=128,
                        return_dense=True,
                        return_sparse=False,
                        return_colbert_vecs=False,
                    )["dense_vecs"].tolist()

                    collection.upsert(
                        ids=valid_ids,
                        embeddings=embeddings,
                        metadatas=valid_metadata,
                        documents=valid_texts,
                    )
                    total_embedded += len(valid_ids)

                pbar.update(1)
        
        self.logger.info(
            "Entity embedding complete: total=%d saved to %s (collection=%s)",
            total_embedded,
            self.CHROMA_PATH,
        )
        
    def _embed_communities(self, path: str):
        client = chromadb.PersistentClient(self.CHROMA_PATH)
        collection = client.get_or_create_collection(name=self.COMMUNITIES_CHROMA_NAME, metadata={"hnsw:space": "cosine"})
        model = BGEM3FlagModel(self.embed_model, use_fp16=True)

        total_rows = pq.ParquetFile(path).metadata.num_rows
        total_batches = (total_rows + 64 - 1) // 64
        self.logger.info(
            "Starting community embedding: path=%s rows=%d batches=%d collection=%s",
            path,
            total_rows,
            total_batches,
            self.COMMUNITIES_CHROMA_NAME,
        )

        total_embedded = 0
        with tqdm(total=total_batches, desc="Embedding communities") as pbar:
            for batch in self._iter_parquet_batches(path, columns=["cluster_id", "entities", "description"], batch_size=64):
                ids = batch["cluster_id"]
                texts = batch["description"]

                valid_idx = [i for i, text in enumerate(texts) if text]
                if valid_idx:
                    valid_ids = [str(ids[i]) for i in valid_idx]
                    valid_texts = [texts[i] for i in valid_idx]

                    embeddings = model.encode(
                        valid_texts,
                        batch_size=len(valid_texts),
                        max_length=1024,  # num_predict=1000 needs real headroom, unlike entities' 128
                        return_dense=True,
                        return_sparse=False,
                        return_colbert_vecs=False,
                    )["dense_vecs"].tolist()

                    collection.upsert(
                        ids=valid_ids,
                        embeddings=embeddings,
                        documents=valid_texts,
                    )
                    total_embedded += len(valid_ids)

                pbar.update(1)

        self.logger.info(
            "Community embedding complete: total=%d saved to %s (collection=%s)",
            total_embedded,
            self.CHROMA_PATH,
            self.COMMUNITIES_CHROMA_NAME,
        )

    async def generate_graph(self) -> None:
        
        # Generate entities and relations from the dataset
        extractor = self.extractor
        extractor.generate()
        entity_path=extractor.ENTITIES_PATH
        relation_path=extractor.RELATIONS_PATH
        del extractor
        
        #Generate entity descriptionjs
        generator = dg.EntityDescriptionGenerator(self.CHUNKS_PATH, entity_path, self.ENTITIES_PATH, self.logger, model=self.desc_model, max_concurrent=self.max_concurrent)
        await generator.generate_descriptions()
        
        self._embed_entities(self.ENTITIES_PATH)
        
        # Generate the graph from the entities and relations
        graph = self._generate_graph(self.ENTITIES_PATH, relation_path)
        self._compute_clusters(graph)
        
        # Create helper relation table for each community
        self._relations_for_community(self.ENTITIES_PATH, relation_path, self.COMMUNITIES_PATH)
        
        #Generate community descriptions
        community_generator = dg.CommunityDescriptionGenerator(self.COMMUNITIES_PATH, self.ENTITIES_PATH, self.COMMUNITIES_DESC_PATH, self.logger, model=self.desc_model, max_concurrent=self.max_concurrent)
        await community_generator.generate_descriptions()
        self._embed_communities(self.COMMUNITIES_PATH)
