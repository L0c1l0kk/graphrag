import asyncio
import json
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
import requests
from tqdm import tqdm
import os

try:
    from .EntityRelationExtractor import EntityRelationExtractor
except ImportError:
    from EntityRelationExtractor import EntityRelationExtractor


class GraphGenerator:
    
    ENTITIES_PATH="data/entities_with_descriptions"
    RELATIONS_PATH="data/relations"
    COMMUNITIES_PATH="data/communities/"
    
    CHUNKS_PATH="data/chunks_arrow"
    
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
            .rename({"head_id": "source", "tail_id": "target"})
            .group_by(["source", "target"])
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
        
        n_communities = len(clustering)
        assert set(clustering.membership) == set(range(n_communities)), "unexpected non-contiguous cluster ids"
        
        with open(self.COMMUNITIES_PATH + "_nclusters.json", "w") as f:
            json.dump(n_communities, f)

        self._save_joined_column(
            self,
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

    def get_context_size(self, model: str, host: str = "http://localhost:11434") -> int:
        pass
    
    def _generate_community_descriptions(
        self,
        entity_db_path: str,
        max_concurrent: int = 8,
        chunk_batch_size: int = 256,
        flush_every: int = 500,) -> None:
        
        model = self.desc_model or "llama3.1:8b-instruct-q4_K_M"
        context_size=self.get_context_size(model)
        n_communities = json.load(open(self.ENTITIES_PATH + "_nclusters.json"))
        
        client = ollama.AsyncClient()
        semaphore = asyncio.Semaphore(max_concurrent)
        
        entity_df = pl.scan_parquet(entity_db_path) \
            .select(
                ["id", "canonical_name", "label", "cluster_id", "source_chunks"]
            )
        