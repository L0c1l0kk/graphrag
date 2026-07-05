"""Print lightweight random samples from generated graph parquet files.

This is intended as a quick smoke test after running the pipeline.
It samples:
- random entity rows from the enriched entity parquet
- random relation rows from the relation parquet
- random communities based on cluster ids in the entity parquet, plus a few
  community-specific relations when those files exist
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import polars as pl


DEFAULT_ENTITIES_PATH = Path("data/entities_with_descriptions")
DEFAULT_RELATIONS_PATH = Path("data/relations")
DEFAULT_COMMUNITIES_DIR = Path("data/communities")
DEFAULT_COMMUNITIES_DESC_PATH = Path("data/communities_with_descriptions")


def _read_frame(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    return pl.read_parquet(path)


def _sample_rows(frame: pl.DataFrame, count: int, seed: int | None) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    count = min(count, frame.height)
    if count <= 0:
        return frame.head(0)
    return frame.sample(n=count, shuffle=True, seed=seed)


def _print_rows(title: str, frame: pl.DataFrame) -> None:
    print(f"\n{title} ({frame.height} rows)")
    if frame.is_empty():
        print("  <no rows>")
        return
    for index, row in enumerate(frame.to_dicts(), start=1):
        print(f"[{index}] {json.dumps(row, indent=2, ensure_ascii=False, default=str)}")


def _print_parquet_summary(label: str, path: Path, frame: pl.DataFrame) -> None:
    print(f"{label}: {path} -> {frame.height} rows")


def _discover_community_files(communities_dir: Path) -> list[Path]:
    files: list[Path] = []

    prefixed_pattern = re.compile(rf"^{re.escape(communities_dir.name)}_community_(\d+)_relations\.parquet$")
    for path in communities_dir.parent.glob(f"{communities_dir.name}_community_*_relations.parquet"):
        if prefixed_pattern.match(path.name):
            files.append(path)

    if communities_dir.exists() and communities_dir.is_dir():
        directory_pattern = re.compile(r"^_community_(\d+)_relations\.parquet$")
        for path in communities_dir.glob("_community_*_relations.parquet"):
            if directory_pattern.match(path.name):
                files.append(path)

    return sorted(set(files))


def _cluster_id_from_path(path: Path) -> int | None:
    match = re.search(r"_community_(\d+)_relations\.parquet$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print random samples from the generated graph parquet files."
    )
    parser.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES_PATH)
    parser.add_argument("--relations", type=Path, default=DEFAULT_RELATIONS_PATH)
    parser.add_argument("--communities-dir", type=Path, default=DEFAULT_COMMUNITIES_DIR)
    parser.add_argument("--communities-desc-dir", type=Path, default=DEFAULT_COMMUNITIES_DESC_PATH)
    parser.add_argument("--n-entities", type=int, default=5)
    parser.add_argument("--n-relations", type=int, default=5)
    parser.add_argument("--n-communities", type=int, default=3)
    parser.add_argument("--n-community-relations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    entities = _read_frame(args.entities)
    relations = _read_frame(args.relations)
    communities_desc = _read_frame(args.communities_desc_dir) if args.communities_desc_dir.exists() else None

    _print_parquet_summary("Entities parquet", args.entities, entities)
    _print_parquet_summary("Relations parquet", args.relations, relations)
    if communities_desc is not None:
        _print_parquet_summary("Communities descriptions parquet", args.communities_desc_dir, communities_desc)
    else:
        print(f"Communities descriptions parquet: {args.communities_desc_dir} -> <missing>")

    entity_sample = _sample_rows(entities, args.n_entities, rng.randint(0, 2**31 - 1) if args.seed is not None else None)
    sampled_entity_ids = entity_sample.select("id").to_series().to_list() if "id" in entity_sample.columns else []

    relation_head_col = "head_id" if "head_id" in relations.columns else "head"
    relation_tail_col = "tail_id" if "tail_id" in relations.columns else "tail"
    related_relations = relations
    if sampled_entity_ids and relation_head_col in relations.columns and relation_tail_col in relations.columns:
        related_relations = relations.filter(
            pl.col(relation_head_col).is_in(sampled_entity_ids)
            | pl.col(relation_tail_col).is_in(sampled_entity_ids)
        )
    relation_sample = _sample_rows(
        related_relations,
        args.n_relations,
        rng.randint(0, 2**31 - 1) if args.seed is not None else None,
    )

    _print_rows("Random entities", entity_sample)
    if sampled_entity_ids:
        print(f"Sampled entity ids: {sampled_entity_ids}")
    print(f"Relations matched to sampled entities: {related_relations.height} rows")
    _print_rows("Random relations involving sampled entities", relation_sample)

    if communities_desc is not None:
        community_id_col = "cluster_id" if "cluster_id" in communities_desc.columns else None
        sampled_communities = _sample_rows(
            communities_desc,
            min(args.n_communities, communities_desc.height),
            rng.randint(0, 2**31 - 1) if args.seed is not None else None,
        )
        _print_rows("Random communities", sampled_communities)

    community_files = _discover_community_files(args.communities_dir)
    if community_files:
        print(f"\nCommunity relation files ({len(community_files)} total)")
        for community_path in rng.sample(community_files, k=min(args.n_communities, len(community_files))):
            cluster_id = _cluster_id_from_path(community_path)
            community_relations = _read_frame(community_path)
            _print_parquet_summary("Community relations parquet", community_path, community_relations)

            print(f"\nCommunity file: {community_path}")
            if cluster_id is not None and "cluster_id" in entities.columns:
                community_entities = entities.filter(pl.col("cluster_id") == cluster_id)
                community_entity_ids = community_entities.select("id").to_series().to_list() if "id" in community_entities.columns else []
                print(f"  cluster_id={cluster_id} entities={community_entities.height}")
                sampled_entities = _sample_rows(
                    community_entities.select([col for col in ["id", "canonical_name", "label", "description", "cluster_id"] if col in community_entities.columns]),
                    min(3, community_entities.height),
                    rng.randint(0, 2**31 - 1) if args.seed is not None else None,
                )
                _print_rows("Community entity sample", sampled_entities)

                if community_entity_ids and relation_head_col in community_relations.columns and relation_tail_col in community_relations.columns:
                    community_related_relations = community_relations.filter(
                        pl.col(relation_head_col).is_in(community_entity_ids)
                        | pl.col(relation_tail_col).is_in(community_entity_ids)
                    )
                    print(f"  relations matched to sampled community entities: {community_related_relations.height} rows")
                    sampled_community_relations = _sample_rows(
                        community_related_relations,
                        args.n_community_relations,
                        rng.randint(0, 2**31 - 1) if args.seed is not None else None,
                    )
                else:
                    sampled_community_relations = _sample_rows(
                        community_relations,
                        args.n_community_relations,
                        rng.randint(0, 2**31 - 1) if args.seed is not None else None,
                    )
                _print_rows("Community relations sample", sampled_community_relations)
            else:
                print("  <cluster_id unavailable in entity parquet or filename>")
    else:
        print("\nCommunities: skipped (no community parquet files found)")


if __name__ == "__main__":
    main()