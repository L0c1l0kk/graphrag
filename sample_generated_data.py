"""Print lightweight random samples from generated graph parquet files.

This is intended as a quick smoke test after running the pipeline.
It samples:
- random entity rows from the enriched entity parquet
- random relation rows from the relation parquet
- random communities, restricted to communities that actually have
  intracommunity relationships, plus a few relations from within each
  sampled community

Why the "has intracommunity relationships" restriction matters:
GraphGenerator._relations_for_community() writes one
`_community_{cluster_id}_relations.parquet` file per cluster id produced by
Leiden clustering, including clusters that end up with zero internal edges
(e.g. singleton communities, or communities only connected to the rest of
the graph via edges that cross cluster boundaries -- those edges are
dropped since both endpoints must fall in the same cluster). Those files
exist on disk but are empty, and CommunityDescriptionGenerator will still
have generated a description for them from an empty relationship list, so
without filtering the sampler can happily "sample" a community that has no
actual relational content -- not a useful smoke test. This script now
checks each community's row count (cheaply, via Parquet metadata, no data
read) before it's eligible for sampling.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


DEFAULT_ENTITIES_PATH = Path("data/entities_with_descriptions")
DEFAULT_RELATIONS_PATH = Path("data/relations")
DEFAULT_COMMUNITIES_DIR = Path("data/communities")
DEFAULT_COMMUNITIES_DESC_PATH = Path("data/communities_with_descriptions")


def _read_frame(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    return pl.read_parquet(path)


def _parquet_row_count(path: Path) -> int:
    """Row count from the Parquet footer only -- no data actually read.

    Mirrors the metadata-only pattern already used elsewhere in the
    pipeline (see DescriptionGenerator.py) so checking thousands of small
    per-community files for emptiness stays cheap even at wiki_dpr scale.
    """
    return pq.ParquetFile(path).metadata.num_rows


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


def _relation_type_label(row: dict) -> str:
    # Post-_generate_graph relation rows carry the pipe-joined "relations"
    # column (see GraphGenerator._generate_graph); pre-graph raw extraction
    # rows carry a singular "relation" column instead. Handle either.
    return row.get("relations") or row.get("relation") or "related_to"


def _enrich_relations_with_entities(
    relations_frame: pl.DataFrame, entities: pl.DataFrame, head_col: str, tail_col: str
) -> pl.DataFrame:
    """Left-join both relation endpoints against the entity table.

    Adds head_name/head_description/tail_name/tail_description columns so a
    relation row can be printed alongside both entities' full descriptions,
    instead of just their bare ids.
    """
    if relations_frame.is_empty():
        return relations_frame
    lookup_cols = [col for col in ["id", "canonical_name", "description"] if col in entities.columns]
    if "id" not in lookup_cols or head_col not in relations_frame.columns or tail_col not in relations_frame.columns:
        return relations_frame

    entity_lookup = entities.select(lookup_cols)
    enriched = relations_frame.join(
        entity_lookup.rename({"id": head_col, "canonical_name": "head_name", "description": "head_description"}),
        on=head_col,
        how="left",
    )
    enriched = enriched.join(
        entity_lookup.rename({"id": tail_col, "canonical_name": "tail_name", "description": "tail_description"}),
        on=tail_col,
        how="left",
    )
    return enriched


def _format_relation_chain(row: dict, head_col: str, tail_col: str) -> str:
    """Render one enriched relation row as: head: desc --[relation]--> tail: desc"""
    relation_label = _relation_type_label(row)
    head_name = row.get("head_name") or row.get(head_col, "?")
    head_desc = row.get("head_description") or "<no description>"
    tail_name = row.get("tail_name") or row.get(tail_col, "?")
    tail_desc = row.get("tail_description") or "<no description>"
    return f"{head_name}: {head_desc}\n      --[{relation_label}]--> {tail_name}: {tail_desc}"


def _first_nonempty(*frames: pl.DataFrame) -> pl.DataFrame | None:
    for frame in frames:
        if frame is not None and not frame.is_empty():
            return frame
    return None


def _communities_with_intracommunity_relations(
    community_files: list[Path], min_relations: int
) -> tuple[list[Path], list[Path]]:
    """Split discovered community files into (qualifying, empty/too-sparse).

    "Qualifying" means the community's own relations parquet -- which by
    construction only contains edges where both endpoints are members of
    that community -- has at least `min_relations` rows.
    """
    qualifying: list[Path] = []
    skipped: list[Path] = []
    for path in community_files:
        row_count = _parquet_row_count(path)
        (qualifying if row_count >= min_relations else skipped).append(path)
    return qualifying, skipped


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
    parser.add_argument(
        "--min-community-relations",
        type=int,
        default=1,
        help=(
            "Minimum number of intracommunity relations a community must have "
            "to be eligible for sampling (default: 1, i.e. exclude communities "
            "with zero internal edges)."
        ),
    )
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

    # Guarantee at least one relation is shown with BOTH endpoints' full
    # descriptions, not just their bare ids -- prefer one already in
    # relation_sample so it's reproducible under --seed, but fall back to
    # the wider pools so this still fires even in small/sparse datasets.
    highlight_pool = _first_nonempty(relation_sample, related_relations, relations)
    if highlight_pool is not None:
        highlighted = _enrich_relations_with_entities(
            highlight_pool.head(1), entities, relation_head_col, relation_tail_col
        )
        print("\nHighlighted relation (both entity descriptions)")
        row = highlighted.to_dicts()[0]
        print(f"  {_format_relation_chain(row, relation_head_col, relation_tail_col)}")

    # Discover community relation files up front and split them into ones
    # that actually have intracommunity relationships vs. empty/too-sparse
    # ones, so both the community-description sample below and the
    # per-community relation sample only ever draw from qualifying
    # communities.
    all_community_files = _discover_community_files(args.communities_dir)
    qualifying_files, skipped_files = _communities_with_intracommunity_relations(
        all_community_files, args.min_community_relations
    )
    qualifying_cluster_ids = {_cluster_id_from_path(p) for p in qualifying_files} - {None}

    print(
        f"\nCommunities discovered: {len(all_community_files)} total, "
        f"{len(qualifying_files)} with >= {args.min_community_relations} intracommunity "
        f"relation(s), {len(skipped_files)} skipped (empty or too sparse)"
    )

    if communities_desc is not None:
        eligible_desc = communities_desc
        if "cluster_id" in communities_desc.columns and qualifying_cluster_ids:
            # communities_with_descriptions stores cluster_id as a string
            # (see DescriptionGenerator._SCHEMA); compare against string ids.
            qualifying_cluster_id_strs = {str(cid) for cid in qualifying_cluster_ids}
            eligible_desc = communities_desc.filter(
                pl.col("cluster_id").is_in(qualifying_cluster_id_strs)
            )
        sampled_communities = _sample_rows(
            eligible_desc,
            min(args.n_communities, eligible_desc.height),
            rng.randint(0, 2**31 - 1) if args.seed is not None else None,
        )
        _print_rows("Random communities (with intracommunity relations)", sampled_communities)

    if qualifying_files:
        print(f"\nCommunity relation files eligible for sampling ({len(qualifying_files)} total)")
        for community_path in rng.sample(qualifying_files, k=min(args.n_communities, len(qualifying_files))):
            cluster_id = _cluster_id_from_path(community_path)
            community_relations = _read_frame(community_path)
            _print_parquet_summary("Community relations parquet", community_path, community_relations)

            print(f"\nCommunity file: {community_path}")
            if cluster_id is not None and "cluster_id" in entities.columns:
                community_entity_count = entities.filter(pl.col("cluster_id") == cluster_id).height
                print(f"  cluster_id={cluster_id} entities={community_entity_count}")

                # The community relations parquet already IS the intracommunity
                # edge set by construction (GraphGenerator._relations_for_community
                # joins on membership for both head and tail), so we sample
                # directly from it rather than sampling entities and relations
                # separately and then trying to line them up after the fact.
                sampled_community_relations = _sample_rows(
                    community_relations,
                    args.n_community_relations,
                    rng.randint(0, 2**31 - 1) if args.seed is not None else None,
                )
                enriched_community_relations = _enrich_relations_with_entities(
                    sampled_community_relations, entities, relation_head_col, relation_tail_col
                )

                print(f"\nCommunity {cluster_id} relations ({enriched_community_relations.height} sampled)")
                if enriched_community_relations.is_empty():
                    print("  <no rows>")
                else:
                    for index, row in enumerate(enriched_community_relations.to_dicts(), start=1):
                        print(f"  [{index}] {_format_relation_chain(row, relation_head_col, relation_tail_col)}")
            else:
                print("  <cluster_id unavailable in entity parquet or filename>")
    elif all_community_files:
        print(
            "\nCommunities: found community files, but none met the "
            f"--min-community-relations={args.min_community_relations} threshold "
            "(every community appears to be a singleton or has only cross-community edges)"
        )
    else:
        print("\nCommunities: skipped (no community parquet files found)")


if __name__ == "__main__":
    main()
