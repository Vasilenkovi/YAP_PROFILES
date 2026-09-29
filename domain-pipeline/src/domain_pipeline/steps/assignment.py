from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from domain_pipeline.steps.common import read_table, write_table
from domain_pipeline.steps.ilc import collect_centroid_metadata_from_tree

if TYPE_CHECKING:
    from domain_pipeline.pipeline import PipelineContext


def domain_assignment(context: PipelineContext, cfg: dict) -> None:
    items_ref = context.artifacts.get(cfg["input_items"])
    embeddings_ref = context.artifacts.get(cfg["input_embeddings"])
    centroids_ref = context.artifacts.get(cfg["centroids_embeddings"])
    centroids_meta_ref = context.artifacts.get(cfg["centroids_metadata"])

    df = read_table(items_ref)
    embeddings = np.load(embeddings_ref.path)
    centroids = np.load(centroids_ref.path)
    centroids_meta = _load_centroids_metadata(context, cfg, centroids_meta_ref)

    if len(df) != embeddings.shape[0]:
        raise ValueError(
            f"Row mismatch: {items_ref.name} rows={len(df)}, "
            f"{embeddings_ref.name} rows={embeddings.shape[0]}"
        )
    if embeddings.shape[1] != centroids.shape[1]:
        raise ValueError(f"Embedding dimensions differ: {embeddings.shape[1]} != {centroids.shape[1]}")
    if len(centroids_meta) != centroids.shape[0]:
        raise ValueError(
            f"Centroid metadata mismatch: rows={len(centroids_meta)}, embeddings={centroids.shape[0]}"
        )

    if cfg.get("scaler", "standard") == "standard":
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        centroids_scaled = scaler.transform(centroids)
    else:
        embeddings_scaled = embeddings
        centroids_scaled = centroids

    kmeans_cfg = cfg.get("kmeans", {})
    kmeans = KMeans(
        n_clusters=len(centroids_scaled),
        init=centroids_scaled,
        n_init=int(kmeans_cfg.get("n_init", 1)),
        max_iter=int(kmeans_cfg.get("max_iter", 1)),
        random_state=int(kmeans_cfg.get("random_state", 42)),
    )
    labels = kmeans.fit_predict(embeddings_scaled)
    distance_metric = cfg.get("distance", {}).get("metric", "cosine")
    distances = pairwise_distances(embeddings_scaled, centroids_scaled, metric=distance_metric)

    cols = cfg.get("output_columns", {})
    cluster_col = cols.get("cluster", "ilc_cluster")
    notation_col = cols.get("notation", "ilc_notation")
    label_col = cols.get("label", "ilc_label")
    distance_col = cols.get("distance", "min_distance_to_ilc_center")

    df[cluster_col] = labels
    df[distance_col] = distances.min(axis=1)
    df[notation_col] = df[cluster_col].map(lambda i: centroids_meta.iloc[int(i)]["notation"])
    df[label_col] = df[cluster_col].map(lambda i: centroids_meta.iloc[int(i)]["label"])

    output_path = context.artifacts.output_path(cfg["output_artifact"], ".csv")
    output_ref = context.artifacts.register(cfg["output_artifact"], "table", output_path, cfg)
    write_table(df, output_path)
    context.artifacts.write_metadata(
        output_ref,
        {
            "row_count": len(df),
            "source_artifact": items_ref.name,
            "input_embeddings": embeddings_ref.name,
            "centroids_embeddings": centroids_ref.name,
            "centroids_count": int(centroids.shape[0]),
        },
    )


def _load_centroids_metadata(context: PipelineContext, cfg: dict, centroids_meta_ref) -> pd.DataFrame:
    try:
        return read_table(centroids_meta_ref)
    except Exception as exc:
        fallback_tree = cfg.get("centroids_metadata_fallback_tree")
        if not fallback_tree:
            raise
        tree_ref = context.artifacts.get(fallback_tree)
        target_depth = int(cfg.get("centroids_target_depth", 4))
        descendant_depth = int(cfg.get("centroids_descendant_depth", 3))
        print(
            f"warning: cannot read {centroids_meta_ref.path}; "
            f"rebuilding centroid metadata from {tree_ref.path}. Original error: {exc}"
        )
        return collect_centroid_metadata_from_tree(tree_ref.path, target_depth, descendant_depth)
