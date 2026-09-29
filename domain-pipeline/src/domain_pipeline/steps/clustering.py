from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from domain_pipeline.steps.common import read_table, write_table

if TYPE_CHECKING:
    from domain_pipeline.pipeline import PipelineContext


def cluster_embeddings(context: PipelineContext, cfg: dict) -> None:
    items_ref = context.artifacts.get(cfg["input_items"])
    embeddings_ref = context.artifacts.get(cfg["input_embeddings"])
    df = read_table(items_ref)
    embeddings = np.load(embeddings_ref.path)
    if len(df) != embeddings.shape[0]:
        raise ValueError(f"Row mismatch: {len(df)} table rows vs {embeddings.shape[0]} embeddings")

    embeddings_scaled = StandardScaler().fit_transform(embeddings) if cfg.get("scaler", "standard") == "standard" else embeddings
    method = cfg.get("method", "kmeans")

    if method == "dbscan":
        labels = _run_dbscan(embeddings_scaled, cfg.get("dbscan", {}))
        label_column = cfg.get("label_column", "dbscan_cluster")
    elif method == "kmeans":
        labels = _run_kmeans(embeddings_scaled, cfg.get("kmeans", {}))
        label_column = cfg.get("label_column", "kmeans_cluster")
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    df[label_column] = labels
    output_path = context.artifacts.output_path(cfg["output_artifact"], ".csv")
    output_ref = context.artifacts.register(cfg["output_artifact"], "table", output_path, cfg)
    write_table(df, output_path)
    context.artifacts.write_metadata(
        output_ref,
        {
            "row_count": len(df),
            "source_artifact": items_ref.name,
            "input_embeddings": embeddings_ref.name,
            "clusters": int(len(set(labels))),
        },
    )


def _run_dbscan(embeddings: np.ndarray, cfg: dict) -> np.ndarray:
    model = DBSCAN(eps=float(cfg.get("eps", 0.5)), min_samples=int(cfg.get("min_samples", 10)))
    return model.fit_predict(embeddings)


def _run_kmeans(embeddings: np.ndarray, cfg: dict) -> np.ndarray:
    random_state = int(cfg.get("random_state", 42))
    if "n_clusters" in cfg:
        n_clusters = int(cfg["n_clusters"])
    else:
        n_clusters = _select_k_by_silhouette(
            embeddings,
            int(cfg.get("min_k", 2)),
            int(cfg.get("max_k", 20)),
            random_state,
        )
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    return model.fit_predict(embeddings)


def _select_k_by_silhouette(embeddings: np.ndarray, min_k: int, max_k: int, random_state: int) -> int:
    best_k = min_k
    best_score = -1.0
    for k in range(min_k, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=random_state).fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_k = k
            best_score = score
    return best_k
