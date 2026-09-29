from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from domain_pipeline.config import resolve_path

if TYPE_CHECKING:
    from domain_pipeline.pipeline import PipelineContext

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda iterable, **_: iterable


def build_ilc_tree(context: PipelineContext, cfg: dict) -> None:
    import requests
    from rdflib import Graph
    from rdflib.namespace import RDF, SKOS

    ttl_path = resolve_path(context.config, cfg["local_ttl_path"])
    ttl_path.parent.mkdir(parents=True, exist_ok=True)
    if not ttl_path.exists():
        response = requests.get(cfg["source_url"], timeout=60)
        response.raise_for_status()
        ttl_path.write_bytes(response.content)

    graph = Graph()
    graph.parse(ttl_path, format="turtle")
    nodes = _read_ilc_nodes(graph, RDF, SKOS)
    root_notations = _root_notations(cfg.get("root_notations", "a-y"))
    root_uris = [uri for uri, data in nodes.items() if data["notation"] in root_notations]
    max_depth = int(cfg.get("max_depth", 3))
    tree = [
        _build_tree(nodes, uri, depth=1, max_depth=max_depth, root_notations=root_notations)
        for uri in sorted(root_uris, key=lambda u: nodes[u]["notation"])
    ]

    output_path = context.artifacts.output_path(cfg["output_artifact"], ".json")
    output_ref = context.artifacts.register(cfg["output_artifact"], "json", output_path, cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
    context.artifacts.write_metadata(output_ref, {"root_count": len(tree), "max_depth": max_depth})


def build_ilc_centroids(context: PipelineContext, cfg: dict) -> None:
    from sentence_transformers import SentenceTransformer

    tree_ref = context.artifacts.get(cfg["input_tree"])
    target_depth = int(cfg.get("target_depth", 4))
    descendant_depth = int(cfg.get("descendant_depth", 3))
    df = collect_centroid_metadata_from_tree(
        tree_ref.path,
        target_depth,
        descendant_depth,
        cfg.get("text_parts", ["description", "label", "descendant_labels"]),
    )

    vector_cfg = cfg.get("vectorizer", {})
    model_name = vector_cfg.get("model", "all-mpnet-base-v2")
    batch_size = int(vector_cfg.get("batch_size", 32))
    model = SentenceTransformer(model_name)
    embeddings = []
    for start in tqdm(range(0, len(df), batch_size), desc="ILC centroids"):
        batch = df["combined_text"].fillna("").astype(str).tolist()[start : start + batch_size]
        embeddings.extend(model.encode(batch, show_progress_bar=False))
    array = np.asarray(embeddings)

    emb_path = context.artifacts.output_path(cfg["output_embeddings_artifact"], ".npy")
    meta_path = context.artifacts.output_path(cfg["output_metadata_artifact"], ".pkl")
    emb_ref = context.artifacts.register(cfg["output_embeddings_artifact"], "embeddings", emb_path, cfg)
    meta_ref = context.artifacts.register(cfg["output_metadata_artifact"], "table_pickle", meta_path, cfg)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, array)
    df.to_pickle(meta_path)
    context.artifacts.write_metadata(emb_ref, {"shape": list(array.shape), "target_depth": target_depth})
    context.artifacts.write_metadata(meta_ref, {"row_count": len(df), "target_depth": target_depth})


def collect_centroid_metadata_from_tree(
    tree_path: str | Path,
    target_depth: int,
    descendant_depth: int,
    text_parts: list[str] | None = None,
) -> pd.DataFrame:
    with Path(tree_path).open("r", encoding="utf-8") as fh:
        tree = json.load(fh)
    nodes = []
    for root in tree:
        nodes.extend(_collect_nodes_at_depth(root, 1, target_depth))

    parts = text_parts or ["description", "label", "descendant_labels"]
    rows = []
    for node in nodes:
        descendant_labels = _collect_descendant_labels(node, 0, descendant_depth)
        text = _combine_ilc_text(node, descendant_labels, parts)
        rows.append({"notation": node.get("notation"), "label": node.get("label"), "combined_text": text})
    return pd.DataFrame(rows)


def _read_ilc_nodes(graph, RDF, SKOS):
    nodes = {}
    for concept in graph.subjects(RDF.type, SKOS.Concept):
        uri = str(concept)
        notation = next((str(o) for o in graph.objects(concept, SKOS.notation)), None)
        if not notation:
            continue
        label = next((str(o) for o in graph.objects(concept, SKOS.prefLabel)), notation)
        definition = next((str(o) for o in graph.objects(concept, SKOS.definition)), "")
        nodes[uri] = {"uri": uri, "notation": notation, "label": label, "description": definition, "children": set()}
    for concept, broader in graph.subject_objects(SKOS.broader):
        child_uri = str(concept)
        parent_uri = str(broader)
        if child_uri in nodes and parent_uri in nodes:
            nodes[parent_uri]["children"].add(child_uri)
    return nodes


def _root_notations(value) -> set[str]:
    if value == "a-y":
        return {chr(c) for c in range(ord("a"), ord("y") + 1)}
    return set(value)


def _build_tree(nodes, uri: str, depth: int, max_depth: int, root_notations: set[str]):
    data = nodes[uri]
    node = {
        "notation": data["notation"],
        "label": data["label"],
        "description": data["description"],
        "children": [],
    }
    if depth < max_depth:
        for child_uri in sorted(data["children"]):
            if nodes[child_uri]["notation"] in root_notations and depth > 1:
                continue
            node["children"].append(_build_tree(nodes, child_uri, depth + 1, max_depth, root_notations))
    return node


def _collect_nodes_at_depth(node: dict, current_depth: int, target_depth: int) -> list[dict]:
    if current_depth == target_depth:
        return [node]
    if current_depth > target_depth:
        return []
    result = []
    for child in node.get("children", []):
        result.extend(_collect_nodes_at_depth(child, current_depth + 1, target_depth))
    return result


def _collect_descendant_labels(node: dict, current_rel_depth: int, max_rel_depth: int) -> list[str]:
    if current_rel_depth >= max_rel_depth:
        return []
    labels = []
    for child in node.get("children", []):
        if child.get("label"):
            labels.append(child["label"])
        labels.extend(_collect_descendant_labels(child, current_rel_depth + 1, max_rel_depth))
    return labels


def _combine_ilc_text(node: dict, descendant_labels: list[str], parts: list[str]) -> str:
    values = []
    if "description" in parts and node.get("description"):
        values.append(node["description"])
    if "label" in parts and node.get("label"):
        values.append(node["label"])
    if "descendant_labels" in parts and descendant_labels:
        values.extend(descendant_labels)
    return " ".join(values).strip()
