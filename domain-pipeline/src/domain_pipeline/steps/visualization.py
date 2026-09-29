from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from domain_pipeline.config import resolve_path
from domain_pipeline.steps.common import parse_keyword_string, parse_ner_string, read_table, write_table

if TYPE_CHECKING:
    from domain_pipeline.pipeline import PipelineContext


def build_umap_projection(context: PipelineContext, cfg: dict) -> None:
    import umap.umap_ as umap

    items_ref = context.artifacts.get(cfg["input_items"])
    embeddings_ref = context.artifacts.get(cfg["input_embeddings"])
    df = read_table(items_ref)
    embeddings = np.load(embeddings_ref.path)
    if len(df) != embeddings.shape[0]:
        raise ValueError(f"Row mismatch: {len(df)} table rows vs {embeddings.shape[0]} embeddings")

    embeddings_scaled = StandardScaler().fit_transform(embeddings) if cfg.get("scaler", "standard") == "standard" else embeddings
    reducer = umap.UMAP(
        n_neighbors=int(cfg.get("n_neighbors", 15)),
        min_dist=float(cfg.get("min_dist", 0.1)),
        random_state=int(cfg.get("random_state", 42)),
    )
    projection = reducer.fit_transform(embeddings_scaled)
    x_col = cfg.get("x_column", "umap_x")
    y_col = cfg.get("y_column", "umap_y")
    df[x_col] = projection[:, 0]
    df[y_col] = projection[:, 1]

    output_path = context.artifacts.output_path(cfg["output_artifact"], ".csv")
    output_ref = context.artifacts.register(cfg["output_artifact"], "table", output_path, cfg)
    write_table(df, output_path)
    context.artifacts.write_metadata(output_ref, {"row_count": len(df), "source_artifact": items_ref.name})


def write_plotly_scatter(context: PipelineContext, cfg: dict) -> None:
    import plotly.express as px

    input_ref = context.artifacts.get(cfg["input_artifact"])
    df = read_table(input_ref)
    output_path = resolve_path(context.config, cfg.get("output_html", f"artifacts/{cfg['input_artifact']}.html"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = px.scatter(
        df,
        x=cfg.get("x_column", "umap_x"),
        y=cfg.get("y_column", "umap_y"),
        color=df[cfg.get("color_column")].astype(str) if cfg.get("color_column") else None,
        hover_data=cfg.get("hover_data"),
        title=cfg.get("title"),
    )
    fig.write_html(output_path)


def build_wordclouds(context: PipelineContext, cfg: dict) -> None:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    input_ref = context.artifacts.get(cfg["input_artifact"])
    df = read_table(input_ref)
    group_columns = cfg.get("group_columns", ["ilc_notation", "ilc_label"])
    output_dir = resolve_path(context.config, cfg.get("output_dir", "artifacts/wordclouds"))
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded = set(cfg.get("exclude_ner_labels", ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"]))
    for keys, group in df.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_name = "_".join(_safe_name(str(part)) for part in keys)
        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        ner_words = []
        keyword_words = []
        for _, row in group.iterrows():
            ner_words.extend(word.lower() for word, label, _ in parse_ner_string(row.get("ner_entities")) if label not in excluded)
            keyword_words.extend(keyword.lower() for keyword, _ in parse_keyword_string(row.get("keywords")))

        _write_wordcloud(" ".join(ner_words), group_dir / "ner_wordcloud.png", WordCloud, plt, cfg)
        _write_wordcloud(" ".join(keyword_words), group_dir / "keywords_wordcloud.png", WordCloud, plt, cfg)
        _write_wordcloud(" ".join(ner_words + keyword_words), group_dir / "combined_wordcloud.png", WordCloud, plt, cfg)


def _write_wordcloud(text: str, output_path: Path, wordcloud_cls, plt, cfg: dict) -> None:
    if not text.strip():
        return
    cloud = wordcloud_cls(
        width=int(cfg.get("width", 1200)),
        height=int(cfg.get("height", 800)),
        background_color=cfg.get("background_color", "white"),
        colormap=cfg.get("colormap", "viridis"),
        max_words=int(cfg.get("max_words", 200)),
        relative_scaling=float(cfg.get("relative_scaling", 0.5)),
        random_state=int(cfg.get("random_state", 42)),
    ).generate(text)
    plt.figure(figsize=(12, 8))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=int(cfg.get("dpi", 150)), bbox_inches="tight")
    plt.close()


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w\-_ ]", "", value).replace(" ", "_").replace("/", "_")
