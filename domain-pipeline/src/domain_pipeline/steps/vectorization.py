from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from domain_pipeline.steps.common import read_table

if TYPE_CHECKING:
    from domain_pipeline.pipeline import PipelineContext

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda iterable, **_: iterable


def vectorize(context: PipelineContext, cfg: dict) -> None:
    from sentence_transformers import SentenceTransformer

    input_ref = context.artifacts.get(cfg["input_artifact"])
    output_path = context.artifacts.output_path(cfg["output_artifact"], ".npy")
    output_ref = context.artifacts.register(cfg["output_artifact"], "embeddings", output_path, cfg)

    if cfg.get("reuse_if_exists", True) and output_path.exists():
        print(f"reuse embeddings: {output_path}")
        return

    df = read_table(input_ref)
    text_column = cfg.get("text_column", "smart_features")
    model_name = cfg.get("model", "all-mpnet-base-v2")
    batch_size = int(cfg.get("batch_size", 32))

    model = SentenceTransformer(model_name)
    texts = df[text_column].fillna("").astype(str).tolist()
    embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embeddings"):
        batch = texts[start : start + batch_size]
        embeddings.extend(model.encode(batch, show_progress_bar=False))

    array = np.asarray(embeddings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)
    context.artifacts.write_metadata(
        output_ref,
        {
            "source_artifact": input_ref.name,
            "row_count": len(df),
            "shape": list(array.shape),
            "text_column": text_column,
            "model": model_name,
        },
    )
