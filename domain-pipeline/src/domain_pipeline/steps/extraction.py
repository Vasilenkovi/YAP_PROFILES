from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd

from domain_pipeline.steps.common import join_text_columns

if TYPE_CHECKING:
    from domain_pipeline.pipeline import PipelineContext

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda iterable, **_: iterable


def extract_ner_keywords(context: PipelineContext, cfg: dict) -> None:
    from keybert import KeyBERT
    from transformers import pipeline

    input_ref = context.artifacts.get(cfg["input_artifact"])
    output_path = context.artifacts.output_path(cfg["output_artifact"], ".csv")
    output_ref = context.artifacts.register(cfg["output_artifact"], "table", output_path, cfg)
    temp_path = output_path.with_suffix(".temp.csv")

    df = pd.read_csv(input_ref.path)
    start_idx = 0
    ner_results: list[str] = []
    kw_results: list[str] = []

    if temp_path.exists():
        temp_df = pd.read_csv(temp_path)
        start_idx = len(temp_df)
        ner_results = temp_df["ner_entities"].tolist()
        kw_results = temp_df["keywords"].tolist()

    ner_cfg = cfg.get("ner", {})
    ner = pipeline(
        "ner",
        model=ner_cfg["model"],
        tokenizer=ner_cfg.get("tokenizer", ner_cfg["model"]),
        aggregation_strategy=ner_cfg.get("aggregation_strategy", "simple"),
        device=ner_cfg.get("device", -1),
    )

    kw_cfg = cfg.get("keywords", {})
    kw_model = KeyBERT(model=kw_cfg.get("model", "all-mpnet-base-v2"))
    text_columns = cfg.get("text_columns", ["title", "description"])
    max_len = int(cfg.get("max_text_len", 3000))
    checkpoint_every = int(cfg.get("checkpoint", {}).get("every_rows", 100))

    for idx in tqdm(range(start_idx, len(df)), desc="NER/keywords"):
        text = join_text_columns(df.iloc[idx], text_columns)[:max_len]
        ner_out = _extract_ner(ner, text)
        kw_out = _extract_keywords(kw_model, text, kw_cfg)
        ner_results.append("; ".join(f"{e['word']}|{e['label']}|{e['score']}" for e in ner_out))
        kw_results.append("; ".join(f"{k['keyword']}|{k['score']}" for k in kw_out))

        if checkpoint_every and idx % checkpoint_every == 0:
            pd.DataFrame({"ner_entities": ner_results, "keywords": kw_results}).to_csv(temp_path, index=False)

    result = df.copy()
    result["ner_entities"] = ner_results
    result["keywords"] = kw_results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    if temp_path.exists():
        os.remove(temp_path)
    context.artifacts.write_metadata(output_ref, {"row_count": len(result), "source_artifact": input_ref.name})


def _extract_ner(ner, text: str) -> list[dict]:
    if not text.strip():
        return []
    try:
        return [
            {"word": e["word"], "label": e["entity_group"], "score": round(float(e["score"]), 4)}
            for e in ner(text)
        ]
    except Exception:
        return []


def _extract_keywords(kw_model, text: str, cfg: dict) -> list[dict]:
    if not text.strip():
        return []
    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=tuple(cfg.get("keyphrase_ngram_range", [1, 3])),
            stop_words=cfg.get("stop_words", "english"),
            use_mmr=cfg.get("use_mmr", True),
            diversity=float(cfg.get("diversity", 0.7)),
            top_n=int(cfg.get("top_n", 10)),
        )
        return [{"keyword": kw, "score": round(float(score), 4)} for kw, score in keywords]
    except Exception:
        return []
