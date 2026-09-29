from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from domain_pipeline.steps.common import parse_keyword_string, parse_ner_string, read_table, write_table

if TYPE_CHECKING:
    from domain_pipeline.pipeline import PipelineContext


def filter_records(context: PipelineContext, cfg: dict) -> None:
    input_ref = context.artifacts.get(cfg["input_artifact"])
    df = read_table(input_ref)

    orcid_column = cfg.get("orcid_column", "orcid")
    with_orcid = df.dropna(subset=[orcid_column]).copy()
    without_orcid = df[df[orcid_column].isna()].copy()

    for part in (with_orcid, without_orcid):
        _filter_ner_by_confidence(part, cfg.get("ner_confidence", {}))
        _filter_keywords_by_confidence(part, cfg.get("keyword_confidence", {}))

    with_path = context.artifacts.output_path(cfg["with_orcid_artifact"], ".csv")
    without_path = context.artifacts.output_path(cfg["without_orcid_artifact"], ".csv")
    with_ref = context.artifacts.register(cfg["with_orcid_artifact"], "table", with_path, cfg)
    without_ref = context.artifacts.register(cfg["without_orcid_artifact"], "table", without_path, cfg)

    write_table(with_orcid, with_path)
    write_table(without_orcid, without_path)
    context.artifacts.write_metadata(with_ref, {"row_count": len(with_orcid), "source_artifact": input_ref.name})
    context.artifacts.write_metadata(without_ref, {"row_count": len(without_orcid), "source_artifact": input_ref.name})


def _filter_ner_by_confidence(df, cfg: dict) -> None:
    if "ner_entities" not in df:
        return
    scores = [score for value in df["ner_entities"] for _, _, score in parse_ner_string(value)]
    if not scores:
        return
    multiplier = float(cfg.get("std_multiplier", 3))
    lower = float(np.mean(scores) - multiplier * np.std(scores))
    upper = float(np.mean(scores) + multiplier * np.std(scores))

    def filter_value(value):
        kept = [
            f"{word}|{label}|{score}"
            for word, label, score in parse_ner_string(value)
            if lower <= score <= upper
        ]
        return "; ".join(kept)

    df["ner_entities"] = df["ner_entities"].apply(filter_value)


def _filter_keywords_by_confidence(df, cfg: dict) -> None:
    if "keywords" not in df:
        return
    min_score = float(cfg.get("min_score", 0.05))

    def filter_value(value):
        kept = [f"{keyword}|{score}" for keyword, score in parse_keyword_string(value) if score >= min_score]
        return "; ".join(kept)

    df["keywords"] = df["keywords"].apply(filter_value)
