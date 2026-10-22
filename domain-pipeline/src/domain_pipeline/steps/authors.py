from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd

from domain_pipeline.steps.common import parse_keyword_string, parse_ner_string, read_table, write_table

if TYPE_CHECKING:
    from domain_pipeline.pipeline import PipelineContext


def _explode_orcids(df: pd.DataFrame, orcid_column: str, cfg: dict) -> pd.DataFrame:
    """orcid_separator: ";" -> одна строка на каждый ORCID работы (по умолчанию не применяется)."""
    separator = cfg.get("orcid_separator")
    if not separator:
        return df
    df = df.copy()
    df[orcid_column] = df[orcid_column].astype(str).str.split(separator)
    df = df.explode(orcid_column)
    df[orcid_column] = df[orcid_column].str.strip()
    return df[df[orcid_column] != ""]


def build_author_profiles(context: PipelineContext, cfg: dict) -> None:
    input_ref = context.artifacts.get(cfg["input_artifact"])
    df = read_table(input_ref)
    orcid_column = cfg.get("orcid_column", "orcid")
    df = _explode_orcids(df.dropna(subset=[orcid_column]), orcid_column, cfg)

    profiles = defaultdict(lambda: {"ner": defaultdict(int), "keywords": defaultdict(int)})
    for _, row in df.iterrows():
        orcid = row[orcid_column]
        for word, _, _ in parse_ner_string(row.get("ner_entities")):
            profiles[orcid]["ner"][word] += 1
        for keyword, _ in parse_keyword_string(row.get("keywords")):
            profiles[orcid]["keywords"][keyword] += 1

    rows = []
    for orcid, data in profiles.items():
        rows.append(
            {
                orcid_column: orcid,
                "num_ner": len(data["ner"]),
                "num_keywords": len(data["keywords"]),
                "ner_stats": dict(data["ner"]),
                "keyword_stats": dict(data["keywords"]),
            }
        )

    result = pd.DataFrame(rows)
    output_path = context.artifacts.output_path(cfg["output_artifact"], ".csv")
    output_ref = context.artifacts.register(cfg["output_artifact"], "table", output_path, cfg)
    write_table(result, output_path)
    context.artifacts.write_metadata(output_ref, {"row_count": len(result), "source_artifact": input_ref.name})


def extract_author_entities(context: PipelineContext, cfg: dict) -> None:
    from transformers import pipeline

    input_ref = context.artifacts.get(cfg["input_artifact"])
    df = read_table(input_ref)
    orcid_column = cfg.get("orcid_column", "orcid")
    text_columns = cfg.get("text_columns", ["title", "description"])
    max_len = int(cfg.get("max_text_len", 5000))
    df = _explode_orcids(df.dropna(subset=[orcid_column]), orcid_column, cfg)

    profiles = df.groupby(orcid_column).agg(
        {column: lambda values: " ".join(values.dropna().astype(str)) for column in text_columns}
    ).reset_index()

    ner_cfg = cfg.get("ner", {})
    ner = pipeline(
        "ner",
        model=ner_cfg.get("model", "xlm-roberta-large-finetuned-conll03-english"),
        aggregation_strategy=ner_cfg.get("aggregation_strategy", "simple"),
        device=ner_cfg.get("device", -1),
    )

    entities = []
    for _, row in profiles.iterrows():
        text = " ".join(str(row.get(column, "")) for column in text_columns)[:max_len]
        try:
            entities.append("; ".join(sorted({item["word"] for item in ner(text)})))
        except Exception:
            entities.append("")
    profiles["entities"] = entities

    output_path = context.artifacts.output_path(cfg["output_artifact"], ".csv")
    output_ref = context.artifacts.register(cfg["output_artifact"], "table", output_path, cfg)
    write_table(profiles, output_path)
    context.artifacts.write_metadata(output_ref, {"row_count": len(profiles), "source_artifact": input_ref.name})


def export_author_entities_for_neo4j(context: PipelineContext, cfg: dict) -> None:
    input_ref = context.artifacts.get(cfg["input_artifact"])
    df = read_table(input_ref)
    orcid_column = cfg.get("orcid_column", "orcid")
    entity_column = cfg.get("entity_column", "entities")

    rows = []
    for _, row in df.iterrows():
        entities = str(row.get(entity_column, "") or "").split("; ")
        for entity in entities:
            entity = entity.strip()
            if entity:
                rows.append({orcid_column: row[orcid_column], "entity": entity})
    result = pd.DataFrame(rows)

    output_path = context.artifacts.output_path(cfg["output_artifact"], ".csv")
    output_ref = context.artifacts.register(cfg["output_artifact"], "table", output_path, cfg)
    write_table(result, output_path)
    context.artifacts.write_metadata(output_ref, {"row_count": len(result), "source_artifact": input_ref.name})
