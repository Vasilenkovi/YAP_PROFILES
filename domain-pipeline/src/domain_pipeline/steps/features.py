from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd

from domain_pipeline.steps.common import clean_token, parse_keyword_string, parse_ner_string, read_table, write_table

if TYPE_CHECKING:
    from domain_pipeline.pipeline import PipelineContext


def build_features(context: PipelineContext, cfg: dict) -> None:
    input_ref = context.artifacts.get(cfg["input_artifact"])
    df = read_table(input_ref)
    output_column = cfg.get("output_column", "smart_features")
    excluded = set(cfg.get("exclude_ner_labels", []))

    df[output_column] = df.apply(lambda row: _combine_features(row, excluded), axis=1)

    output_path = context.artifacts.output_path(cfg["output_artifact"], ".csv")
    output_ref = context.artifacts.register(cfg["output_artifact"], "table", output_path, cfg)
    write_table(df, output_path)
    context.artifacts.write_metadata(output_ref, {"row_count": len(df), "source_artifact": input_ref.name})


def filter_features(context: PipelineContext, cfg: dict) -> None:
    input_ref = context.artifacts.get(cfg["input_artifact"])
    df = read_table(input_ref)
    input_column = cfg.get("input_column", "smart_features")
    output_column = cfg.get("output_column", f"{input_column}_filtered")
    df_cfg = cfg.get("document_frequency", {})
    max_df_ratio = float(df_cfg.get("max_df_ratio", 0.10))
    min_word_length = int(df_cfg.get("min_word_length", 2))

    high_df_words = _document_frequency_words(df[input_column], max_df_ratio, min_word_length)
    df[output_column] = df[input_column].apply(lambda text: _remove_words(text, high_df_words, min_word_length))

    output_path = context.artifacts.output_path(cfg["output_artifact"], ".csv")
    output_ref = context.artifacts.register(cfg["output_artifact"], "table", output_path, cfg)
    write_table(df, output_path)
    context.artifacts.write_metadata(
        output_ref,
        {"row_count": len(df), "source_artifact": input_ref.name, "removed_words_count": len(high_df_words)},
    )


def _combine_features(row: pd.Series, excluded_ner_labels: set[str]) -> str:
    keywords = [clean_token(keyword) for keyword, _ in parse_keyword_string(row.get("keywords"))]
    entities = [
        clean_token(word)
        for word, label, _ in parse_ner_string(row.get("ner_entities"))
        if label not in excluded_ner_labels
    ]
    return " ".join([value for value in keywords + entities if value])


def _document_frequency_words(series, max_df_ratio: float, min_word_length: int) -> set[str]:
    counts: dict[str, int] = defaultdict(int)
    total = len(series)
    for text in series:
        if pd.isna(text):
            continue
        words = {w.lower() for w in str(text).split() if len(w) >= min_word_length}
        for word in words:
            counts[word] += 1
    threshold = total * max_df_ratio
    return {word for word, count in counts.items() if count > threshold}


def _remove_words(text, high_df_words: set[str], min_word_length: int) -> str:
    if pd.isna(text):
        return ""
    return " ".join(
        word
        for word in str(text).split()
        if len(word) >= min_word_length and word.lower() not in high_df_words
    )
