from __future__ import annotations

import re
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from domain_pipeline.artifacts import ArtifactRef


def read_table(ref: ArtifactRef) -> pd.DataFrame:
    if ref.type == "table":
        return pd.read_csv(ref.path)
    if ref.type == "table_pickle":
        try:
            return pd.read_pickle(ref.path)
        except Exception:
            with ref.path.open("rb") as fh:
                obj = pickle.load(fh)
            if isinstance(obj, pd.DataFrame):
                return obj
            return pd.DataFrame(obj)
    raise ValueError(f"Expected table artifact, got {ref.type}: {ref.name}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pkl":
        df.to_pickle(path)
    else:
        df.to_csv(path, index=False)


def join_text_columns(row: pd.Series, columns: list[str]) -> str:
    parts = []
    for column in columns:
        value = row.get(column, "")
        if pd.notna(value):
            parts.append(str(value))
    return " ".join(parts)


def parse_ner_string(value: Any) -> list[tuple[str, str, float]]:
    if value is None or not isinstance(value, str) or value.strip() == "":
        return []
    entities = []
    for entity in value.split("; "):
        try:
            word, label, score = entity.split("|")
            entities.append((word, label, float(score)))
        except ValueError:
            continue
    return entities


def parse_keyword_string(value: Any) -> list[tuple[str, float]]:
    if value is None or not isinstance(value, str) or value.strip() == "":
        return []
    keywords = []
    for item in value.split("; "):
        try:
            keyword, score = item.split("|")
            keywords.append((keyword, float(score)))
        except ValueError:
            continue
    return keywords


def clean_token(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()
