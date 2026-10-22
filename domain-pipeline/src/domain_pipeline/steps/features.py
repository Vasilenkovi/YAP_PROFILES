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


# Общенаучная лексика, не несущая информации о предметной области.
ACADEMIC_STOP_WORDS = {
    "study", "studies", "studied", "studying", "paper", "article", "work", "works", "research", "researchers",
    "result", "results", "resulting", "method", "methods", "methodology", "approach", "approaches", "using", "use",
    "used", "uses", "usage", "based", "basis", "analysis", "analyses", "analyzed", "analysed", "data", "model",
    "models", "modeling", "modelling", "new", "novel", "different", "various", "several", "number", "numbers",
    "high", "higher", "low", "lower", "increase", "increased", "increasing", "decrease", "decreased", "effect",
    "effects", "influence", "impact", "obtained", "showed", "shown", "show", "shows", "found", "compared",
    "comparison", "presented", "present", "proposed", "propose", "proposes", "developed", "development", "provide",
    "provides", "provided", "investigated", "investigation", "considered", "consider", "conditions", "condition",
    "characteristics", "parameters", "parameter", "properties", "property", "possible", "possibility", "important",
    "significant", "significantly", "main", "key", "current", "currently", "recent", "recently", "years", "year",
    "time", "times", "case", "cases", "process", "processes", "problem", "problems", "task", "tasks", "level",
    "levels", "value", "values", "rate", "rates", "factor", "factors", "aim", "purpose", "objective", "relevance",
    "conclusion", "conclusions", "experimental", "experiment", "experiments", "theoretical", "practical", "general",
    "specific", "special", "total", "large", "small", "strong", "good", "better", "best", "vast", "majority",
    "allows", "allow", "allowed", "make", "made", "makes", "carried", "performed", "determined", "determine",
    "evaluate", "evaluated", "evaluation", "assessment", "estimate", "estimated", "data", "information", "field",
    "fields", "area", "areas", "type", "types", "way", "ways", "order", "part", "parts", "first", "second",
    "however", "also", "well", "within", "among", "due", "respectively", "approximately", "including", "et", "al",
}


def refine_features(context: PipelineContext, cfg: dict) -> None:
    """Очистка NER/keywords до склейки признаков: скор, метки NER, цифры, стоп-слова, частотность.

    В отличие от build_features + filter_features, работает с исходными колонками
    keywords/ner_entities (со скорами и метками), поэтому может отсекать по ним.
    """
    import re

    input_ref = context.artifacts.get(cfg["input_artifact"])
    df = read_table(input_ref)
    output_column = cfg.get("output_column", "smart_features_refined")

    kw_cfg = cfg.get("keywords", {})
    kw_min_score = float(kw_cfg.get("min_score", 0.15))
    kw_keep_top = int(kw_cfg.get("always_keep_top", 3))
    ner_cfg = cfg.get("ner", {})
    keep_labels = set(ner_cfg.get("keep_labels", ["PRODUCT", "FAC", "LOC", "EVENT", "LAW", "PERSON", "LANGUAGE"]))
    ner_blacklist = re.compile(ner_cfg.get("blacklist_regex", r"$^"), re.IGNORECASE)
    tok_cfg = cfg.get("tokens", {})
    min_len = int(tok_cfg.get("min_word_length", 3))
    drop_digits = bool(tok_cfg.get("drop_tokens_with_digits", True))
    stop_words = set(tok_cfg.get("extra_stop_words", []))
    if tok_cfg.get("english_stop_words", True):
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop_words |= set(ENGLISH_STOP_WORDS)
    if tok_cfg.get("academic_stop_words", True):
        stop_words |= ACADEMIC_STOP_WORDS
    max_df_ratio = float(cfg.get("document_frequency", {}).get("max_df_ratio", 0.05))
    token_re = re.compile(r"[\w\-]+", re.UNICODE)

    def clean_tokens(text: str) -> list[str]:
        tokens = []
        for tok in token_re.findall(str(text)):
            low = tok.lower().strip("-_")
            if len(low) < min_len or low in stop_words:
                continue
            if drop_digits and any(ch.isdigit() for ch in low):
                continue
            tokens.append(low)
        return tokens

    def doc_tokens(row) -> list[str]:
        kws = sorted(parse_keyword_string(row.get("keywords")), key=lambda x: -x[1])
        kept_kw = [k for i, (k, s) in enumerate(kws) if s >= kw_min_score or (i < kw_keep_top and s > 0)]
        ents = [w for w, label, _ in parse_ner_string(row.get("ner_entities"))
                if label in keep_labels and not ner_blacklist.search(w)]
        seen, out = set(), []
        for tok in (t for phrase in kept_kw + ents for t in clean_tokens(phrase)):
            if tok not in seen:  # дубли внутри документа не нужны
                seen.add(tok)
                out.append(tok)
        return out

    docs = df.apply(doc_tokens, axis=1)
    doc_freq = defaultdict(int)
    for toks in docs:
        for tok in set(toks):
            doc_freq[tok] += 1
    threshold = len(df) * max_df_ratio
    frequent = {t for t, c in doc_freq.items() if c > threshold}
    df[output_column] = docs.apply(lambda toks: " ".join(t for t in toks if t not in frequent))

    output_path = context.artifacts.output_path(cfg["output_artifact"], ".csv")
    output_ref = context.artifacts.register(cfg["output_artifact"], "table", output_path, cfg)
    write_table(df, output_path)
    frequent_path = output_path.with_name(output_path.stem + ".removed_frequent_tokens.txt")
    frequent_path.write_text("\n".join(sorted(frequent, key=lambda t: -doc_freq[t])), encoding="utf-8")
    context.artifacts.write_metadata(
        output_ref,
        {
            "row_count": len(df),
            "source_artifact": input_ref.name,
            "removed_frequent_tokens": len(frequent),
            "empty_docs": int((df[output_column].str.strip() == "").sum()),
            "median_tokens": float(df[output_column].str.split().str.len().median()),
        },
    )
