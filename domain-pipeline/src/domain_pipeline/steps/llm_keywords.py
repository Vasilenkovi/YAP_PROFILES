from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from domain_pipeline.config import resolve_path
from domain_pipeline.steps.common import join_text_columns, read_table, write_table

if TYPE_CHECKING:
    from domain_pipeline.pipeline import PipelineContext

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda iterable, **_: iterable


# Синонимы ключей, которые модели иногда возвращают вместо заданных.
KEY_ALIASES = {
    "methods": "method", "technique": "method", "techniques": "method",
    "tools": "tool", "technology": "tool", "technologies": "tool", "instrument": "tool", "instruments": "tool",
    "tools_technologies": "tool", "disciplines": "discipline", "field": "discipline", "fields": "discipline",
}


def extract_llm_keywords(context: PipelineContext, cfg: dict) -> None:
    """Few-shot извлечение ключевых слов компетенций локальной LLM (замена NER/KeyBERT).

    Выход совместим с остальными шагами: колонка keywords в формате "term|score",
    поэтому build_features / refine_features / build_author_profiles работают без изменений.
    Дополнительно — по колонке на категорию (llm_method, llm_tool, llm_discipline) и llm_features.
    """
    input_ref = context.artifacts.get(cfg["input_artifact"])
    df = read_table(input_ref)
    if cfg.get("limit"):
        df = df.head(int(cfg["limit"])).copy()

    fewshot = load_fewshot(resolve_path(context.config, cfg["fewshot_path"]))
    categories = list(fewshot["categories"])
    messages_prefix = build_messages_prefix(fewshot)

    output_path = context.artifacts.output_path(cfg["output_artifact"], ".csv")
    output_ref = context.artifacts.register(cfg["output_artifact"], "table", output_path, cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".partial.jsonl")

    done = _read_checkpoint(checkpoint_path)
    text_columns = cfg.get("text_columns", ["title", "description"])
    max_len = int(cfg.get("max_text_len", 2500))
    validation = cfg.get("validation", {})
    id_column = cfg.get("id_column") or input_ref.metadata.get("id_column")
    keys = [str(df.iloc[i][id_column]) if id_column else str(i) for i in range(len(df))]

    backend = make_backend(cfg.get("backend", {}))
    batch_size = int(cfg.get("backend", {}).get("batch_size", 1))
    todo = [i for i in range(len(df)) if keys[i] not in done]
    with checkpoint_path.open("a", encoding="utf-8") as ckpt:
        for start in tqdm(range(0, len(todo), batch_size), desc="LLM keywords"):
            idx = todo[start : start + batch_size]
            texts = [join_text_columns(df.iloc[i], text_columns)[:max_len] for i in idx]
            replies = backend.generate([messages_prefix + [{"role": "user", "content": t}] for t in texts])
            for i, text, reply in zip(idx, texts, replies):
                terms, status = parse_reply(reply, categories)
                terms = validate_terms(terms, text, validation)
                record = {"key": keys[i], "status": status, "terms": terms, "raw": reply}
                ckpt.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[keys[i]] = record
            ckpt.flush()

    records = [done[k] for k in keys]
    result = df.copy()
    for cat in categories:
        result[f"llm_{cat}"] = ["; ".join(r["terms"].get(cat, [])) for r in records]
    result["llm_status"] = [r["status"] for r in records]
    if cfg.get("keep_raw_response", True):
        result["llm_raw"] = [r["raw"] for r in records]
    weights = cfg.get("category_weights", {})
    result[cfg.get("keywords_output_column", "keywords")] = [
        "; ".join(f"{t}|{float(weights.get(cat, 1.0))}" for cat in categories for t in r["terms"].get(cat, []))
        for r in records
    ]
    result[cfg.get("features_output_column", "llm_features")] = [
        " ".join(t for cat in categories for t in r["terms"].get(cat, [])) for r in records
    ]

    write_table(result, output_path)
    statuses = result["llm_status"].value_counts().to_dict()
    context.artifacts.write_metadata(
        output_ref,
        {
            "row_count": len(result),
            "source_artifact": input_ref.name,
            "backend": cfg.get("backend", {}),
            "fewshot_path": str(cfg["fewshot_path"]),
            "fewshot_examples": len(fewshot["examples"]),
            "status_counts": {str(k): int(v) for k, v in statuses.items()},
            "mean_terms": {cat: float(result[f"llm_{cat}"].str.split("; ").apply(lambda x: len([t for t in x if t])).mean())
                           for cat in categories},
        },
    )
    if cfg.get("remove_checkpoint_on_success", False):
        checkpoint_path.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- prompt
def load_fewshot(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    categories = data.get("categories") or {"method": "", "tool": "", "discipline": ""}
    if isinstance(categories, list):
        categories = {c: "" for c in categories}
    examples = data.get("examples") or []
    for ex in examples:
        if "text" not in ex or "output" not in ex:
            raise ValueError(f"Few-shot example must have 'text' and 'output': {ex.get('id', ex)}")
        missing = set(categories) - set(ex["output"])
        if missing:
            raise ValueError(f"Few-shot example {ex.get('id')} lacks categories: {sorted(missing)}")
    return {"categories": categories, "examples": examples, "instruction": data.get("instruction", "")}


def build_messages_prefix(fewshot: dict[str, Any]) -> list[dict[str, str]]:
    cats = fewshot["categories"]
    definitions = "\n".join(f"- {name}: {desc}" for name, desc in cats.items() if desc)
    system = fewshot["instruction"] or (
        "Extract competence keywords from the scientific abstract. Return only a JSON object with keys "
        + ", ".join(f'"{c}"' for c in cats) + "; each value is a list of short noun phrases."
    )
    if definitions:
        system += "\nCategories:\n" + definitions
    messages = [{"role": "system", "content": system}]
    for ex in fewshot["examples"]:
        messages.append({"role": "user", "content": ex["text"]})
        out = {c: ex["output"].get(c, []) for c in cats}
        messages.append({"role": "assistant", "content": json.dumps(out, ensure_ascii=False)})
    return messages


# ----------------------------------------------------------------------------- parsing / validation
def parse_reply(reply: str, categories: list[str]) -> tuple[dict[str, list[str]], str]:
    """Достаёт первый JSON-объект из ответа модели; терпим к ```json``` и лишнему тексту."""
    if not reply or not reply.strip():
        return {c: [] for c in categories}, "empty"
    text = re.sub(r"```(?:json)?", "", reply)
    obj = None
    start = text.find("{")
    while start != -1 and obj is None:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1) if obj is None else start
    status = "ok"
    if not isinstance(obj, dict):
        obj = _salvage_truncated(text)
        if not any(obj.values()):
            return {c: [] for c in categories}, "parse_error"
        status = "truncated"

    terms: dict[str, list[str]] = {c: [] for c in categories}
    for key, value in obj.items():
        norm = KEY_ALIASES.get(str(key).strip().lower(), str(key).strip().lower())
        if norm not in terms:
            continue
        if isinstance(value, str):
            value = [v for v in re.split(r"[;,]", value)]
        if isinstance(value, list):
            terms[norm].extend(str(v) for v in value if isinstance(v, (str, int, float)))
    return terms, status


def _salvage_truncated(text: str) -> dict[str, list[str]]:
    """Ответ оборван по max_new_tokens (обычно из-за зацикливания): берём завершённые строки списков."""
    result: dict[str, list[str]] = {}
    for match in re.finditer(r'"([A-Za-z_]+)"\s*:\s*\[(.*?)(?:\]|$)', text, flags=re.S):
        items = re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(2))
        result[match.group(1)] = items
    return result


def validate_terms(terms: dict[str, list[str]], text: str, cfg: dict) -> dict[str, list[str]]:
    """Нормализация и защита от галлюцинаций: для категорий из require_in_text термин должен
    опираться на слова аннотации (совпадение по префиксу 5 символов у каждого значимого слова)."""
    max_terms = int(cfg.get("max_terms_per_category", 6))
    max_words = int(cfg.get("max_words_per_term", 5))
    lowercase = bool(cfg.get("lowercase", True))
    require = set(cfg.get("require_in_text", ["method", "tool"]))
    text_words = {w[:5] for w in re.findall(r"[a-zа-я0-9]+", text.lower()) if len(w) >= 3}
    acronyms = set(re.findall(r"\b[A-Z][A-Za-z0-9-]*[A-Z0-9]\b", text))

    out: dict[str, list[str]] = {}
    for cat, values in terms.items():
        seen, kept = set(), []
        for raw in values:
            term = re.sub(r"\s+", " ", str(raw)).strip(" .;:,-\"'")
            if not term or len(term) < 2 or len(term.split()) > max_words or term.replace(".", "").isdigit():
                continue
            if cat in require and not _grounded(term, text_words, acronyms):
                continue
            norm = term.lower() if lowercase else term
            if norm.lower() not in seen:
                seen.add(norm.lower())
                kept.append(norm)
        out[cat] = kept[:max_terms]
    if cfg.get("unique_across_categories", True):
        # малые модели копируют одни и те же термины во все категории — оставляем первую по порядку
        used: set[str] = set()
        for cat in out:
            out[cat] = [t for t in out[cat] if t.lower() not in used]
            used.update(t.lower() for t in out[cat])
    return out


def _grounded(term: str, text_words: set[str], acronyms: set[str]) -> bool:
    if term in acronyms:
        return True
    words = [w for w in re.findall(r"[a-zа-я0-9]+", term.lower()) if len(w) >= 3]
    if not words:
        return False
    hits = sum(w[:5] in text_words for w in words)
    # раскрытая аббревиатура ("powder X-ray diffraction" при "PXRD") — допускаем половину слов
    return hits / len(words) >= 0.5


def _read_checkpoint(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["key"]] = rec
    return done


# ----------------------------------------------------------------------------- backends
def make_backend(cfg: dict):
    kind = cfg.get("type", "transformers")
    if kind == "transformers":
        return TransformersBackend(cfg)
    if kind == "ollama":
        return OllamaBackend(cfg)
    if kind == "openai_compatible":
        return OpenAICompatibleBackend(cfg)
    raise ValueError(f"Unknown LLM backend: {kind}")


class TransformersBackend:
    """Модель в процессе пайплайна; greedy decoding, батчи с left padding."""

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = int(cfg.get("max_new_tokens", 256))
        # repetition_penalty, no_repeat_ngram_size и др. — передаются в model.generate как есть
        self.generation_kwargs = dict(cfg.get("generation_kwargs", {}))
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"], padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = getattr(torch, cfg.get("dtype", "float16"))
        device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype).to(device).eval()
        self.device = device

    def generate(self, conversations: list[list[dict]]) -> list[str]:
        prompts = [
            self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **batch, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id, **self.generation_kwargs,
            )
        new_tokens = out[:, batch["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


class OllamaBackend:
    """Ollama (http://localhost:11434): модели 7-8B в Q4 помещаются в 6 ГБ VRAM."""

    def __init__(self, cfg: dict):
        import requests

        self.session = requests.Session()
        self.url = cfg.get("base_url", "http://localhost:11434").rstrip("/") + "/api/chat"
        self.model = cfg["model"]
        self.timeout = int(cfg.get("timeout", 300))
        self.options = {"temperature": float(cfg.get("temperature", 0)), "num_predict": int(cfg.get("max_new_tokens", 256))}
        self.options.update(cfg.get("options", {}))

    def generate(self, conversations: list[list[dict]]) -> list[str]:
        replies = []
        for conv in conversations:
            resp = self.session.post(
                self.url,
                json={"model": self.model, "messages": conv, "stream": False, "format": "json", "options": self.options},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            replies.append(resp.json().get("message", {}).get("content", ""))
        return replies


class OpenAICompatibleBackend:
    """LM Studio / llama.cpp server / vLLM: POST {base_url}/chat/completions."""

    def __init__(self, cfg: dict):
        import os

        import requests

        self.session = requests.Session()
        self.url = cfg.get("base_url", "http://localhost:1234/v1").rstrip("/") + "/chat/completions"
        api_key = os.getenv(cfg.get("api_key_env", "LLM_API_KEY"), "not-needed")
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self.model = cfg["model"]
        self.timeout = int(cfg.get("timeout", 300))
        self.params = {"temperature": float(cfg.get("temperature", 0)), "max_tokens": int(cfg.get("max_new_tokens", 256))}
        if cfg.get("json_mode", False):
            self.params["response_format"] = {"type": "json_object"}

    def generate(self, conversations: list[list[dict]]) -> list[str]:
        replies = []
        for conv in conversations:
            resp = self.session.post(self.url, json={"model": self.model, "messages": conv, **self.params}, timeout=self.timeout)
            resp.raise_for_status()
            replies.append(resp.json()["choices"][0]["message"]["content"])
        return replies
