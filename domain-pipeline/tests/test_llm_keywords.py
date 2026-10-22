import json

import pytest

from domain_pipeline.steps.llm_keywords import build_messages_prefix, load_fewshot, parse_reply, validate_terms

CATS = ["method", "tool", "discipline"]


def test_parse_plain_json():
    terms, status = parse_reply('{"method": ["DFT"], "tool": ["SEM"], "discipline": ["catalysis"]}', CATS)
    assert status == "ok"
    assert terms == {"method": ["DFT"], "tool": ["SEM"], "discipline": ["catalysis"]}


def test_parse_fenced_json_with_text_and_aliases():
    reply = 'Sure!\n```json\n{"methods": ["XRD analysis"], "technologies": "SEM, TEM", "fields": []}\n```'
    terms, status = parse_reply(reply, CATS)
    assert status == "ok"
    assert terms["method"] == ["XRD analysis"]
    assert [t.strip() for t in terms["tool"]] == ["SEM", "TEM"]


def test_parse_errors():
    assert parse_reply("", CATS)[1] == "empty"
    assert parse_reply("no json here", CATS)[1] == "parse_error"
    assert parse_reply('{"method": [', CATS)[1] == "parse_error"


def test_validate_drops_ungrounded_methods_but_keeps_disciplines():
    text = "Coatings were studied by scanning electron microscopy (SEM) and X-ray diffraction."
    terms = {
        "method": ["X-ray diffraction", "molecular dynamics"],
        "tool": ["SEM", "scanning electron microscopy", "scanning electron microscopy"],
        "discipline": ["surface engineering"],
    }
    out = validate_terms(terms, text, {"require_in_text": ["method", "tool"]})
    assert out["method"] == ["x-ray diffraction"]                     # молекулярной динамики в тексте нет
    assert out["tool"] == ["sem", "scanning electron microscopy"]     # аббревиатура + дубликат убран
    assert out["discipline"] == ["surface engineering"]               # дисциплина может быть выведена


def test_validate_limits_and_noise():
    out = validate_terms({"method": ["2024", "a", "one two three four five six", "ok term"] + [f"t{i}" for i in range(10)]},
                         "ok term " + " ".join(f"t{i}" for i in range(10)), {"require_in_text": [], "max_terms_per_category": 3})
    assert out["method"] == ["ok term", "t0", "t1"]


def test_fewshot_file_and_prompt(tmp_path):
    path = tmp_path / "fs.json"
    path.write_text(json.dumps({
        "categories": {"method": "m", "tool": "t", "discipline": "d"},
        "examples": [{"text": "abc", "output": {"method": ["x"], "tool": [], "discipline": ["y"]}}],
    }), encoding="utf-8")
    fs = load_fewshot(path)
    msgs = build_messages_prefix(fs)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert json.loads(msgs[2]["content"]) == {"method": ["x"], "tool": [], "discipline": ["y"]}


def test_fewshot_missing_category(tmp_path):
    path = tmp_path / "fs.json"
    path.write_text(json.dumps({"categories": ["method", "tool"],
                                "examples": [{"text": "a", "output": {"method": []}}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_fewshot(path)


def test_project_fewshot_file_is_valid():
    fs = load_fewshot("../../tpu_domains/llm/fewshot_competences.json")
    assert list(fs["categories"]) == CATS and len(fs["examples"]) >= 3


def test_truncated_reply_is_salvaged():
    reply = '{"method": ["electrochemical sensors", "biosensors", "immuno", "immunoreactive conditi'
    terms, status = parse_reply(reply, CATS)
    assert status == "truncated"
    assert terms["method"] == ["electrochemical sensors", "biosensors", "immuno"]


def test_terms_unique_across_categories():
    text = "surface-enhanced raman spectroscopy of 2D materials"
    terms = {"method": ["surface-enhanced Raman spectroscopy"], "tool": ["surface-enhanced Raman spectroscopy", "2D materials"],
             "discipline": ["surface-enhanced Raman spectroscopy", "nanotechnology"]}
    out = validate_terms(terms, text, {"require_in_text": []})
    assert out == {"method": ["surface-enhanced raman spectroscopy"], "tool": ["2d materials"], "discipline": ["nanotechnology"]}
