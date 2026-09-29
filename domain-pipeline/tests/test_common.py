from domain_pipeline.steps.common import parse_keyword_string, parse_ner_string


def test_parse_ner_string_skips_invalid_parts():
    parsed = parse_ner_string("MIT|ORG|0.99; bad; Paris|GPE|0.8")

    assert parsed == [("MIT", "ORG", 0.99), ("Paris", "GPE", 0.8)]


def test_parse_keyword_string_skips_invalid_parts():
    parsed = parse_keyword_string("semantic web|0.7; bad")

    assert parsed == [("semantic web", 0.7)]
