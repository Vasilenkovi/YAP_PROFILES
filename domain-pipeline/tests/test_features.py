import pandas as pd

from domain_pipeline.steps.features import _combine_features, _document_frequency_words, _remove_words


def test_combine_features_excludes_meaningless_ner_labels():
    row = pd.Series(
        {
            "keywords": "machine learning|0.9; graph mining|0.8",
            "ner_entities": "MIT|ORG|0.99; 2020|DATE|0.95",
        }
    )

    result = _combine_features(row, {"DATE"})

    assert result == "machine learning graph mining MIT"


def test_document_frequency_filter_removes_frequent_words():
    words = _document_frequency_words(pd.Series(["alpha beta", "alpha gamma", "alpha delta"]), 0.5, 2)

    assert words == {"alpha"}
    assert _remove_words("alpha beta gamma", words, 2) == "beta gamma"
