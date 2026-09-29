from pathlib import Path

from domain_pipeline.config import load_config, resolve_path


def test_load_config_resolves_relative_paths():
    config = load_config(Path("configs/pipeline.reuse_existing.yaml"))

    assert resolve_path(config, "artifacts").name == "artifacts"
