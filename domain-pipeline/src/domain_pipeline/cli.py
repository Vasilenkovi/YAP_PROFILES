from __future__ import annotations

import argparse
import os

from domain_pipeline.config import load_config, resolve_path
from domain_pipeline.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="domain-pipeline")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run configured pipeline steps.")
    run.add_argument("--config", required=True)
    run.add_argument("--from", dest="from_step", default=None)
    run.add_argument("--step", dest="only_step", default=None)

    validate = subparsers.add_parser("validate", help="Validate configured artifacts.")
    validate.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    config = load_config(args.config)
    _configure_model_cache(config)
    if args.command == "validate":
        run_pipeline(config, only_step="validate_artifacts")
    elif args.command == "run":
        run_pipeline(config, from_step=args.from_step, only_step=args.only_step)


def _configure_model_cache(config: dict) -> None:
    """project.hf_home: куда скачиваются модели HuggingFace / sentence-transformers.

    Выставляется до импорта transformers (шаги импортируют модели лениво).
    """
    hf_home = config.get("project", {}).get("hf_home")
    if not hf_home:
        return
    path = resolve_path(config, hf_home)
    path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(path)
    print(f"HF_HOME={path}")
