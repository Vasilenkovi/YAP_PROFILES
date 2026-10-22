from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from domain_pipeline.artifacts import ArtifactStore
from domain_pipeline.steps.assignment import domain_assignment
from domain_pipeline.steps.authors import build_author_profiles, export_author_entities_for_neo4j, extract_author_entities
from domain_pipeline.steps.clustering import cluster_embeddings
from domain_pipeline.steps.extraction import extract_ner_keywords
from domain_pipeline.steps.llm_keywords import extract_llm_keywords
from domain_pipeline.steps.features import build_features, filter_features, refine_features
from domain_pipeline.steps.filtering import filter_records
from domain_pipeline.steps.ilc import build_ilc_centroids, build_ilc_tree
from domain_pipeline.steps.source import harvest_oai_records
from domain_pipeline.steps.visualization import build_umap_projection, build_wordclouds, write_plotly_scatter


StepFunc = Callable[["PipelineContext", dict], None]


@dataclass
class PipelineContext:
    config: dict
    artifacts: ArtifactStore


STEP_REGISTRY: dict[str, StepFunc] = {
    "harvest_oai_records": harvest_oai_records,
    "extract_ner_keywords": extract_ner_keywords,
    "extract_llm_keywords": extract_llm_keywords,
    "filter_records": filter_records,
    "build_author_profiles": build_author_profiles,
    "extract_author_entities": extract_author_entities,
    "export_author_entities_for_neo4j": export_author_entities_for_neo4j,
    "build_features": build_features,
    "filter_features": filter_features,
    "refine_features": refine_features,
    "vectorize": build_features,  # replaced in import below to keep registry readable
    "cluster_embeddings": cluster_embeddings,
    "build_ilc_tree": build_ilc_tree,
    "build_ilc_centroids": build_ilc_centroids,
    "domain_assignment": domain_assignment,
    "build_umap_projection": build_umap_projection,
    "write_plotly_scatter": write_plotly_scatter,
    "build_wordclouds": build_wordclouds,
}

from domain_pipeline.steps.vectorization import vectorize  # noqa: E402

STEP_REGISTRY["vectorize"] = vectorize


def run_pipeline(config: dict, from_step: str | None = None, only_step: str | None = None) -> None:
    context = PipelineContext(config=config, artifacts=ArtifactStore(config))

    if only_step == "validate_artifacts":
        for message in context.artifacts.validate():
            print(message)
        return

    steps = config.get("steps") or {}
    ordered_step_names = list(STEP_REGISTRY)

    if only_step:
        ordered_step_names = [only_step]
    elif from_step:
        if from_step == "validate_artifacts":
            for message in context.artifacts.validate():
                print(message)
            ordered_step_names = ordered_step_names
        elif from_step in ordered_step_names:
            ordered_step_names = ordered_step_names[ordered_step_names.index(from_step) :]
        else:
            raise KeyError(f"Unknown start step: {from_step}")

    # Ключ шага может иметь суффикс "@tag" (domain_assignment@asjc): несколько экземпляров одного
    # шага выполняются в порядке реестра, а между собой — в порядке следования в конфиге.
    for unknown in {key.split("@", 1)[0] for key in steps} - set(STEP_REGISTRY) - {"validate_artifacts"}:
        raise KeyError(f"Unknown step in config: {unknown}")
    for name in ordered_step_names:
        if name not in STEP_REGISTRY:
            raise KeyError(f"Unknown step: {name}")
        for key, step_cfg in steps.items():
            if key.split("@", 1)[0] != name or not (step_cfg or {}).get("enabled", False):
                continue
            print(f"==> {key}")
            STEP_REGISTRY[name](context, step_cfg)
