from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

from domain_pipeline.config import resolve_path

if TYPE_CHECKING:
    from domain_pipeline.pipeline import PipelineContext


OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "dc": "http://purl.org/dc/elements/1.1/"}
DC_COLUMNS = [
    "title",
    "creators",
    "contributors",
    "subjects",
    "description",
    "publisher",
    "date",
    "type",
    "language",
    "relation",
    "source",
    "coverage",
    "rights",
    "identifiers",
    "formats",
]
DC_TAGS = {
    "title": "title",
    "creators": "creator",
    "contributors": "contributor",
    "subjects": "subject",
    "description": "description",
    "publisher": "publisher",
    "date": "date",
    "type": "type",
    "language": "language",
    "relation": "relation",
    "source": "source",
    "coverage": "coverage",
    "rights": "rights",
    "identifiers": "identifier",
    "formats": "format",
}


def harvest_oai_records(context: PipelineContext, cfg: dict) -> None:
    import requests

    output_path = context.artifacts.output_path(cfg["output_artifact"], ".csv")
    output_ref = context.artifacts.register(cfg["output_artifact"], "table", output_path, cfg)
    token_path = resolve_path(context.config, cfg.get("token_path", f"artifacts/{cfg['output_artifact']}.token"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.parent.mkdir(parents=True, exist_ok=True)

    base_url = cfg["base_url"]
    params = {"verb": "ListRecords", "metadataPrefix": cfg.get("metadata_prefix", "oai_dc")}
    if cfg.get("set"):
        params["set"] = cfg["set"]
    if token_path.exists():
        params = {"verb": "ListRecords", "resumptionToken": token_path.read_text(encoding="utf-8").strip()}

    rows_written = 0
    with output_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["oai_identifier", *DC_COLUMNS, "orcid"])
        if output_path.stat().st_size == 0:
            writer.writeheader()

        while True:
            response = requests.get(base_url, params=params, timeout=int(cfg.get("timeout", 60)))
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for record in root.findall(".//oai:record", OAI_NS):
                row = _parse_oai_record(record)
                writer.writerow(row)
                rows_written += 1

            token = root.find(".//oai:resumptionToken", OAI_NS)
            token_text = token.text.strip() if token is not None and token.text else ""
            if not token_text:
                if token_path.exists():
                    token_path.unlink()
                break
            token_path.write_text(token_text, encoding="utf-8")
            params = {"verb": "ListRecords", "resumptionToken": token_text}

            max_batches = cfg.get("max_batches")
            if max_batches and rows_written >= int(max_batches) * int(cfg.get("approx_batch_size", 100)):
                break

    context.artifacts.write_metadata(output_ref, {"rows_written_last_run": rows_written})


def _parse_oai_record(record) -> dict[str, str]:
    header_id = record.findtext("oai:header/oai:identifier", default="", namespaces=OAI_NS)
    metadata = record.find("oai:metadata", OAI_NS)
    row = {"oai_identifier": header_id, "orcid": ""}
    for column in DC_COLUMNS:
        row[column] = ""
    if metadata is None:
        return row

    for column in DC_COLUMNS:
        tag = DC_TAGS[column]
        values = [node.text.strip() for node in metadata.findall(f".//dc:{tag}", OAI_NS) if node.text]
        row[column] = "; ".join(values)

    identifiers = row.get("identifiers", "")
    row["orcid"] = _extract_orcid(identifiers)
    return row


def _extract_orcid(value: str) -> str:
    for part in value.replace(",", ";").split(";"):
        part = part.strip()
        if "orcid.org/" in part:
            return part.rsplit("/", 1)[-1]
    return ""
