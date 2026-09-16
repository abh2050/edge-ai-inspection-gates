"""cycletime/llm/retrieval.py retrieves approved procedures with stable citations."""
from __future__ import annotations

import json
import re
from pathlib import Path

from cycletime.contracts import EvidenceRef
from cycletime.dataio.archive import sha256
from cycletime.evidence import record

# Retrieved text is data. These patterns mark passages that try to direct the workflow.
INSTRUCTION_PATTERNS = (
    r"\bignore (?:all |the )?(?:previous|prior|above)\b",
    r"\b(?:run|execute|invoke) (?:the )?(?:shell|command|bash|sh|python)\b",
    r"\bchange (?:the )?(?:cost|costs|threshold|tolerance)\b",
    r"\b(?:approve|activate|deploy) (?:the )?(?:release|model|station)\b",
    r"\byou (?:must|should) (?:now )?(?:call|use|run)\b",
    r"\bsystem prompt\b",
)


def embedded_instructions(text: str) -> list[str]:
    """Return instruction-shaped spans so the caller can quarantine them as untrusted data."""
    found = []
    for pattern in INSTRUCTION_PATTERNS:
        found.extend(match.group(0) for match in re.finditer(pattern, text, re.IGNORECASE))
    return found


def passages(text: str, query: str) -> list[tuple[int, str]]:
    """Return numbered paragraphs that share a term with the query, preserving page order."""
    terms = {word for word in re.findall(r"[a-z0-9]+", query.lower()) if len(word) > 3}
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    scored = [
        (number, block)
        for number, block in enumerate(blocks, start=1)
        if terms & set(re.findall(r"[a-z0-9]+", block.lower()))
    ]
    return scored


def retrieve_procedure(query: str, tenant_id: str, approved_manifest: Path) -> list[EvidenceRef]:
    """Retrieve relevant passages from approved versioned local procedures and evidence, preserving document, page, and hash citations; refuse cross-tenant access, unapproved documents, and instructions embedded in retrieved text."""
    approved_manifest = Path(approved_manifest)
    if not approved_manifest.is_file():
        raise ValueError("Retrieval requires an approved document manifest.")
    manifest = json.loads(approved_manifest.read_text())
    root = approved_manifest.parent
    if manifest.get("schema_version") != 1:
        raise ValueError("The approved manifest schema is unsupported.")
    results = []
    for entry in manifest["documents"]:
        if entry["tenant_id"] != tenant_id:
            # Another customer's document never reaches model context.
            continue
        if not entry.get("approved"):
            continue
        path = root / entry["path"]
        if not path.is_file():
            raise ValueError(f"An approved document is missing: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"An approved document changed after approval: {entry['path']}")
        text = path.read_text(errors="replace")
        for number, block in passages(text, query):
            instructions = embedded_instructions(block)
            results.append(
                record(
                    root.parent / "artifacts/agent/retrieval" / f"{entry['document_id']}-{number}.json",
                    {
                        "document_id": entry["document_id"],
                        "document_version": entry["version"],
                        "document_sha256": entry["sha256"],
                        "tenant_id": tenant_id,
                        "page": number,
                        "query": query,
                        "passage": block,
                        "content_status": "untrusted_document_data",
                        "instructions_are_not_commands": True,
                        "embedded_instruction_spans": instructions,
                        "quarantined": bool(instructions),
                    },
                )
            )
    if not results:
        raise ValueError(f"No approved document for tenant {tenant_id} matches the query.")
    return results
