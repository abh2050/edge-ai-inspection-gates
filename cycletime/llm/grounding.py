"""cycletime/llm/grounding.py checks narrative claims against immutable evidence."""
from __future__ import annotations

import math
import re
from pathlib import Path

from cycletime.contracts import JSON, EvidenceRef
from cycletime.evidence import read, record
from cycletime.llm.retrieval import embedded_instructions
from cycletime.llm.schemas import AgentReply

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
CAUSAL = re.compile(
    r"\b(because|caused by|due to|results? from|leads? to|responsible for)\b", re.IGNORECASE
)
HYPOTHESIS = re.compile(r"\b(hypothesis|hypothesize|may|might|could|possible|unverified)\b", re.IGNORECASE)
UNCERTAINTY = re.compile(
    r"\b(insufficient|unknown|not measured|unavailable|no evidence|cannot determine)\b", re.IGNORECASE
)
SAVINGS = re.compile(r"\b(savings?|saved|roi|payback|profit)\b", re.IGNORECASE)


def numeric_fields(value: object, prefix: str = "") -> dict[str, float]:
    """Flatten every numeric field in cited evidence so a claim can bind to an exact field."""
    fields: dict[str, float] = {}
    if isinstance(value, bool):
        return fields
    if isinstance(value, int | float):
        return {prefix or "value": float(value)}
    if isinstance(value, dict):
        for key, item in value.items():
            fields.update(numeric_fields(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            fields.update(numeric_fields(item, f"{prefix}[{index}]"))
    return fields


def binds(number: float, fields: dict[str, float]) -> str | None:
    """Return the evidence field a stated number reproduces, allowing only rounding differences."""
    for name, value in fields.items():
        if value == number or math.isclose(value, number, rel_tol=1e-9, abs_tol=1e-12):
            return name
        for places in range(7):
            if round(value, places) == number:
                return name
    return None


def validate_claims(reply: AgentReply, evidence: list[EvidenceRef]) -> EvidenceRef:
    """Resolve every cited hash and bind numeric statements to exact metric fields; require an explicit uncertainty statement when evidence is absent; refuse unsupported causes of defects, invented savings, arbitrary numeric substitutions, and instructions found in source documents."""
    available = {ref.sha256: ref for ref in evidence}
    rows: list[JSON] = []
    errors: list[str] = []
    for index, claim in enumerate(reply.claims):
        cited = []
        fields: dict[str, float] = {}
        for sha in claim.evidence_sha256:
            ref = available.get(sha)
            if ref is None:
                errors.append(f"claim {index}: cites unresolved evidence {sha[:12]}")
                continue
            content = read(ref)
            cited.append({"sha256": sha, "path": Path(ref.path).name})
            fields.update(numeric_fields(content))
            if isinstance(content, dict) and content.get("content_status") == "untrusted_document_data":
                spans = content.get("embedded_instruction_spans") or embedded_instructions(
                    str(content.get("passage", ""))
                )
                if spans:
                    errors.append(
                        f"claim {index}: repeats instructions embedded in {content.get('document_id')}"
                    )
        bindings = {}
        for token in NUMBER.findall(claim.text):
            number = float(token)
            field = binds(number, fields)
            if field is None:
                errors.append(f"claim {index}: number {token} matches no cited evidence field")
            else:
                bindings[token] = field
        if CAUSAL.search(claim.text) and not HYPOTHESIS.search(claim.text):
            errors.append(f"claim {index}: states a cause without a hypothesis label")
        if SAVINGS.search(claim.text) and not bindings:
            errors.append(f"claim {index}: claims savings without a cited computed field")
        rows.append(
            {
                "claim": claim.text,
                "cited_evidence": cited,
                "numeric_bindings": bindings,
                "causal_language": bool(CAUSAL.search(claim.text)),
                "hypothesis_labeled": bool(HYPOTHESIS.search(claim.text)),
            }
        )
    if reply.status == "insufficient_evidence" and not any(
        UNCERTAINTY.search(claim.text) for claim in reply.claims
    ):
        errors.append("an insufficient-evidence reply requires an explicit uncertainty statement")
    if not evidence:
        errors.append("claim validation requires at least one evidence record")
    workspace = Path(evidence[0].path).parents[2] if evidence else Path(".")
    report = record(
        workspace / "artifacts/agent/grounding.json",
        {
            "status": "failed" if errors else "passed",
            "reply_status": reply.status,
            "claims": rows,
            "errors": errors,
            "evidence_sha256": sorted(available),
            "numeric_source": "cited evidence fields only; the LLM may not substitute numbers",
        },
    )
    if errors:
        raise ValueError(f"Claim validation failed: {'; '.join(errors)}")
    return report
