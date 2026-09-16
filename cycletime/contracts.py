"""This module defines evidence records and refuses to create synthetic observations."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Precision = Literal["fp32", "fp16", "int8"]
JSON = dict[str, object]

@dataclass(frozen=True)
class Artifact:
    """This record identifies an export and does not attest to its accuracy."""
    path: Path
    sha256: str
    precision: Precision
    preprocessing_sha256: str
    dataset_sha256: str

@dataclass(frozen=True)
class EvidenceRef:
    """This record cites preserved evidence and does not verify its contents."""
    path: Path
    sha256: str
    created_at_utc: str
    run_id: str

@dataclass(frozen=True)
class PredictionCache:
    """This record locates one evaluation and does not authorize another run."""
    artifact: Artifact
    samples: EvidenceRef
    metrics: EvidenceRef
    evaluator_sha256: str

@dataclass(frozen=True)
class ProviderSpec:
    """This record requests execution settings and does not prove ANE placement."""
    id: str
    name: str
    options: dict[str, str]
    require_ane_evidence: bool

@dataclass(frozen=True)
class Measurement:
    """This record identifies measured samples and does not permit inferred timings."""
    artifact: Artifact
    provider: ProviderSpec
    samples: EvidenceRef
    environment: EvidenceRef
    placement: EvidenceRef | None
    summary: EvidenceRef

@dataclass(frozen=True)
class OperatingPoint:
    """This record identifies a cost decision and does not authorize station activation."""
    artifact_sha256: str
    provider_id: str
    threshold: float
    expected_cost_usd_per_shift: float
    recall: float
    false_reject_rate: float
    feasible: bool
    evidence: tuple[EvidenceRef, ...]
