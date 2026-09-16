"""This module defines LLM schemas and refuses unrestricted tool names."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Claim(BaseModel):
    """This record carries a cited claim and does not validate its truth."""
    model_config = ConfigDict(extra="forbid")
    text: str
    evidence_sha256: list[str] = Field(min_length=1)

class ToolRequest(BaseModel):
    """This record restricts tool names and does not authorize execution."""
    model_config = ConfigDict(extra="forbid")
    tool: Literal["read_evidence", "retrieve_procedure", "propose_experiment", "run_gate", "render_explanation"]
    arguments: dict[str, object]
    reason: str

class AgentReply(BaseModel):
    """This record requires a bounded response and does not approve measurements."""
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim]
    request: ToolRequest | None
    status: Literal["request_tool", "complete", "insufficient_evidence"]


class ReadEvidenceArguments(BaseModel):
    """This record names one workspace evidence file and does not grant access."""
    model_config = ConfigDict(extra="forbid")
    path: str
    tenant_id: str


class RetrieveProcedureArguments(BaseModel):
    """This record carries a retrieval query and does not approve any document."""
    model_config = ConfigDict(extra="forbid")
    query: str
    tenant_id: str


class ProposeExperimentArguments(BaseModel):
    """This record describes a proposed experiment and does not schedule work."""
    model_config = ConfigDict(extra="forbid")
    question: str
    gate: int
    tenant_id: str


class RunGateArguments(BaseModel):
    """This record names one allowlisted gate and does not authorize long measurement."""
    model_config = ConfigDict(extra="forbid")
    gate: int
    tenant_id: str
    expected_wall_seconds: int = Field(ge=0)


class RenderExplanationArguments(BaseModel):
    """This record names verified evidence for a cited explanation and does not edit numbers."""
    model_config = ConfigDict(extra="forbid")
    question: str
    evidence_sha256: list[str] = Field(min_length=1)
    tenant_id: str


TOOL_ARGUMENTS = {
    "read_evidence": ReadEvidenceArguments,
    "retrieve_procedure": RetrieveProcedureArguments,
    "propose_experiment": ProposeExperimentArguments,
    "run_gate": RunGateArguments,
    "render_explanation": RenderExplanationArguments,
}
