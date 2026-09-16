"""cycletime/agent/tools.py maps approved requests to deterministic local functions."""
from __future__ import annotations

from pathlib import Path

from cycletime.agent.policy import inside_workspace, request_key
from cycletime.contracts import JSON, EvidenceRef
from cycletime.evidence import read, record, reference, serialize
from cycletime.llm.schemas import TOOL_ARGUMENTS, ToolRequest


def tool_read_evidence(arguments: JSON, config: JSON) -> JSON:
    """Read one recorded evidence file from the workspace."""
    workspace = Path(str(config["_workspace"])).resolve()
    path = (workspace / str(arguments["path"])).resolve()
    if not inside_workspace(path, workspace) or not path.is_file():
        raise ValueError("The tool reads recorded evidence inside the workspace only.")
    ref = reference(path)
    return {"evidence": serialize(ref), "content": read(ref)}


def tool_retrieve_procedure(arguments: JSON, config: JSON) -> JSON:
    """Retrieve approved passages for the caller's own tenant."""
    from cycletime.llm.retrieval import retrieve_procedure

    manifest = config["knowledge"]["approved_manifest"]
    if not manifest:
        raise ValueError("Retrieval requires a configured approved manifest.")
    workspace = Path(str(config["_workspace"])).resolve()
    refs = retrieve_procedure(str(arguments["query"]), str(arguments["tenant_id"]), workspace / manifest)
    return {"passages": [serialize(ref) for ref in refs], "content_status": "untrusted_document_data"}


def tool_propose_experiment(arguments: JSON, config: JSON) -> JSON:
    """Describe one bounded experiment without scheduling or estimating its latency."""
    return {
        "question": arguments["question"],
        "gate": arguments["gate"],
        "protocol": "the recorded gate protocol defines warmups, samples, and duration",
        "estimated_latency_ms": None,
        "estimation_refused": "latency comes only from a measured run",
        "requires_operator_authorization": True,
    }


def tool_run_gate(arguments: JSON, config: JSON) -> JSON:
    """Run one allowlisted gate through its recorded pipeline."""
    from cycletime import gates

    number = int(arguments["gate"])
    if number not in config["execution"]["gate_allowlist"]:
        raise ValueError(f"Gate {number} is outside the allowlist.")
    function = (gates.gate0, gates.gate1, gates.gate2, gates.gate3, gates.gate4, gates.gate5)[number]
    evidence = function(Path(str(config["_workspace"])) / "config")
    return {"gate": number, "evidence": serialize(evidence)}


def tool_render_explanation(arguments: JSON, config: JSON) -> JSON:
    """Draft a cited explanation from verified evidence after measurement completes."""
    from cycletime.llm.client import generate
    from cycletime.llm.grounding import validate_claims

    workspace = Path(str(config["_workspace"])).resolve()
    refs = [reference(workspace / path) for path in config["_evidence_paths"]]
    cited = [ref for ref in refs if ref.sha256 in set(arguments["evidence_sha256"])]
    if not cited:
        raise ValueError("The explanation requires verified evidence that the workspace records.")
    reply = generate(cited, config, config["_costs"], transport=config.get("_transport"))
    grounding = validate_claims(reply, cited)
    return {
        "question": arguments["question"],
        "claims": [claim.model_dump() for claim in reply.claims],
        "status": reply.status,
        "grounding": serialize(grounding),
    }


DISPATCH = {
    "read_evidence": tool_read_evidence,
    "retrieve_procedure": tool_retrieve_procedure,
    "propose_experiment": tool_propose_experiment,
    "run_gate": tool_run_gate,
    "render_explanation": tool_render_explanation,
}


def execute_tool(request: ToolRequest, authorization: EvidenceRef, config: JSON) -> EvidenceRef:
    """Dispatch an allowlisted function after validating the authorization and arguments; return structured evidence and preserve failures; refuse dynamic imports from model text, arbitrary commands, self-authorization, and network destinations outside configuration."""
    workspace = Path(str(config["_workspace"])).resolve()
    granted = read(authorization)
    key = request_key(request)
    if granted.get("authorized_by") != "cycletime.agent.policy":
        raise ValueError("Only the policy module authorizes a tool call.")
    if granted.get("idempotency_key") != key or granted.get("tool") != request.tool:
        raise ValueError("The authorization does not cover this exact request.")
    if request.tool not in DISPATCH or request.tool not in config["allowed_tools"]:
        raise ValueError(f"The tool {request.tool} has no allowlisted implementation.")
    # Arguments are revalidated here; model text never names a module, path, or command to import.
    arguments = TOOL_ARGUMENTS[request.tool](**request.arguments).model_dump()
    destination = f"artifacts/agent/tool-results/{key[:16]}.json"
    try:
        result = DISPATCH[request.tool](arguments, config)
        status = "completed"
        error = None
    except Exception as exc:  # noqa: BLE001 - every failed tool call stays in the audit trail.
        result = None
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    return record(
        workspace / destination,
        {
            "tool": request.tool,
            "arguments": arguments,
            "idempotency_key": key,
            "authorization": serialize(authorization),
            "status": status,
            "result": result,
            "error": error,
            "network_destinations": config.get("network_destinations_from_explicit_config_only", True)
            and "configured endpoints only",
        },
    )
