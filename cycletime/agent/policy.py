"""cycletime/agent/policy.py authorizes bounded tool arguments against a recorded plan."""
from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from cycletime.contracts import JSON, EvidenceRef
from cycletime.evidence import digest, read, record
from cycletime.llm.schemas import TOOL_ARGUMENTS, ToolRequest

LONG_MEASUREMENT_GATES = (4,)


class PolicyError(ValueError):
    """Carry a refused request so the audit trail records the attempt."""

    def __init__(self, reason: str, request: ToolRequest):
        self.reason = reason
        self.request = request
        super().__init__(reason)


def request_key(request: ToolRequest) -> str:
    """Derive one idempotency key from the tool and its exact arguments."""
    return digest({"tool": request.tool, "arguments": request.arguments})


def inside_workspace(candidate: Path, workspace: Path) -> bool:
    """Return whether a resolved path stays inside the workspace without following links outward."""
    try:
        resolved = (workspace / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    except OSError:
        return False
    return workspace.resolve() in resolved.parents or resolved == workspace.resolve()


def authorize(
    request: ToolRequest, config: JSON, authorized_plan: EvidenceRef, state: JSON
) -> EvidenceRef:
    """Validate a per-tool argument schema, workspace paths, remaining budgets, tenant scope, and the authorized plan digest; refuse shell commands, mutable cost or threshold changes, deployment, extra gates, and repeated reserved evaluations."""
    workspace = Path(str(config["_workspace"])).resolve()
    plan = read(authorized_plan)
    if config["execution"]["require_authorized_plan_digest"] and plan.get(
        "plan_digest"
    ) != state.get("plan_digest"):
        raise PolicyError("The request does not match the authorized plan digest.", request)
    if plan.get("authorization_status") != "authorized_by_operator":
        raise PolicyError("The plan lacks recorded operator authorization.", request)
    if request.tool not in config["allowed_tools"]:
        raise PolicyError(f"The tool {request.tool} is not allowlisted.", request)
    forbidden = set(config["forbidden_actions"])
    text = f"{request.tool} {request.reason} {request.arguments}".lower()
    for action in forbidden:
        if action.replace("_", " ") in text or action in text:
            raise PolicyError(f"The request names the forbidden action {action}.", request)
    try:
        arguments = TOOL_ARGUMENTS[request.tool](**request.arguments)
    except ValidationError as exc:
        raise PolicyError(f"The arguments fail the {request.tool} schema: {exc}", request) from exc
    tenant = getattr(arguments, "tenant_id", None)
    if config["knowledge"]["enforce_tenant_scope"] and tenant != plan["tenant_id"]:
        raise PolicyError("The request leaves the authorized tenant scope.", request)
    if request.tool == "read_evidence" and not inside_workspace(Path(arguments.path), workspace):
        raise PolicyError("The request reads a path outside the workspace.", request)
    if request.tool == "run_gate":
        if arguments.gate not in config["execution"]["gate_allowlist"]:
            raise PolicyError(f"Gate {arguments.gate} is outside the allowlist.", request)
        long_measurement = (
            arguments.gate in LONG_MEASUREMENT_GATES
            or arguments.expected_wall_seconds > int(config["max_wall_seconds"])
        )
        if long_measurement and not config["allow_long_measurements"]:
            raise PolicyError("A long measurement requires explicit authorization.", request)
        if arguments.expected_wall_seconds > int(config["max_measurement_wall_seconds"]):
            raise PolicyError("The request exceeds the measurement wall-clock limit.", request)
    budgets = {
        "steps": (state.get("steps", 0), int(config["max_steps"])),
        "tool_calls": (state.get("tool_calls", 0), int(config["max_tool_calls"])),
        "input_tokens": (state.get("input_tokens", 0), int(config["max_input_tokens"])),
        "output_tokens": (state.get("output_tokens", 0), int(config["max_output_tokens"])),
        "wall_seconds": (state.get("wall_seconds", 0), int(config["max_wall_seconds"])),
    }
    for name, (used, limit) in budgets.items():
        if used >= limit:
            raise PolicyError(f"The workflow exhausted its {name} budget.", request)
    key = request_key(request)
    completed = state.get("completed_tool_keys", [])
    if key in completed:
        raise PolicyError("The workflow already completed this exact request.", request)
    if request.tool == "run_gate" and arguments.gate in state.get("reserved_evaluation_gates", []):
        raise PolicyError("A reserved evaluation for this gate is already in flight.", request)
    return record(
        workspace / "artifacts/agent/authorizations" / f"{key[:16]}.json",
        {
            "tool": request.tool,
            "arguments": request.arguments,
            "reason": request.reason,
            "idempotency_key": key,
            "plan_digest": plan["plan_digest"],
            "tenant_id": tenant,
            "budgets_remaining": {name: limit - used for name, (used, limit) in budgets.items()},
            "authorized_by": "cycletime.agent.policy",
            "authorization_scope": "one_execution_of_this_exact_request",
        },
    )
