"""cycletime/agent/runner.py checkpoints a bounded plan-act-observe workflow."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from cycletime.agent.policy import LONG_MEASUREMENT_GATES, PolicyError, authorize, request_key
from cycletime.agent.tools import execute_tool
from cycletime.contracts import JSON, EvidenceRef
from cycletime.evidence import digest, read, record, reference, restore, serialize
from cycletime.llm.schemas import ToolRequest

GATE_ORDER = ("gate0", "gate1", "gate2", "gate3", "gate4", "gate5")


def audit(path: Path, entry: JSON) -> None:
    """Append one immutable audit line; the trail never rewrites an earlier entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(entry, utc=datetime.now(UTC).isoformat()), sort_keys=True) + "\n")


def inventory(workspace: Path) -> JSON:
    """Summarize recorded gate evidence without recomputing or interpreting measurements."""
    found = {}
    for name in GATE_ORDER:
        path = workspace / f"artifacts/{name}.json"
        found[name] = (
            {"status": json.loads(path.read_text())["status"], "path": f"artifacts/{name}.json"}
            if path.is_file()
            else None
        )
    return found


def propose(config: JSON, evidence: list[EvidenceRef]) -> EvidenceRef:
    """Draft a typed experiment plan from the current evidence and list missing prerequisites; refuse executing measurements, relaxing gates, and inventing outcomes."""
    workspace = Path(str(config["_workspace"])).resolve()
    recorded = inventory(workspace)
    missing = [name for name, row in recorded.items() if row is None or row["status"] != "passed"]
    steps: list[JSON] = []
    for name in missing:
        number = int(name[-1])
        steps.append(
            {
                "tool": "run_gate",
                "arguments": {
                    "gate": number,
                    "tenant_id": config["_tenant_id"],
                    "expected_wall_seconds": 5400 if number in LONG_MEASUREMENT_GATES else 600,
                },
                "reason": f"{name} has no passed evidence",
                "requires_long_measurement_authorization": number in LONG_MEASUREMENT_GATES,
            }
        )
    if not missing:
        steps.append(
            {
                "tool": "render_explanation",
                "arguments": {
                    "question": "Which measured configuration satisfies the line budget, and why does the cost-selected threshold differ from the maximum-F1 threshold?",
                    "evidence_sha256": [ref.sha256 for ref in evidence],
                    "tenant_id": config["_tenant_id"],
                },
                "reason": "every gate recorded passed evidence",
                "requires_long_measurement_authorization": False,
            }
        )
    plan = {
        "tenant_id": config["_tenant_id"],
        "gate_inventory": recorded,
        "missing_prerequisites": missing,
        "steps": steps,
        "context_evidence": [serialize(ref) for ref in evidence],
        "authorization_status": "unauthorized_draft",
        "execution_refused": "the planner drafts steps and never executes a measurement",
        "outcome_estimates_refused": "the planner never states an unmeasured result",
        "gate_relaxation_refused": "the planner never proposes changing a gate, tolerance, or cost",
    }
    plan["plan_digest"] = digest(
        {"tenant_id": plan["tenant_id"], "steps": plan["steps"], "missing": missing}
    )
    return record(workspace / str(config["execution"]["proposal_path"]), plan)


def load_state(path: Path) -> JSON:
    """Resume recorded progress so a restart repeats no completed tool call."""
    if not path.is_file():
        return {
            "steps": 0,
            "tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "wall_seconds": 0,
            "completed_tool_keys": [],
            "reserved_evaluation_gates": [],
            "results": [],
        }
    return json.loads(path.read_text())


def run(config: JSON, authorized_plan: EvidenceRef) -> EvidenceRef:
    """Run plan-act-observe-verify steps through policy and allowlisted tools; persist checkpoints and an append-only audit trail with idempotency keys; refuse unapproved long measurements, unbounded loops, repeating in-flight evaluation, and calling an LLM during timed work."""
    workspace = Path(str(config["_workspace"])).resolve()
    execution = config["execution"]
    checkpoint_path = workspace / str(execution["checkpoint_path"])
    audit_path = workspace / str(execution["audit_path"])
    plan = read(authorized_plan)
    if plan.get("authorization_status") != "authorized_by_operator":
        raise ValueError("Gate execution requires an operator-authorized plan.")
    state = load_state(checkpoint_path)
    state["plan_digest"] = plan["plan_digest"]
    audit(audit_path, {"event": "run_started", "plan_digest": plan["plan_digest"], "resumed_steps": state["steps"]})
    for step in plan["steps"]:
        if state["steps"] >= int(config["max_steps"]):
            audit(audit_path, {"event": "halted", "reason": "step budget exhausted"})
            break
        request = ToolRequest(tool=step["tool"], arguments=step["arguments"], reason=step["reason"])
        key = request_key(request)
        if key in state["completed_tool_keys"]:
            audit(audit_path, {"event": "skipped_completed_step", "idempotency_key": key})
            continue
        state["steps"] += 1
        try:
            authorization = authorize(request, config, authorized_plan, state)
        except PolicyError as exc:
            audit(audit_path, {"event": "policy_refused", "reason": exc.reason, "tool": request.tool, "idempotency_key": key})
            state["results"].append({"idempotency_key": key, "status": "refused", "reason": exc.reason})
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
            continue
        measurement = request.tool == "run_gate"
        if measurement:
            # A reservation survives a crash, so a resumed run never repeats inference in flight.
            state["reserved_evaluation_gates"].append(int(request.arguments["gate"]))
            config = dict(config, measurement_in_flight=True)
            checkpoint_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        audit(audit_path, {"event": "tool_authorized", "tool": request.tool, "idempotency_key": key})
        result = execute_tool(request, authorization, config)
        observed = read(result)
        if measurement:
            config = dict(config, measurement_in_flight=False)
            state["reserved_evaluation_gates"].remove(int(request.arguments["gate"]))
        state["tool_calls"] += 1
        if observed["status"] == "completed":
            state["completed_tool_keys"].append(key)
        state["results"].append({"idempotency_key": key, "status": observed["status"], "result": serialize(result)})
        audit(
            audit_path,
            {
                "event": "tool_observed",
                "tool": request.tool,
                "idempotency_key": key,
                "status": observed["status"],
                "error": observed["error"],
                "evidence": str(result.path.relative_to(workspace)),
            },
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        if observed["status"] == "failed":
            audit(audit_path, {"event": "halted", "reason": "tool failed", "idempotency_key": key})
            break
    report = record(
        workspace / "artifacts/agent/run.json",
        {
            "plan": serialize(authorized_plan),
            "plan_digest": plan["plan_digest"],
            "tenant_id": plan["tenant_id"],
            "steps_taken": state["steps"],
            "tool_calls": state["tool_calls"],
            "results": state["results"],
            "checkpoint": str(checkpoint_path.relative_to(workspace)),
            "audit_trail": str(audit_path.relative_to(workspace)),
            "measurements_unchanged": True,
            "acceptance_criteria_unchanged": True,
        },
    )
    audit(audit_path, {"event": "run_completed", "evidence": "artifacts/agent/run.json"})
    return report


def authorize_plan(proposal: EvidenceRef, workspace: Path) -> EvidenceRef:
    """Record an operator's authorization of a drafted plan; the agent never authorizes its own plan."""
    plan = read(proposal)
    return record(
        workspace / "artifacts/agent/authorized-plan.json",
        dict(plan, authorization_status="authorized_by_operator", authorized_from=serialize(proposal)),
    )


def restore_plan(reference_json: JSON) -> EvidenceRef:
    """Restore a recorded plan reference."""
    return restore(reference_json)


def plan_reference(path: Path) -> EvidenceRef:
    """Reference a recorded plan on disk."""
    return reference(path)
