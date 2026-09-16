"""This module creates ORT sessions and records observed provider placement."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnxruntime as ort

from cycletime.contracts import JSON, Artifact, EvidenceRef, ProviderSpec
from cycletime.evidence import record
from cycletime.export.metadata import load_artifact


def create_session(
    artifact: Artifact, provider: ProviderSpec, config: JSON, scope: str = "bench"
) -> tuple[object, EvidenceRef]:
    """Create a synchronous session and record observed ORT placement; refuse unsupported providers and any unsupported ANE claim."""
    verified, _ = load_artifact(artifact.path)
    if verified != artifact:
        raise ValueError("Artifact provenance changed before provider setup.")
    if provider.name not in ort.get_available_providers():
        raise ValueError(f"Required provider is unavailable: {provider.name}")
    if config["concurrency"] != 1:
        raise ValueError("The benchmark requires single-request concurrency.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(config["intra_op_threads"])
    options.inter_op_num_threads = int(config["inter_op_threads"])
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_profiling = True
    options.profile_file_prefix = str(
        artifact.path.parent / f"profile-{artifact.path.stem}-{provider.id}"
    )
    session = ort.InferenceSession(
        str(artifact.path),
        sess_options=options,
        providers=[(provider.name, provider.options)],
    )
    active = session.get_providers()
    if provider.name not in active:
        raise ValueError(f"ORT did not activate the required provider: {provider.name}")
    model_input = session.get_inputs()[0]
    if model_input.name != "image" or model_input.shape != [1, 3, 256, 256]:
        raise ValueError("The benchmark requires the static batch-one input contract.")
    session.run(None, {"image": np.zeros((1, 3, 256, 256), dtype=np.float32)})
    profile_source = Path(session.end_profiling())
    events = json.loads(profile_source.read_text())
    counts: dict[str, int] = {}
    assignments = []
    for event in events:
        assigned = event.get("args", {}).get("provider")
        if assigned:
            counts[assigned] = counts.get(assigned, 0) + 1
            assignments.append(
                {
                    "name": event.get("name"),
                    "provider": assigned,
                    "op_name": event.get("args", {}).get("op_name"),
                }
            )
    # Each measurement scope keeps its own placement evidence so one gate cannot overwrite another's.
    scope_dir = artifact.path.parents[1] / ("placement" if scope == "bench" else f"placement/{scope}")
    profile_dir = scope_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / f"{artifact.path.stem}-{provider.id}.json"
    profile_source.replace(profile_path)
    provider_config = next(item for item in config["providers"] if item["id"] == provider.id)
    default_trace = artifact.path.parents[1] / "placement/ane-evidence.json"
    configured_trace = provider_config.get("placement_evidence_path")
    ane_requested = provider.options.get("MLComputeUnits") == "CPUAndNeuralEngine"
    if (
        (provider.require_ane_evidence or ane_requested)
        and not configured_trace
        and default_trace.is_file()
    ):
        configured_trace = str(default_trace)
    ane_confirmed = False
    external_trace = None
    ane_power_samples: list[float] = []
    if configured_trace:
        external_trace = Path(configured_trace)
        if not external_trace.is_absolute():
            external_trace = artifact.path.parents[2] / external_trace
        trace = json.loads(external_trace.read_text())
        entry = trace.get("artifacts", {}).get(artifact.sha256, trace)
        if (
            entry.get("artifact_sha256") != artifact.sha256
            or entry.get("provider_id") != provider.id
            or entry.get("provider_name") != provider.name
            or entry.get("requested_options") != provider.options
        ):
            raise ValueError("ANE trace does not match the artifact and provider options.")
        trace_path = Path(entry["trace_path"])
        if not trace_path.is_absolute():
            trace_path = artifact.path.parents[2] / trace_path
        if sha256(trace_path.read_bytes()).hexdigest() != entry["trace_sha256"]:
            raise ValueError("ANE placement trace changed after capture.")
        ane_power_samples = [float(value) for value in entry["ane_power_samples_mw"]]
        measured_active = bool(ane_power_samples and any(value > 0 for value in ane_power_samples))
        if measured_active != bool(entry.get("ane_active")):
            raise ValueError("ANE placement summary conflicts with its raw power samples.")
        ane_confirmed = bool(
            measured_active
            and entry.get("artifact_sha256") == artifact.sha256
            and entry.get("provider_id") == provider.id
        )
    cpu_nodes = counts.get("CPUExecutionProvider", 0)
    coreml_nodes = counts.get("CoreMLExecutionProvider", 0)
    fallback_allowed = bool(provider_config.get("allow_ort_cpu_fallback", False))
    fallback_observed = provider.name != "CPUExecutionProvider" and cpu_nodes > 0
    if provider.name == "CPUExecutionProvider":
        label = "CPUExecutionProvider"
        placement_passed = cpu_nodes > 0
    else:
        partitions = (
            "CoreML mixed placement; ORT CPU nodes observed"
            if fallback_observed
            else "CoreML placement; no ORT CPU nodes observed"
        )
        if ane_confirmed:
            ane = "ANE activity confirmed"
        elif external_trace:
            ane = "ANE telemetry measured 0 mW"
        else:
            ane = "ANE requested but unconfirmed"
        label = f"{partitions}; {ane}"
        placement_passed = (
            coreml_nodes > 0
            and (fallback_allowed or not fallback_observed)
            and (not provider.require_ane_evidence or ane_confirmed)
        )
    evidence = record(
        scope_dir / f"{artifact.path.stem}-{provider.id}.json",
        {
            "artifact_sha256": artifact.sha256,
            "provider_id": provider.id,
            "requested_provider": provider.name,
            "requested_options": provider.options,
            "active_providers": active,
            "profile_provider_event_counts": counts,
            "profile_assignments": assignments,
            "profile_path": str(profile_path),
            "profile_sha256": __import__("hashlib").sha256(profile_path.read_bytes()).hexdigest(),
            "coreml_event_count": coreml_nodes,
            "ort_cpu_event_count": cpu_nodes,
            "ort_cpu_fallback_observed": fallback_observed,
            "ort_cpu_fallback_allowed": fallback_allowed,
            "ane_required": provider.require_ane_evidence,
            "coreml_internal_cpu_possible": provider.name == "CoreMLExecutionProvider",
            "ane_requested": ane_requested,
            "ane_confirmed": ane_confirmed,
            "ane_power_sample_count": len(ane_power_samples),
            "ane_power_max_mw": max(ane_power_samples, default=None),
            "external_trace": str(external_trace) if external_trace else None,
            "placement_label": label,
            "placement_passed": placement_passed,
            "onnxruntime_version": ort.__version__,
            "measurement_scope": scope,
        },
    )
    return session, evidence
