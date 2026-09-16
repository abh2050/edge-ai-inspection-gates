"""cycletime/llm/client.py isolates the LLM transport from inspection."""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from cycletime.contracts import JSON, EvidenceRef
from cycletime.evidence import digest, read, record
from cycletime.llm.schemas import AgentReply

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".npz", ".npy", ".pt", ".onnx"}


class LLMUnavailableError(RuntimeError):
    """Signal that no verified reply exists; inspection continues without one."""


def pricing(costs: JSON) -> tuple[float, float]:
    """Return configured token prices; refuse to run without them."""
    llm = costs.get("llm") or {}
    prices = (llm.get("input_cost_usd_per_million_tokens"), llm.get("output_cost_usd_per_million_tokens"))
    if any(price is None for price in prices):
        raise ValueError("The LLM requires configured input and output token prices.")
    return float(prices[0]), float(prices[1])


def build_context(context: list[EvidenceRef]) -> list[JSON]:
    """Read cited evidence as JSON; refuse image bytes or model weights in model context."""
    payload = []
    for ref in context:
        if Path(ref.path).suffix.lower() in IMAGE_SUFFIXES:
            raise ValueError("Raw images and model weights must never reach the LLM.")
        payload.append({"sha256": ref.sha256, "path": Path(ref.path).name, "content": read(ref)})
    return payload


def generate(
    context: list[EvidenceRef],
    agent_config: JSON,
    costs: JSON,
    transport: Callable[[JSON], JSON] | None = None,
) -> AgentReply:
    """Call the explicitly configured provider adapter with validated structured output and bounded time, retries, tokens, and dollars; record provider model, prompt digest, usage, and citations; refuse missing pricing or credentials, raw images, benchmark concurrency, and fabricated fallback replies."""
    if not agent_config.get("enabled"):
        raise LLMUnavailableError("The agent is disabled; inspection continues offline.")
    if agent_config.get("measurement_in_flight"):
        raise ValueError("The LLM must not run during timed measurement work.")
    input_price, output_price = pricing(costs)
    endpoint = os.environ.get(str(agent_config["endpoint_env"]))
    api_key = os.environ.get(str(agent_config["api_key_env"]))
    model = os.environ.get(str(agent_config["model_env"]))
    if not endpoint or not api_key or not model:
        raise ValueError("The LLM requires a configured endpoint, model, and credential.")
    if agent_config["wire_protocol"] != "provider_adapter_required":
        raise ValueError("The configured wire protocol is unsupported.")
    if transport is None:
        raise LLMUnavailableError(
            "No provider adapter is configured; the workflow records the outage instead of inventing a reply."
        )
    payload = build_context(context)
    prompt = {
        "model": model,
        "temperature": agent_config["sampling_temperature"],
        "require_structured_output": agent_config["require_structured_output"],
        "schema": AgentReply.model_json_schema(),
        "evidence": payload,
    }
    prompt_digest = digest(prompt)
    attempts = []
    started = perf_counter()
    for attempt in range(int(agent_config["max_retries"]) + 1):
        elapsed = perf_counter() - started
        if elapsed > float(agent_config["timeout_seconds"]):
            attempts.append({"attempt": attempt, "error": "timeout"})
            break
        try:
            raw = transport(prompt)
            usage = raw.get("usage", {})
            used_in = int(usage.get("input_tokens", 0))
            used_out = int(usage.get("output_tokens", 0))
            if used_in > int(agent_config["max_input_tokens"]) or used_out > int(
                agent_config["max_output_tokens"]
            ):
                raise ValueError("The reply exceeded its configured token budget.")
            spend = used_in / 1e6 * input_price + used_out / 1e6 * output_price
            if spend > float((costs.get("llm") or {}).get("max_cost_usd_per_run", 0)):
                raise ValueError("The reply exceeded the configured dollar limit.")
            reply = AgentReply(**(raw["reply"] if isinstance(raw.get("reply"), dict) else json.loads(raw["reply"])))
            if agent_config["require_evidence_citations"] and any(
                not claim.evidence_sha256 for claim in reply.claims
            ):
                raise ValueError("Every claim requires at least one evidence citation.")
            record(
                Path(str(agent_config["_workspace"])) / "artifacts/agent/llm-calls" / f"{prompt_digest[:16]}.json",
                {
                    "provider_model": model,
                    "prompt_digest": prompt_digest,
                    "attempts": attempts + [{"attempt": attempt, "status": "accepted"}],
                    "usage": {"input_tokens": used_in, "output_tokens": used_out},
                    "estimated_cost_usd": spend,
                    "token_prices_usd_per_million": [input_price, output_price],
                    "cited_evidence_sha256": sorted(
                        {sha for claim in reply.claims for sha in claim.evidence_sha256}
                    ),
                    "context_evidence_sha256": [item["sha256"] for item in payload],
                    "raw_images_included": False,
                },
            )
            return reply
        except (ValidationError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            attempts.append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})
    record(
        Path(str(agent_config["_workspace"])) / "artifacts/agent/llm-calls" / f"{prompt_digest[:16]}-failed.json",
        {
            "provider_model": model,
            "prompt_digest": prompt_digest,
            "attempts": attempts,
            "outcome": "bounded_retries_exhausted_without_a_valid_reply",
        },
    )
    raise LLMUnavailableError("The provider returned no valid structured reply within its retry budget.")
