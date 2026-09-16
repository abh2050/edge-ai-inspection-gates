"""This module dispatches commands and refuses to turn failures into passes."""
import argparse
import sys
from pathlib import Path

from cycletime import gates
from cycletime.agent.runner import propose
from cycletime.config import load_config
from cycletime.dataio.fetch import fetch_dataset


def main() -> int:
    """This function must dispatch explicit commands; it must refuse unsupported commands and missing authorization."""
    parser = argparse.ArgumentParser(prog="cycletime")
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch")
    gate = sub.add_parser("gate")
    gate.add_argument("number", type=int, choices=range(6))
    agent = sub.add_parser("agent")
    agent.add_argument("action", choices=["plan", "run"])
    sub.add_parser("release-check")
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            fetch_dataset(Path("data/manifest.json"), Path("data/raw"))
        elif args.command == "gate":
            functions = (gates.gate0, gates.gate1, gates.gate2, gates.gate3, gates.gate4, gates.gate5)
            evidence = functions[args.number](args.config_dir)
            print(f"Gate {args.number} passed. Evidence: {evidence.path}")
        elif args.command == "release-check":
            gates.release_check(args.config_dir)
        else:
            from cycletime.agent.runner import run as run_agent
            from cycletime.evidence import read, reference

            config = load_config(args.config_dir / "agent.yaml")
            if config.get("enabled") is not True:
                raise ValueError("Agent execution requires explicit configuration.")
            workspace = args.config_dir.resolve().parent
            config["_workspace"] = str(workspace)
            config["_tenant_id"] = config.get("tenant_id", "local-research")
            config["_costs"] = load_config(args.config_dir / "costs.yaml")
            if args.action == "plan":
                evidence = reference(workspace / "artifacts/gate5.json")
                print(f"Proposal: {propose(config, [evidence]).path}")
            else:
                authorization = config["execution"]["authorization_path"]
                if not authorization:
                    raise ValueError(
                        "Agent execution requires an operator-authorized plan in "
                        "config/agent.yaml execution.authorization_path."
                    )
                plan = reference(workspace / str(authorization))
                if read(plan).get("authorization_status") != "authorized_by_operator":
                    raise ValueError("The recorded plan lacks operator authorization.")
                print(f"Run evidence: {run_agent(config, plan).path}")
    except (NotImplementedError, ValueError, FileNotFoundError) as exc:
        print(f"Command failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
