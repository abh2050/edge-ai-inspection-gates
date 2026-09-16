"""cycletime/gates.py defines each acceptance gate and refuses unfinished execution."""
from __future__ import annotations

from pathlib import Path

from cycletime.contracts import EvidenceRef


def gate0(config_dir: Path) -> EvidenceRef:
    """This function must verify approved hashes, fifteen categories, and defect-free training. It must refuse missing dataset provenance and skipped checks."""
    from cycletime.config import load_config
    from cycletime.dataio.verify import verify_dataset

    config_dir = config_dir.resolve()
    config = load_config(config_dir / "dataset.yaml")
    return verify_dataset(config_dir.parent / str(config["manifest"]), config)

def gate1(config_dir: Path) -> EvidenceRef:
    """This function must validate Gate 0, train one category, export canonical FP32, freeze the initial threshold, evaluate FP32 once, and enforce AUROC floors. It must refuse training on test data and false success from a stub."""
    from cycletime.model.pipeline import run

    return run(config_dir)

def gate2(config_dir: Path) -> EvidenceRef:
    """This function must validate Gate 1, reuse FP32, export FP16 and INT8, check all output parity, evaluate each new artifact once, and enforce AUROC floors. It must refuse skipped precision checks or additional FP32 evaluation."""
    from cycletime.export.pipeline import run

    return run(config_dir)

def gate3(config_dir: Path) -> EvidenceRef:
    """This function must validate Gate 2, attempt every artifact/provider pair, check provider parity and placement, benchmark, and write docs/latency.md with a failure record for unsupported pairs. It must refuse passing incomplete matrices or untimestamped latency."""
    from cycletime.bench.pipeline import run

    return run(config_dir)

def gate4(config_dir: Path) -> EvidenceRef:
    """This function must validate Gate 3 and run a separate 30-minute target-rate test for every required pair; write raw samples and each minute p99. It must refuse overlapping runs, missing windows, or a pass after missed pacing requirements."""
    from cycletime.bench.sustained_pipeline import run

    return run(config_dir)

def gate5(config_dir: Path) -> EvidenceRef:
    """This function must validate prior gate digests, compute configured costs from cached predictions, select a feasible point, draw curves, and write docs/scorecard.md. It must refuse stale evidence, no feasible point, and presenting retrospective research as release acceptance."""
    from cycletime.cost.pipeline import run

    return run(config_dir)

def release_check(config_dir: Path) -> EvidenceRef:
    """This function must require commercial data rights, independent customer validation, signed artifacts, shift soak tests, fault recovery, and acceptance evidence. It must refuse null production requirements, passing research gates as release qualification, and activation."""
    from cycletime.production.release_check import run

    return run(config_dir)
