"""This module trains the student and preserves the frozen teacher."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torchvision.models import mobilenet_v3_small

from cycletime.contracts import JSON, EvidenceRef
from cycletime.dataio.archive import sha256
from cycletime.dataio.calibration import validate_partitions
from cycletime.dataio.images import load_image, preprocessing_digest
from cycletime.evidence import digest, read, record, reference, serialize
from cycletime.model.network import Detector


def load_checkpoint(checkpoint: EvidenceRef) -> tuple[Detector, dict]:
    """Load verified local weights; refuse changed checkpoint bytes."""
    data = read(checkpoint)
    path = Path(data["weights_path"])
    if sha256(path) != data["weights_sha256"]:
        raise ValueError("Checkpoint hash mismatch.")
    model = Detector()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval(), data


def train(model_config: JSON, dataset_config: JSON, partitions: EvidenceRef) -> EvidenceRef:
    """Train on normal fit images and record frozen teacher provenance; refuse missing ADR, unverified weights, test access, or architecture changes."""
    project = Path(str(dataset_config["_project"]))
    adr = project / str(model_config["adr"])
    teacher = model_config["teacher"]
    weights = project / teacher["local_path"]
    if not adr.is_file() or not (project / teacher["license_record"]).is_file():
        raise ValueError("The model requires its ADR and teacher rights record.")
    if sha256(weights) != teacher["sha256"] or not teacher["source_url"].startswith(
        "https://download.pytorch.org/models/"
    ):
        raise ValueError("Teacher provenance does not match the approved weights.")
    if (
        model_config["architecture"] != "convolutional_student_teacher"
        or model_config["backbone"] != "mobilenet_v3_small"
    ):
        raise ValueError("The model does not match the ADR.")
    if model_config["training"]["category"] != dataset_config["category"]:
        raise ValueError("Training category mismatch.")
    parts = validate_partitions(partitions)
    identity = digest(
        {
            "model": model_config,
            "dataset": dataset_config,
            "fit": parts["fit"],
            "validation": parts["validation"],
            "hashes": parts["file_hashes"],
            "adr": sha256(adr),
            "code": sha256(Path(__file__)),
            "network": sha256(Path(__file__).with_name("network.py")),
        }
    )
    destination = project / "artifacts/training/checkpoint.json"
    if destination.exists():
        cached = reference(destination)
        if read(cached)["training_identity"] != identity:
            raise ValueError(
                "Existing checkpoint belongs to a different experiment; record a new decision first."
            )
        load_checkpoint(cached)
        return cached
    config = model_config["training"]
    if config["device"] != "cpu" or config["epochs"] <= 0:
        raise ValueError(
            "This implementation requires configured CPU training and positive epochs."
        )
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.use_deterministic_algorithms(True)
    model = Detector(model_config["feature_stages"])
    pretrained = mobilenet_v3_small(weights=None)
    pretrained.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.teacher.load_state_dict(pretrained.features.state_dict())
    del pretrained
    root = project / str(dataset_config["root"])
    images = torch.from_numpy(
        np.stack([load_image(root / name, dataset_config) for name in parts["fit"]])
    )
    optimizer = torch.optim.Adam(model.student.parameters(), lr=config["learning_rate"])
    loader = torch.utils.data.DataLoader(
        images,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    model.train()
    with torch.no_grad():
        feature_shapes = [list(f.shape) for f in model.features(model.teacher.eval(), images[:1])]
    history = []
    for epoch in range(config["epochs"]):
        total = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack([d.mean() for d in model.distances(batch)]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        history.append({"epoch": epoch + 1, "fit_loss": total / len(images)})
        print(
            f"Training epoch {epoch + 1}/{config['epochs']}: fit loss {history[-1]['fit_loss']:.6f}",
            flush=True,
        )
        record(
            project / "artifacts/training/progress.json", {"history": history, "status": "training"}
        )
    model.eval()
    with torch.inference_mode():
        scales = torch.zeros(3)
        for batch in images.split(config["batch_size"]):
            scales += torch.stack([d.mean() for d in model.distances(batch)]) * len(batch)
        model.scales.copy_((scales / len(images)).clamp_min(1e-6))
    path = project / "artifacts/training/student_teacher.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return record(
        destination,
        {
            "weights_path": str(path),
            "weights_sha256": sha256(path),
            "training_identity": identity,
            "model_config": model_config,
            "dataset_config": dataset_config,
            "partitions": serialize(partitions),
            "history": history,
            "feature_shapes": feature_shapes,
            "normalization_scales": model.scales.tolist(),
            "normalization_source": "training_fit_partition",
            "preprocessing_sha256": preprocessing_digest(dataset_config),
            "dataset_sha256": dataset_config["_dataset_sha256"],
            "torch_version": torch.__version__,
            "teacher_sha256": teacher["sha256"],
            "adr_sha256": sha256(adr),
        },
    )
