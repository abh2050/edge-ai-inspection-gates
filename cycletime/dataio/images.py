"""This module applies shared deterministic image preprocessing without fitting on test data."""

from pathlib import Path

import numpy as np
from PIL import Image

from cycletime.dataio.archive import sha256
from cycletime.evidence import digest


def preprocessing_digest(config: dict) -> str:
    """Bind settings and preprocessing code; refuse unserializable settings."""
    return digest({"input": config["input"], "code": sha256(Path(__file__))})


def load_image(path: Path, config: dict) -> np.ndarray:
    """Decode RGB, resize bilinearly, and normalize; refuse unsupported preprocessing settings."""
    c = config["input"]
    if c["color"] != "RGB" or c["layout"] != "NCHW" or c["image_interpolation"] != "bilinear":
        raise ValueError("Unsupported preprocessing configuration.")
    with Image.open(path) as im:
        im = im.convert("RGB").resize(
            (c["resized_width"], c["resized_height"]), Image.Resampling.BILINEAR
        )
        x = np.asarray(im, dtype=np.float32) / np.float32(255)
    x = (x - np.asarray(c["mean"], np.float32)) / np.asarray(c["std"], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def decode_image(path: Path) -> Image.Image:
    """Decode an image completely before a benchmark clock starts; refuse lazy file access."""
    with Image.open(path) as image:
        decoded = image.convert("RGB")
        decoded.load()
    return decoded


def preprocess_decoded(image: Image.Image, config: dict) -> np.ndarray:
    """Resize and normalize an already decoded RGB image; refuse unsupported settings."""
    c = config["input"]
    if c["color"] != "RGB" or c["layout"] != "NCHW" or c["image_interpolation"] != "bilinear":
        raise ValueError("Unsupported preprocessing configuration.")
    resized = image.resize(
        (c["resized_width"], c["resized_height"]), Image.Resampling.BILINEAR
    )
    x = np.asarray(resized, dtype=np.float32) / np.float32(255)
    x = (x - np.asarray(c["mean"], np.float32)) / np.asarray(c["std"], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))
