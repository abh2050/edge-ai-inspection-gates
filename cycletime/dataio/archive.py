"""This module verifies local archives and refuses untrusted or unsafe extraction."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


def sha256(path: Path) -> str:
    """Hash an existing regular file; refuse symbolic links."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def approved_manifest(path: Path) -> dict:
    """Load recorded independent provenance; refuse missing hashes or unapproved URLs."""
    data = json.loads(path.read_text())
    digest = data.get("archive_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("The manifest requires an independently approved SHA256.")
    for key in ("digest_source", "verified_by", "verified_at_utc"):
        if not data.get(key):
            raise ValueError(f"Missing provenance: {key}")
    url = urlparse(data.get("archive_url") or "")
    if url.scheme != "https" or url.hostname not in data.get("approved_download_hosts", []):
        raise ValueError("The archive URL must use HTTPS and an approved host.")
    return data


def inspect_archive(
    archive: Path, expected_sha256: str, destination: Path | None = None
) -> dict[str, str]:
    """Hash archive members and optionally extract into an empty staging directory; refuse digest mismatches, links, special files, traversal, and duplicate paths."""
    if sha256(archive) != expected_sha256:
        raise ValueError("The archive SHA256 does not match the approved manifest.")
    if destination is not None and (destination.is_symlink() or any(destination.iterdir())):
        raise ValueError("Extraction requires an empty staging directory.")
    hashes = {}
    seen = set()
    with tarfile.open(archive, "r|*") as stream:
        for member in stream:
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or not (member.isfile() or member.isdir())
                or "\\" in member.name
            ):
                raise ValueError(f"Unsafe archive member: {member.name}")
            name = relative.as_posix()
            if name in seen:
                raise ValueError(f"Duplicate archive member: {name}")
            seen.add(name)
            if member.isdir():
                if destination is not None:
                    (destination / name).mkdir(parents=True, exist_ok=True)
                continue
            source = stream.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable archive member: {name}")
            digest = hashlib.sha256()
            if destination is not None:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        output.write(chunk)
            else:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            hashes[name] = digest.hexdigest()
    if not hashes:
        raise ValueError("The archive is empty.")
    return hashes


def require_extraction_space(archive: Path, expected_sha256: str, parent: Path) -> None:
    """Check expanded archive size against free space; refuse insufficient capacity before extraction."""
    if sha256(archive) != expected_sha256:
        raise ValueError("The archive SHA256 does not match the approved manifest.")
    with tarfile.open(archive, "r|*") as stream:
        expanded_bytes = sum(member.size for member in stream if member.isfile())
    reserve = 512 * 1024 * 1024
    available = shutil.disk_usage(parent).free
    if available < expanded_bytes + reserve:
        raise ValueError(
            f"Insufficient disk space: extraction needs {expanded_bytes / 2**30:.2f} GiB "
            f"plus 0.50 GiB reserve; only {available / 2**30:.2f} GiB is free."
        )
