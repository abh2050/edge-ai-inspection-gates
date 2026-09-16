"""cycletime/dataio/fetch.py acquires the approved archive and verifies it before extraction."""
from __future__ import annotations

import shutil
import ssl
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from cycletime.contracts import EvidenceRef
from cycletime.dataio.archive import (
    approved_manifest,
    inspect_archive,
    require_extraction_space,
    sha256,
)
from cycletime.evidence import record


def download(url: str, destination: Path, approved_hosts: list[str]) -> None:
    """Stream one HTTPS archive from an approved host; refuse redirects that leave the allowlist."""

    class BoundedRedirect(urllib.request.HTTPRedirectHandler):
        """Follow redirects only while they stay on an approved HTTPS host."""

        def redirect_request(self, request, fp, code, msg, headers, newurl):
            target = urlparse(newurl)
            if target.scheme != "https" or target.hostname not in approved_hosts:
                raise ValueError(f"The download redirected to an unapproved destination: {newurl}")
            return super().redirect_request(request, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(
        BoundedRedirect, urllib.request.HTTPSHandler(context=ssl.create_default_context())
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with opener.open(url, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
    partial.replace(destination)


def fetch_dataset(manifest_path: Path, destination: Path) -> EvidenceRef:
    """Download an allowlisted HTTPS archive and verify its pre-recorded SHA256 before safe extraction; refuse null provenance, archive traversal, links, and self-approved hashes."""
    manifest_path = Path(manifest_path)
    destination = Path(destination)
    manifest = approved_manifest(manifest_path)
    project = manifest_path.resolve().parent.parent
    source = urlparse(manifest["digest_source"])
    archive_host = urlparse(manifest["archive_url"]).hostname
    if source.scheme != "https" or source.hostname == archive_host:
        # A hash published by the download host itself is not independent verification.
        raise ValueError("The approved digest requires an independent published source.")
    archive = project / str(manifest["archive_path"])
    if archive.exists() and sha256(archive) == manifest["archive_sha256"]:
        acquisition = "reused_verified_local_archive"
    else:
        if archive.exists():
            raise ValueError(
                "A local archive exists with a different digest; remove it before fetching again."
            )
        download(manifest["archive_url"], archive, manifest["approved_download_hosts"])
        # The freshly downloaded bytes prove nothing on their own; the recorded digest decides.
        if sha256(archive) != manifest["archive_sha256"]:
            archive.unlink()
            raise ValueError("The downloaded archive does not match the approved SHA256.")
        acquisition = "downloaded_and_verified_against_approved_digest"
    root = project / destination if not destination.is_absolute() else destination
    if root.exists() and any(root.iterdir()):
        raise ValueError("Extraction requires an empty destination; the dataset already exists.")
    require_extraction_space(archive, manifest["archive_sha256"], root.parent)
    staging = root.parent / f"{root.name}.staging"
    if staging.exists():
        raise ValueError("A previous extraction staging directory remains; remove it first.")
    staging.mkdir(parents=True)
    try:
        hashes = inspect_archive(archive, manifest["archive_sha256"], staging)
        extracted = next(staging.iterdir())
        if root.exists():
            raise ValueError("The destination appeared during extraction.")
        extracted.replace(root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(staging, ignore_errors=True)
    return record(
        project / "artifacts/fetch.json",
        {
            "dataset": manifest["dataset"],
            "archive_path": str(archive.relative_to(project)),
            "archive_sha256": manifest["archive_sha256"],
            "digest_source": manifest["digest_source"],
            "verified_by": manifest["verified_by"],
            "acquisition": acquisition,
            "destination": str(root.relative_to(project)),
            "member_count": len(hashes),
            "license": manifest["license"],
            "usage": "noncommercial research only",
            "self_approved_hash_refused": True,
        },
    )
