# Gate 0 verifies the local dataset.

The implementation checks the archive against the independently published SHA256 in data/manifest.json.
The verifier checks available disk space before it extracts files into a temporary directory.
The verifier rejects unsafe archive paths, links, duplicate members, and special files.
The verifier decodes each PNG and checks each extracted file against its archive member hash.
The verifier accepts the archive's readme.txt and license.txt files within each category and verifies their hashes.
The verifier requires all fifteen categories and normal training images only.
The verifier checks defect masks for coverage and matching dimensions.
The verifier rejects content overlap between training and test images.
The verifier publishes data/raw/mvtec_ad only after these checks succeed.
The verifier records per-file hashes in data/manifest.json and writes artifacts/gate0.json after success.

The operator runs `make gate0` from the repository root.
The command rechecks existing extracted data against the approved archive on subsequent runs.
The local archive must remain at data/raw/mvtec_anomaly_detection.tar.xz.
The extraction requires 5,271,247,124 bytes plus a 512 MiB reserve.
Gate 0 passed after the operator freed disk space and the verifier accepted the archive's category metadata.
The successful run verified 6,644 files, including 5,354 images and 1,258 masks.
The evidence in artifacts/gate0.json records the archive digest, dataset digest, checks, and timestamp.

The focused checks run with `uv run --frozen pytest tests/unit/test_gate0.py`.
All seventeen focused tests pass.
The full suite now reports forty-two passing tests and seventeen failures.
The Gate 3 hardware test reports the missing ANE placement evidence as a real gate failure.
The repository has a generated dependency lock and a Python 3.12 environment.
The directory has no Git repository, so the lock has not been committed.
