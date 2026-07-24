#!/usr/bin/env python3
"""Commit the validated text allowlist to the existing CA-AMA Space."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


REPO_ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_ROOT = REPO_ROOT / "release" / "hf_space_text"
ALLOWLIST_PATH = REPO_ROOT / "release" / "hf_upload_allowlist.txt"
MANIFEST_PATH = REPO_ROOT / "release" / "hf_upload_manifest.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()

    allowlist = [
        line.strip()
        for line in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        expected_hash, relative = line.split("  ", 1)
        manifest[relative] = expected_hash
    if set(allowlist) != set(manifest):
        raise SystemExit("allowlist and SHA-256 manifest paths differ")

    operations = []
    for relative in allowlist:
        path = PAYLOAD_ROOT / relative
        if not path.is_file() or sha256(path) != manifest[relative]:
            raise SystemExit(f"payload hash mismatch: {relative}")
        data = path.read_bytes()
        data.decode("utf-8")
        if b"\0" in data:
            raise SystemExit(f"NUL byte in text payload: {relative}")
        operations.append(
            CommitOperationAdd(path_in_repo=relative, path_or_fileobj=path)
        )

    api = HfApi()
    before = api.repo_info(repo_id=args.repo_id, repo_type="space")
    if before.sha != args.expected_head:
        raise SystemExit(
            f"Space head changed: expected {args.expected_head}, found {before.sha}"
        )

    result = api.create_commit(
        repo_id=args.repo_id,
        repo_type="space",
        revision="main",
        parent_commit=args.expected_head,
        operations=operations,
        commit_message="Publish canonical five-claim CA-AMA logbook",
        commit_description=(
            "Canonical additive CPU reproduction with literal theorem-scope "
            "checks, general DSIC verification, an exact full-scale Claim 4 "
            "seed (honestly BLOCKED pending five seeds), a four-route Claim 5 "
            "audit, independent checks, negative controls, and manifests. "
            "Preserves the judged and previously published evidence trees."
        ),
    )
    after = api.repo_info(repo_id=args.repo_id, repo_type="space")
    if after.sha != result.oid:
        raise SystemExit(
            f"post-commit head mismatch: result {result.oid}, head {after.sha}"
        )
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "previous_revision": args.expected_head,
                "new_revision": after.sha,
                "uploaded_text_paths": len(operations),
                "commit_url": result.commit_url,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
