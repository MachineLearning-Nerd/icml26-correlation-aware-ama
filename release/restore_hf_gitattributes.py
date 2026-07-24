#!/usr/bin/env python3
"""Restore the judged Space .gitattributes after automatic LFS additions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


EXPECTED_SHA256 = (
    "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--protected-file", required=True, type=Path)
    args = parser.parse_args()

    data = args.protected_file.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise SystemExit("protected .gitattributes hash mismatch")

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
        operations=[
            CommitOperationAdd(
                path_in_repo=".gitattributes",
                path_or_fileobj=args.protected_file,
            )
        ],
        commit_message="Restore protected Space attributes",
        commit_description=(
            "Restore .gitattributes byte-for-byte to judged revision "
            "1c13494fc9e76a381d76c681cfd582495eb79d02 after the Hub "
            "automatically appended two LFS rules during evidence upload."
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
                "restored_path": ".gitattributes",
                "sha256": EXPECTED_SHA256,
                "commit_url": result.commit_url,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
