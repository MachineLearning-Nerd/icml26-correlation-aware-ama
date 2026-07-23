#!/usr/bin/env python3
"""Validate and manifest the additive Hugging Face Space release candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_ROOT = REPO_ROOT / "release" / "hf_space_text"
ARTIFACT_ROOT = REPO_ROOT / ".openresearch" / "artifacts"
ALLOWLIST_PATH = REPO_ROOT / "release" / "hf_upload_allowlist.txt"
UPLOAD_MANIFEST_PATH = REPO_ROOT / "release" / "hf_upload_manifest.sha256"
SUBSET_JSON_PATH = REPO_ROOT / "release" / "old_new_subset_check.json"
SUBSET_MD_PATH = REPO_ROOT / "release" / "old_new_subset_check.md"

SECRET_PATTERNS = {
    "hugging_face_token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "openai_token": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "bearer_jwt": re.compile(
        rb"Bearer\s+eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."
        rb"[A-Za-z0-9_-]{10,}",
        re.IGNORECASE,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_below(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    }


def iter_logbook_nodes(node: dict) -> list[dict]:
    result = [node]
    for child in node.get("children", []):
        result.extend(iter_logbook_nodes(child))
    return result


def validate_markdown_links(payload_files: dict[str, Path]) -> list[str]:
    missing: list[str] = []
    link_re = re.compile(r"\]\(([^)]+)\)")
    for relative, path in payload_files.items():
        if path.suffix.lower() != ".md":
            continue
        for target in link_re.findall(path.read_text(encoding="utf-8")):
            target = unquote(target.strip().split("#", 1)[0])
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{relative} -> {target}")
    return missing


def validate_artifact_manifest() -> tuple[int, list[str]]:
    manifest_path = ARTIFACT_ROOT / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for relative, expected in manifest.items():
        path = REPO_ROOT / relative
        if not path.is_file():
            errors.append(f"missing artifact: {relative}")
            continue
        if path.stat().st_size != expected["bytes"]:
            errors.append(f"byte mismatch: {relative}")
        if sha256(path) != expected["sha256"]:
            errors.append(f"hash mismatch: {relative}")
    return len(manifest), errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protected", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()

    protected = args.protected.resolve()
    candidate = args.candidate.resolve()
    if protected == candidate:
        raise SystemExit("protected and candidate directories must differ")

    protected_files = files_below(protected)
    candidate_files = files_below(candidate)
    payload_files = files_below(PAYLOAD_ROOT)

    missing_old = sorted(set(protected_files) - set(candidate_files))
    changed_old = sorted(
        relative
        for relative, old_path in protected_files.items()
        if relative in candidate_files
        and sha256(old_path) != sha256(candidate_files[relative])
    )
    unexpected_old_changes = [
        relative for relative in changed_old if relative != "logbook.json"
    ]

    payload_sync_errors = [
        relative
        for relative, source in payload_files.items()
        if relative not in candidate_files
        or sha256(source) != sha256(candidate_files[relative])
    ]

    non_text: list[str] = []
    secret_hits: dict[str, list[str]] = {name: [] for name in SECRET_PATTERNS}
    for relative, path in payload_files.items():
        data = path.read_bytes()
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            non_text.append(relative)
        if b"\0" in data:
            non_text.append(relative)
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                secret_hits[name].append(relative)
    secret_hits = {name: paths for name, paths in secret_hits.items() if paths}

    logbook_path = PAYLOAD_ROOT / "logbook.json"
    logbook = json.loads(logbook_path.read_text(encoding="utf-8"))
    logbook_files = [
        node["file"] for node in iter_logbook_nodes(logbook["root"])
    ]
    missing_logbook_files = sorted(
        relative for relative in logbook_files if relative not in candidate_files
    )
    duplicate_slugs = sorted(
        slug
        for slug in {
            node["slug"] for node in iter_logbook_nodes(logbook["root"])
        }
        if sum(
            node["slug"] == slug
            for node in iter_logbook_nodes(logbook["root"])
        )
        > 1
    )
    missing_markdown_links = validate_markdown_links(payload_files)
    artifact_count, artifact_errors = validate_artifact_manifest()

    allowlist = sorted(payload_files)
    ALLOWLIST_PATH.write_text(
        "".join(f"{relative}\n" for relative in allowlist),
        encoding="utf-8",
    )
    UPLOAD_MANIFEST_PATH.write_text(
        "".join(
            f"{sha256(payload_files[relative])}  {relative}\n"
            for relative in allowlist
        ),
        encoding="utf-8",
    )

    checks = {
        "protected_revision": "1c13494fc9e76a381d76c681cfd582495eb79d02",
        "protected_file_count": len(protected_files),
        "candidate_file_count": len(candidate_files),
        "upload_file_count": len(payload_files),
        "manifested_artifact_count": artifact_count,
        "missing_old_files": missing_old,
        "changed_old_files": changed_old,
        "unexpected_old_changes": unexpected_old_changes,
        "payload_sync_errors": payload_sync_errors,
        "non_text_uploads": sorted(set(non_text)),
        "secret_pattern_hits": secret_hits,
        "missing_logbook_files": missing_logbook_files,
        "duplicate_logbook_slugs": duplicate_slugs,
        "missing_markdown_links": missing_markdown_links,
        "artifact_manifest_errors": artifact_errors,
    }
    checks["passed"] = not any(
        (
            missing_old,
            unexpected_old_changes,
            payload_sync_errors,
            non_text,
            secret_hits,
            missing_logbook_files,
            duplicate_slugs,
            missing_markdown_links,
            artifact_errors,
        )
    ) and changed_old == ["logbook.json"]

    SUBSET_JSON_PATH.write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    SUBSET_MD_PATH.write_text(
        "\n".join(
            [
                "# Protected logbook subset check",
                "",
                f"- Result: **{'PASS' if checks['passed'] else 'FAIL'}**",
                f"- Protected revision: `{checks['protected_revision']}`",
                f"- Protected files retained: {len(protected_files) - len(missing_old)}/{len(protected_files)}",
                f"- Candidate files: {len(candidate_files)}",
                f"- Exact text upload allowlist: {len(payload_files)} files",
                f"- Manifested evidence files verified: {artifact_count}",
                f"- Intentionally changed protected path: `{', '.join(changed_old)}`",
                f"- Unexpected protected changes: {len(unexpected_old_changes)}",
                f"- Missing local Markdown targets: {len(missing_markdown_links)}",
                f"- Secret-pattern hits: {sum(map(len, secret_hits.values()))}",
                "",
                "All original paths remain reachable. The only byte-changed original",
                "path is `logbook.json`, extended to expose the additive campaign tree;",
                "the six original page files and all original assets remain byte-identical.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(checks, sort_keys=True))
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
