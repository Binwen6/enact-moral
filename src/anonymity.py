from __future__ import annotations

import argparse
import re
from pathlib import Path

from config import PROJECT_ROOT


TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
IGNORED_DIRECTORIES = {".git", "__pycache__", "outputs", "build", "dist"}
IGNORED_CONTENT_FILES = {"src/anonymity.py"}


def forbidden_patterns(extra_denied: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    fragments = [
        "/" + "mnt" + "/",
        "/" + "data" + "/" + "USER" + "/",
        "/" + "home" + "/",
        "lbw" + "2026",
    ]
    patterns = [(f"forbidden path or identity fragment {value!r}", re.compile(re.escape(value), re.IGNORECASE)) for value in fragments]
    patterns.extend([
        ("email address", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
        ("secret assignment", re.compile(r"\b(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|SECRET_KEY)\s*[=:]\s*['\"]?[A-Za-z0-9_-]{12,}", re.IGNORECASE)),
        ("private key", re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY")),
    ])
    patterns.extend(
        (f"denied value {value!r}", re.compile(re.escape(value), re.IGNORECASE))
        for value in extra_denied
    )
    return patterns


def candidate_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if relative.as_posix() in IGNORED_CONTENT_FILES:
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", "NOTICE", "DATA_LICENSE"}:
            files.append(path)
    return sorted(files)


def scan_repository(root: Path, extra_denied: list[str]) -> list[str]:
    violations: list[str] = []
    for path in candidate_files(root):
        relative = path.relative_to(root)
        text = path.read_text(encoding="utf-8")
        for description, pattern in forbidden_patterns(extra_denied):
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                violations.append(f"{relative}:{line}: {description}")
        if path.name == ".env":
            violations.append(f"{relative}: environment secret file")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Check release files for identity paths, emails, and secrets.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--deny", action="append", default=[])
    arguments = parser.parse_args()
    violations = scan_repository(arguments.root.resolve(), arguments.deny)
    if violations:
        raise RuntimeError("Anonymous-release check failed:\n" + "\n".join(violations))
    print({"status": "ok", "files": len(candidate_files(arguments.root.resolve()))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
