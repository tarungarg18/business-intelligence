"""Guard against committing credentials.

`.env.example` is deliberately tracked as a template, which makes it the one
file in the repo where a real key can hide in plain sight. This happened once:
keys were pasted into the template and pushed before anyone noticed.

These tests fail loudly if it happens again.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Patterns for credentials that must never appear in tracked files.
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),          # OpenAI-style
    re.compile(r"\bAQ\.[A-Za-z0-9_-]{20,}"),       # Google ephemeral
    re.compile(r"\bAIza[A-Za-z0-9_-]{30,}"),       # Google API key
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),   # GitHub token
]

# Keys that are expected to appear as empty assignments in the template.
TEMPLATE_KEYS = ("GEMINI_API_KEY", "OPENCODE_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        pytest.skip("not a git repository")
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def test_env_file_is_not_tracked():
    """The real .env must never be under version control."""
    tracked = {p.name for p in _tracked_files()}
    assert ".env" not in tracked


def test_env_example_has_no_filled_keys():
    """Every key in the template must be empty."""
    example = ROOT / ".env.example"
    if not example.exists():
        pytest.skip(".env.example not present")

    for line in example.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() in TEMPLATE_KEYS:
            assert value.strip() == "", (
                f"{key.strip()} has a value in .env.example, which IS committed. "
                f"Move it to .env and rotate the exposed credential."
            )


def test_no_tracked_file_contains_a_credential():
    """Scan every tracked text file for credential-shaped strings."""
    offenders: list[str] = []
    for path in _tracked_files():
        if not path.exists() or path.suffix in {".pdf", ".pptx", ".png", ".jpg", ".npy"}:
            continue
        # This file necessarily contains the patterns themselves.
        if path.name == Path(__file__).name:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in SECRET_PATTERNS:
            match = pattern.search(text)
            if match:
                offenders.append(
                    f"{path.relative_to(ROOT)}: {match.group()[:12]}... "
                    f"matches {pattern.pattern}"
                )

    assert not offenders, "credential-shaped strings in tracked files:\n" + "\n".join(
        offenders
    )


def test_gitignore_covers_env():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    stripped = {line.strip() for line in ignored}
    assert ".env" in stripped, ".gitignore must exclude .env"
