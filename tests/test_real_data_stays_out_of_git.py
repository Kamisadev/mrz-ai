"""The real passports must never be committed.

`.gitignore` is a request, not a guarantee. `git add -f` overrides it, a future
edit to the ignore file can quietly stop matching, and `git add -A` from inside a
directory it does not cover picks things up — that last one has already happened
once in this repository's history, on this author's watch. This repository is
public. A passport pushed to it is on GitHub permanently and in other people's
clones within minutes, and no later commit takes it back.

So the ignore rule gets a test. It is worth a whole file for the same reason a
smoke alarm is worth a battery: the failure is silent, irreversible, and lands on
somebody who did not choose to take the risk.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "real"


def tracked(pathspec: str) -> list[str]:
    """The files git has under ``pathspec``, or an empty list if this is not a checkout."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", pathspec],
            cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git here
        pytest.skip("git is not available")
    if result.returncode != 0:  # pragma: no cover - not a checkout
        pytest.skip("not a git checkout")
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_no_real_passport_is_tracked_by_git() -> None:
    """The one that matters. Everything else in this file is a way of failing earlier."""
    found = tracked("real")
    assert not found, (
        "git is tracking files under real/ — these are real identity documents and "
        f"this repository is public: {found}. Do not commit. Remove them from the "
        "index with `git rm --cached`, and if any of this has been pushed, treat the "
        "documents as disclosed and tell whoever they belong to."
    )


def test_the_ignore_rule_that_protects_them_is_still_there() -> None:
    """Checked as text, so deleting the rule fails here and not on a passport.

    The rule is a whole directory with no negated pattern inside it. That is the
    design: `real/*` plus a `!real/README.md` exception is one typo away from
    matching a `.jpg`, and there is no version of that mistake worth the
    convenience of a tracked README.
    """
    rules = (ROOT / ".gitignore").read_text().splitlines()
    assert "/real/" in rules, ".gitignore no longer ignores real/"
    assert not [line for line in rules if line.startswith("!") and "real" in line], (
        "a negated pattern has appeared for real/ — the directory is ignored whole, "
        "on purpose"
    )


@pytest.mark.skipif(not REAL.exists(), reason="no real set on this machine")
def test_git_agrees_the_real_files_are_ignored() -> None:
    """Asks git itself, rather than trusting a reading of the rules.

    `check-ignore` answers the question that actually matters — would this file be
    committed — including every precedence rule this test's author might have got
    wrong. Runs only where the set exists, since there is nothing to ask about
    otherwise.
    """
    files = [path for path in REAL.rglob("*") if path.is_file()]
    if not files:
        pytest.skip("the real set is empty")

    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=ROOT, input="\n".join(str(path) for path in files),
        capture_output=True, text=True, timeout=30, check=False,
    )
    ignored = set(result.stdout.splitlines())
    exposed = [str(path) for path in files if str(path) not in ignored]
    assert not exposed, f"git would commit these real passports: {exposed}"
