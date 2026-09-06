"""M0 smoke tests: the toolchain is wired up and the deliverables stay safe.

These are deliberately cheap. Their job is to fail loudly if the environment
drifts, not to test behaviour that does not exist yet.
"""

import subprocess
import sys
from pathlib import Path

import replay

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports() -> None:
    assert replay.__version__


def test_python_is_312_or_newer() -> None:
    assert sys.version_info >= (3, 12)


def test_playwright_is_installed() -> None:
    from playwright.sync_api import sync_playwright

    assert sync_playwright is not None


def test_cli_reports_version() -> None:
    """Exercise the installed console script, which is what a user actually runs."""
    result = subprocess.run(
        [str(REPO_ROOT / ".venv" / "bin" / "replay"), "version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == replay.__version__


def test_no_env_file_is_committed() -> None:
    """A real .env must never reach the repo. .env.example is fine."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    offenders = [f for f in tracked if Path(f).name.startswith(".env") and f != ".env.example"]
    assert not offenders, f"secret-bearing files tracked in git: {offenders}"


def test_chromium_launches() -> None:
    """The most fragile part of a fresh clone: the browser binary is actually there."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content("<h1>ok</h1>")
            assert page.inner_text("h1") == "ok"
        finally:
            browser.close()
