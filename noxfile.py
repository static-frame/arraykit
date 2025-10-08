# noxfile.py
import nox
import sys
import os

ARTIFACTS = (
    "*.egg-info",
    ".hypothesis",
    "build",
    "dist",
    "src/*.so",
)

# Optional: set defaults so `nox` with no args runs tests.
# Comment this out if you don't want defaults.
nox.options.sessions = ["test"]

@nox.session(python=False)  # run in the current environment
def clean(session):
    """Clean build artifacts and uninstall arraykit."""
    session.run(
        sys.executable, "-m", "pip",
        "--disable-pip-version-check", "uninstall", "--yes", "arraykit",
        external=True
    )
    for artifact in sorted(ARTIFACTS):
        session.run("rm", "-rf", artifact, external=True)

@nox.session(python=False)
def build(session):
    """Build/install locally without build isolation (keeps verbose for warnings)."""
    session.run(
        sys.executable, "-m", "pip",
        "--disable-pip-version-check", "-v",
        "install", "--no-build-isolation", ".",
        external=True
    )

@nox.session(python=False)
def test(session):
    """Run pytest with native traceback."""
    session.run(
        "pytest", "-s", "--disable-pytest-warnings", "--tb=native",
        external=True
    )

@nox.session(python=False)
def performance(session):
    """Run performance benches. Pass names via env: NAMES='foo,bar' nox -s performance"""
    names = os.environ.get("NAMES", "")
    args = [sys.executable, "-m", "performance"]
    if names:
        args.extend(["--names", names])
    session.run(*args, external=True)

@nox.session(python=False)
def lint(session):
    """Run pylint static analysis."""
    session.run(
        "pylint", "-f", "colorized", "*.py", "performance", "src", "test",
        external=True
    )
