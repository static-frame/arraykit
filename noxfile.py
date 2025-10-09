import nox
import sys

ARTIFACTS = (
    "*.egg-info",
    ".hypothesis",
    "build",
    "dist",
    "src/*.so",
)

# Make `nox` default to running tests if you just do `nox`
nox.options.sessions = ["test"]


def do_clean(session: nox.Session) -> None:
    # uninstall arraykit
    session.run(
        sys.executable, "-m", "pip",
        "--disable-pip-version-check", "uninstall", "--yes", "arraykit",
        external=True
    )
    # remove artifacts
    for artifact in sorted(ARTIFACTS):
        session.run("rm", "-rf", artifact, external=True)

def do_build(session: nox.Session) -> None:
    # keep -v to see warnings; no build isolation to match your invoke cmd
    session.run(
        sys.executable, "-m", "pip",
        "--disable-pip-version-check",
        "install", "-v", "--no-build-isolation", ".",
        external=True
    )

def do_test(session: nox.Session) -> None:
    session.run(
        "pytest",
        "-s",
        "--disable-pytest-warnings",
        "--tb=native",
        external=True,
    )

def do_performance(session: nox.Session) -> None:
    """Run performance benchmarks."""
    args = [sys.executable, "-m", "performance"]

    if session.posargs:
        args.extend(["--names"] + session.posargs)

    session.run(*args, external=True)

def do_lint(session: nox.Session) -> None:
    session.run(
        "pylint",
        "-f", "colorized",
        "*.py", "performance", "src", "test",
        external=True,
    )


# NOTE: use `nox -s build` to launch a session

@nox.session(python=False)  # use current environment
def clean(session):
    """Clean build artifacts and uninstall arraykit."""
    do_clean(session)

@nox.session(python=False)
def build(session):
    """Clean then build/install locally (like invoke: build depends on clean)."""
    do_clean(session)
    do_build(session)

@nox.session(python=False)
def test(session):
    """Build then run pytest (like invoke: test depends on build)."""
    do_clean(session)
    do_build(session)
    do_test(session)

@nox.session(python=False)
def performance(session):
    """Build then run performance benches (like invoke: performance depends on build)."""
    do_clean(session)
    do_build(session)
    do_performance(session)

@nox.session(python=False)
def lint(session):
    """Run pylint static analysis."""
    do_lint(session)
