"""Nox sessions."""

from pathlib import Path
from typing import no_type_check

import nox


_package = "banquo"
_python_versions = ["3.11"]

nox.options.sessions = (
    "safety",
    "typeguard",
    "tests",
    "docs",
)


@no_type_check
@nox.session(python=_python_versions)
def safety(session: nox.Session) -> None:
    """Scan dependencies for insecure packages."""
    session.install(".[test]")
    session.run("safety", "scan", "--full-report")


@no_type_check
@nox.session(python=_python_versions)
def typeguard(session: nox.Session) -> None:
    """Runtime type checking using Typeguard."""
    session.install(".[test]")
    session.run("pytest", f"--typeguard-_packages={_package}", *session.posargs)


@no_type_check
@nox.session(python=_python_versions)
def tests(session: nox.Session) -> None:
    """Run the test and xdoctest suites."""
    session.install(".[test]")
    session.run(
        "pytest", "--cov=banquo", "--cov-report=xml", "--xdoctest", *session.posargs
    )


@no_type_check
@nox.session
def docs(session: nox.Session) -> None:
    """Build documentation."""
    with open(Path("docs/requirements.txt")) as f:
        requirements = f.read().splitlines()
    session.install(".[docs]")
    session.install(*requirements)
    session.chdir("docs")
    session.run("rm", "-rf", "_build/", external=True)
    build_dir = Path("docs", "_build", "html")

    if "serve" in session.posargs:
        sphinx_args = ["-W", ".", "--watch", ".", "--open-browser", str(build_dir)]
        session.run("sphinx-autobuild", *sphinx_args)
    else:
        sphinx_args = ["-W", ".", str(build_dir)]
        session.run("sphinx-build", *sphinx_args)
