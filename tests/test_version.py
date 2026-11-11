"""The reported package version must match pyproject.toml."""

import tomllib
from pathlib import Path

import lachesis
from lachesis.output import _lachesis_version


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    expected = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert lachesis.__version__ == expected
    assert _lachesis_version() == expected
