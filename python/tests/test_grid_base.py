"""Tests for the grid protocol."""

from lachesis.grid.base import IsochroneGrid


def test_protocol_exists():
    assert IsochroneGrid is not None
