# -*- coding: utf-8 -*-

"""IRT2 import-time things."""


from pathlib import Path
from ktz.filesystem import path as kpath


_root_path = kpath(__file__).parent.parent
assert _root_path.name == "irt2"


class _DIR:

    ROOT: Path = _root_path
    DATA: Path = kpath(_root_path / "data", create=True)


class ENV:
    """IRT2 environment."""

    DIR = _DIR


class IRT2Error(Exception):
    """General error."""

    pass
