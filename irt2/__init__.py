# -*- coding: utf-8 -*-

"""IRT2 import-time things."""


import os
from pathlib import Path
from ktz.filesystem import path as kpath


_root_path = kpath(__file__).parent.parent


# check whether data directory is overwritten
ENV_DIR_DATA = "IRT2_DATA"
if ENV_DIR_DATA in os.environ:
    _data_path = kpath(os.environ[ENV_DIR_DATA])
else:
    _data_path = kpath(_root_path / "data", create=True)


class _DIR:

    ROOT: Path = _root_path
    DATA: Path = _data_path
    CONF: Path = kpath(_root_path / "conf", create=True)
    CACHE: Path = kpath(_data_path / "cache", create=True)


class ENV:
    """IRT2 environment."""

    DIR = _DIR


class IRT2Error(Exception):
    """General error."""

    pass
