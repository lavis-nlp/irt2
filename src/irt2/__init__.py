"""IRT2 import-time things."""

import logging
import logging.config
import os
from pathlib import Path

import yaml
from ktz.filesystem import path as path
from rich.console import Console

_root_path = Path(__file__).absolute().parents[2]


debug = False
version = "1.0.1"


# check whether data directory is overwritten
ENV_DIR_DATA = "IRT2_DATA"
if ENV_DIR_DATA in os.environ:
    _data_path = path(os.environ[ENV_DIR_DATA])
else:
    _data_path = _root_path / "data"


class _DIR:
    ROOT: Path = _root_path
    DATA: Path = path(_data_path, create=True)
    CONF: Path = path(_root_path / "conf", create=True)
    CACHE: Path = path(_data_path / "cache", create=True)


class ENV:
    """IRT2 environment."""

    DIR = _DIR


class IRT2Error(Exception):
    """General error."""

    pass


ENV_LOG_CONF = "IRT2_LOG_CONF"
ENV_LOG_FILE = "IRT2_LOG_FILE"


log = logging.getLogger(__name__)

# if used as library do not log anything
log.addHandler(logging.NullHandler())


def init_logging():
    """Read the logging configuration from conf/ and initialize."""
    global log

    def _env(key, default):
        if key in os.environ:
            return os.environ[key]
        return default

    # expeting and removing the NullHandler
    assert len(log.handlers) == 1, "log misconfiguration"
    log.removeHandler(log.handlers[0])

    conf_file = _env(ENV_LOG_CONF, ENV.DIR.CONF / "logging.yaml")
    if not Path(conf_file).exists():
        return

    with path(conf_file, is_file=True).open(mode="r") as fd:
        conf = yaml.safe_load(fd)

    logfile = conf["handlers"]["logfile"]
    logfile["filename"] = _env(
        ENV_LOG_FILE,
        logfile["filename"].format(ENV=ENV),
    )

    if not logfile["filename"].startswith("/"):
        logfile["filename"] = str(ENV.DIR.ROOT / logfile["filename"])

    logging.config.dictConfig(conf)
    logging.captureWarnings(True)

    log.info("logging initialized")


# rich console is quiet by default
console = Console(quiet=True)


def tee(log_instance):
    def _tee(*messages: str, level=logging.INFO):
        for message in messages:
            log_instance.log(level, message)
        console.log(*messages, _stack_offset=2)

    return _tee
