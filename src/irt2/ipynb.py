import logging
import logging.config

import yaml

import irt2


# TODO remove this and use __init__.py:init_logging
def setup_logging():
    with (irt2.ENV.DIR.CONF / "logging.yaml").open(mode="r") as fd:
        conf = yaml.safe_load(fd)

        conf["handlers"]["stdout"]["formatter"] = "plain"
        conf["loggers"]["root"]["handlers"] = ["stdout"]

        logging.config.dictConfig(conf)

    log = logging.getLogger("irt2.ipynb")
    log.info("test")
