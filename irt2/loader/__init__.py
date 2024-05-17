import logging
from pathlib import Path
from typing import Iterable

import irt2
import yaml
from irt2.dataset import IRT2
from ktz.filesystem import path

from .blp import load_fb15k237, load_umls, load_wikidata5m, load_wn18rr
from .irt import load_irt2

log = logging.getLogger("irt2.loader")
tee = irt2.tee(log)


LOADER = {
    "irt2": load_irt2,
    "blp/umls": load_umls,
    "blp/wn18rr": load_wn18rr,
    "blp/fb15k237": load_fb15k237,
    "blp/wikidata5m": load_wikidata5m,
}


def from_config(
    config: dict,
    only: Iterable[str] | None = None,
    without: Iterable[str] | None = None,
):
    datasets = {}

    for name, options in config["datasets"].items():
        if only is not None and name not in only:
            continue

        if without is not None and name in without:
            continue

        tee(f"reading configuration for {name}")

        loader = LOADER[options["loader"]]
        dataset: IRT2 = loader(
            options["path"],
            **options.get("kwargs", {}),
        )

        if "percentage" in options:
            dataset = dataset.tasks_subsample_kgc(
                percentage_val=options["percentage"]["validation"],
                percentage_test=options["percentage"]["test"],
            )

        datasets[name] = dataset
        tee(f"loaded {str(dataset)}")

    return datasets


def from_config_file(
    config_file: str | Path,
    only: Iterable[str] | None = None,
    without: Iterable[str] | None = None,
):
    with path(config_file, is_file=True).open(mode="r") as fd:
        config = yaml.safe_load(fd)

    return from_config(
        config,
        only=only,
        without=without,
    )
