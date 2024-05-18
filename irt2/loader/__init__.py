import logging
from pathlib import Path
from typing import Generator, Iterable

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
    root_path: str | Path | None = None,
) -> Generator[tuple[str, IRT2], None, None]:
    # ) -> Generator[tuple[str, IRT2], None None]:
    assert (set(only or []) | set(without or [])).issubset(config["datasets"])
    root = Path(".") if root_path is None else path(root_path, is_dir=True)

    datasets = {}
    for name, options in config["datasets"].items():
        # --- filter datasets

        if only is not None and name not in only:
            continue

        if without is not None and name in without:
            continue

        # --- load

        tee(f"reading configuration for {name}")

        loader = LOADER[options["loader"]]
        dataset: IRT2 = loader(
            root / options["path"],
            **options.get("kwargs", {}),
        )

        dataset.meta["loader"] = options

        # --- filter gt

        if "subsample" in options:
            sub_options = options["subsample"]

            if "seed" in sub_options:
                seed = sub_options["seed"]
            else:
                seed = config["seed"]

            dataset = dataset.tasks_subsample_kgc(
                percentage_val=sub_options["validation"],
                percentage_test=sub_options["test"],
                seed=seed,
            )

        # --- return

        tee(f"loaded {str(dataset)}")
        yield name, dataset


def from_config_file(
    config_file: str | Path,
    **kwargs,
) -> Generator[tuple[str, IRT2], None, None]:
    with path(config_file, is_file=True).open(mode="r") as fd:
        config = yaml.safe_load(fd)

    yield from from_config(config, **kwargs)


def from_kwargs(
    dataset_path: str | Path,
    loader: str,
    name: str = "default",
    **kwargs,
):
    config = dict(
        datasets={
            name: dict(
                path=dataset_path,
                loader=loader,
                **kwargs,
            )
        }
    )

    yield from from_config(config)
