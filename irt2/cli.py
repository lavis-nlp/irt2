import logging
import os
import sys
from pathlib import Path
from typing import Literal

import click
import pretty_errors
import pudb
import rich_click as click

import irt2
from irt2 import dataset, evaluation
from irt2.loader import LOADER

log = logging.getLogger(__name__)
tee = irt2.tee(log)

os.environ["PYTHONBREAKPOINT"] = "pudb.set_trace"


pretty_errors.configure(
    filename_display=pretty_errors.FILENAME_EXTENDED,
    lines_after=2,
    line_number_first=True,
)


@click.group()
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    default=False,
    help="suppress console output",
)
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    default=False,
    help="activate debug mode (drop into pudb on error)",
)
def main(quiet: bool, debug: bool):
    """Use irt2m from the command line."""
    irt2.debug = debug
    irt2.init_logging()
    irt2.console.quiet = quiet

    tee(f"IRT2 ({irt2.version}) - Inductive Reasoning with Text")
    tee(f"initialized root path: {irt2.ENV.DIR.ROOT}")
    tee(f"executing from: {os.getcwd()}")


# --- load


def _load_dataset(folder: str | Path, loader: str) -> dataset.IRT2:
    assert loader in LOADER

    tee(f"loading {folder} using loader '{loader}'")
    ds = LOADER[loader](folder)

    tee(str(ds))
    return ds


@main.command("load")
@click.argument(
    "folder",
    nargs=1,
    type=click.Path(exists=True),
)
@click.option(
    "--loader",
    nargs=1,
    default=list(LOADER)[0],
    required=False,
    type=click.Choice(list(LOADER), case_sensitive=False),
    help="loader to use for foreign datasets",
)
@click.option(
    "--debug",
    is_flag=True,
    required=False,
    default=False,
    help="drop into debugger session",
)
@click.option(
    "--attach",
    is_flag=True,
    required=False,
    default=False,
    help="drop into ipython session",
)
@click.option(
    "--table",
    is_flag=True,
    required=False,
    default=False,
    help="print csv table data",
)
def main_corpus_load(
    folder: str,
    loader: str,
    debug: bool,
    attach: bool,
    table: bool,
):
    """Load a dataset for inspection."""
    ds = _load_dataset(folder, loader)

    if debug:
        breakpoint()

    if attach:
        print(f"\nlocal variable: 'ds': {ds}\n")
        from IPython import embed

        embed()

    if table:
        print(",".join(map(str, ds.table_row)), "")

    irt2.console.print("exiting.")


_shared_options = [
    click.option(
        "--head-task",
        type=str,
        required=True,
        help="all predictions from the head task",
    ),
    click.option(
        "--tail-task",
        type=str,
        required=True,
        help="all predictions from the tail task",
    ),
    click.option(
        "--irt2",
        type=str,
        required=True,
        help="path to irt2 data",
    ),
    click.option(
        "--loader",
        nargs=1,
        default=list(LOADER)[0],
        required=False,
        type=click.Choice(list(LOADER), case_sensitive=False),
        help="loader to use for foreign datasets",
    ),
    click.option(
        "--split",
        type=str,
        required=True,
        help="one of validation, test",
    ),
    click.option(
        "--max-rank",
        type=int,
        default=100,
        help="only consider the first n ranks (target filtered)",
    ),
    click.option(
        "--model",
        type=str,
        help="optional name of the model",
    ),
    click.option(
        "--out",
        type=str,
        help="optional output file for metrics",
    ),
]


# thanks https://stackoverflow.com/questions/40182157
def add_options(options):
    def _proxy(fn):
        [option(fn) for option in reversed(options)]
        return fn

    return _proxy


def _evaluate(
    task: Literal["kgc", "ranking"],
    head_task: str,
    tail_task: str,
    irt2: str,
    loader: str,
    split: str,
    max_rank: int,
    model: str | None,
    out: str | None,
):
    if out and Path(out).exists():
        print(f"skipping {out}")

    ds = _load_dataset(irt2, loader)
    assert split == "validation" or split == "test"

    metrics = evaluation.evaluate(
        ds,
        task=task,
        split=split,
        head_predictions=evaluation.load_csv(head_task),
        tail_predictions=evaluation.load_csv(tail_task),
        max_rank=max_rank,
    )

    evaluation.create_report(
        metrics,
        ds,
        task,
        split,
        model=model,
        filenames=dict(
            head_task=head_task,
            tail_task=tail_task,
        ),
        out=out,
    )


@main.command(name="evaluate-ranking")
@add_options(_shared_options)
def cli_eval_ranking(*args, **kwargs):
    """Evaluate the open-world ranking task."""
    _evaluate("ranking", *args, **kwargs)


@main.command(name="evaluate-kgc")
@add_options(_shared_options)
def cli_eval_kgc(*args, **kwargs):
    """
    Evaluate the open-world ranking task.

    It is possible to provide gzipped files: Just make
    sure the file suffix is *.gz.

    """
    _evaluate("kgc", *args, **kwargs)


def entry():
    try:
        main()

    except Exception as exc:
        if not irt2.debug:
            raise exc

        log.error("debug: catched exception, starting debugger")
        log.error(str(exc))

        _, _, tb = sys.exc_info()
        pudb.post_mortem(tb)

        raise exc
