import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import click
import pretty_errors
import pudb
import rich_click as click

import irt2
from irt2 import dataset, evaluation

log = logging.getLogger(__name__)
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

    irt2.console.print(f"IRT2 ({irt2.version}) - Inductive Reasoning with Text")
    irt2.console.print(f"initialized root path: {irt2.ENV.DIR.ROOT}")
    irt2.console.print(f"executing from: {os.getcwd()}")


@main.command("load")
@click.argument(
    "folder",
    nargs=1,
    type=click.Path(exists=True),
)
@click.option(
    "--loader",
    nargs=1,
    default=list(dataset.LOADER)[0],
    required=False,
    type=click.Choice(
        list(dataset.LOADER),
        case_sensitive=False,
    ),
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
def main_corpus_load(
    folder: str,
    loader: str,
    debug: bool,
    attach: bool,
):
    """Load a dataset for inspection."""

    ds = dataset.load(folder, loader)
    irt2.console.print(str(ds))

    if debug:
        breakpoint()

    if attach:
        print(f"\nlocal variable: 'ds': {ds}\n")
        from IPython import embed

        embed()

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


@main.command(name="evaluate-ranking")
@add_options(_shared_options)
def cli_eval_ranking(
    head_task: str,
    tail_task: str,
    irt2: str,
    split: str,
    max_rank: int,
    model: str | None,
    out: str | None,
):
    """Evaluate the open-world ranking task."""
    if out and Path(out).exists():
        print(f"skipping {out}")

    assert split == "validation" or split == "test"

    dataset, gt_head, gt_tail = evaluation.load_gt(
        irt2,
        task="ranking",
        split=split,
    )

    metrics = evaluation.compute_metrics_from_csv(
        (head_task, gt_head),
        (tail_task, gt_tail),
        max_rank,
    )

    report = dict(
        date=datetime.now().isoformat(),
        dataset=dataset.name,
        model=model or "unknown",
        task="ranking",
        split=split,
        filename_head=head_task,
        filename_tail=tail_task,
        metrics=metrics,
    )

    evaluation.write_report(report, out)


@main.command(name="evaluate-kgc")
@add_options(_shared_options)
def cli_eval_kgc(
    irt2: str,
    head_task: str,
    tail_task: str,
    split: Literal["validation", "test"],
    max_rank: int,
    model: str | None,
    out: str | None,
):
    """
    Evaluate the open-world ranking task.

    It is possible to provide gzipped files: Just make
    sure the file suffix is *.gz.

    """
    if out and Path(out).exists():
        print(f"skipping {out}")

    dataset, gt_head, gt_tail = evaluation.load_gt(
        irt2,
        task="kgc",
        split=split,
    )

    metrics = evaluation.compute_metrics_from_csv(
        (head_task, gt_head),
        (tail_task, gt_tail),
        max_rank,
    )

    report = dict(
        date=datetime.now().isoformat(),
        dataset=dataset.name,
        model=model or "unknown",
        task="kgc",
        split=split,
        filename_head=head_task,
        filename_tail=tail_task,
        metrics=metrics,
    )

    evaluation.write_report(report, out)


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
