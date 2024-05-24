import random
from functools import partial
from typing import Callable, Literal

import irt2
from irt2 import evaluation
from irt2.dataset import IRT2
from irt2.evaluation import Predictions
from irt2.types import MID, RID, VID, Split
from ktz.collections import dflat

Tasks = dict[tuple[MID, RID], set[VID]]


def get_vid2mids(ds: IRT2, split: Literal["validation", "test"]):
    if ds.name.startswith("IRT"):
        splits = (Split.train,)
    elif ds.name.startswith("BLP"):
        splits = (Split.train, Split.valid)
        if split == "test":
            splits += (Split.test,)
    else:
        assert False, f"unknown {ds.name}"

    vid2mids = {}
    for target_split in splits:
        # an overwrite here is okay as IRT only has a single split
        # for lookup and BLP vid2mids sets are disjoint
        vid2mids |= ds.idmap.vid2mids[target_split]

    return vid2mids, splits


# --- MODELS


def true_vids(tasks: Tasks, **_) -> Predictions:
    """This model cheats and always answers always correctly."""
    for (mid, rid), vids in tasks.items():
        yield (mid, rid), ((vid, 1) for vid in vids)


def true_mentions(
    tasks: Tasks,
    ds: IRT2,
    split: Literal["validation", "test"],
    **_,
) -> Predictions:
    """This model cheats and knows the correct mentions."""

    vid2mids, splits = get_vid2mids(ds, split)
    for (mid, rid), gt_vids in tasks.items():
        mentions = [ds.idmap.mid2str[mid] for vid in gt_vids for mid in vid2mids[vid]]
        pr_vids = ds.find_by_mention(
            *mentions,
            splits=splits,
        )

        yield (mid, rid), ((vid, 1) for vid in pr_vids)


def random_guessing(
    tasks: Tasks,
    ds: IRT2,
    split: Literal["validation", "test"],
    seed: int,
    **_,
) -> Predictions:
    """This model is just guessing randomly."""
    rng = random.Random()
    rng.seed(seed)

    vid2mids, _ = get_vid2mids(ds, split)
    candidates = set(vid2mids)

    perm = list(candidates)
    for (mid, rid), vids in tasks.items():
        yield (mid, rid), ((vid, rng.random()) for vid in rng.sample(perm, k=100))


MODELS = {
    "true-vertices": true_vids,
    "true-mentions": true_mentions,
    "random-guessing": random_guessing,
}


# --- RUNNER


def flatten(report: dict):
    before = dict(
        dataset=report["dataset"],
        model=report["model"],
        date=report["date"],
        split=report["split"],
    )

    metrics = dflat(report["metrics"], sep=" ")
    metrics = dict(sorted(metrics.items()))

    return before | metrics


def evaluate(
    ds: IRT2,
    name: str,
    split: Literal["validation", "test"],
    head_predictions: Predictions,
    tail_predictions: Predictions,
):
    metrics = evaluation.evaluate(
        ds=ds,
        task="kgc",
        split=split,
        head_predictions=head_predictions,
        tail_predictions=tail_predictions,
    )

    return evaluation.create_report(
        metrics,
        ds,
        task="kgc",
        split=split,
        model=name,
        filenames=dict(notebook="ipynb/control-experiments.ipynb"),
    )


def run(
    ds: IRT2,
    name: str,
    model: Callable,
    split: str,
    seed: int,
):
    predictor = partial(
        model,
        ds=ds,
        split=split,
        seed=seed,
    )

    assert split == "validation" or split == "test"

    if split == "validation":
        head_predictions = predictor(ds.open_kgc_val_heads)
        tail_predictions = predictor(ds.open_kgc_val_tails)

    elif split == "test":
        head_predictions = predictor(ds.open_kgc_test_heads)
        tail_predictions = predictor(ds.open_kgc_test_tails)

    report = evaluate(
        ds=ds,
        name=name,
        split=split,
        head_predictions=head_predictions,
        tail_predictions=tail_predictions,
    )

    return report


import csv
from pathlib import Path

from irt2.loader import from_config_file
from ktz.collections import dconv, dflat


def _run_all(datasets_config, models, splits, seed: int):
    datasets = from_config_file(
        root_path=irt2.ENV.DIR.ROOT,
        **datasets_config,
    )

    for _, dataset in datasets:
        print("\n", str(dataset))

        for split in splits:
            assert split == "validation" or split == "test"

            if split == "validation":
                n_heads = len(dataset.open_kgc_val_heads)
                n_tails = len(dataset.open_kgc_val_tails)

            elif split == "test":
                n_heads = len(dataset.open_kgc_test_heads)
                n_tails = len(dataset.open_kgc_test_tails)

            else:
                assert False

            options = dataset.meta["loader"]
            percentage = None
            if "subsample" in options:
                percentage = options["subsample"].get(split, None)

            print(
                "  " + split,
                f"percentage={percentage}",
                f"{n_heads} head and {n_tails} tail tasks" f" = {n_heads + n_tails}",
                sep="\n    - ",
            )

            meta = {
                "percentage": percentage,
                "total tasks": n_heads + n_tails,
                "head tasks": n_heads,
                "tail tasks": n_tails,
            }

            # print(', '.join(map(str, dataset.table_row)))
            for model in models:
                print("    - model: ", model)
                report = run(dataset, model, MODELS[model], split, seed)

                h10 = report["metrics"]["all"]["micro"]["hits_at_10"]  # type: ignore
                print(f"    - result: {h10:2.3f}")

                yield meta | flatten(report)


def run_all(out, datasets_config, models, splits, seed: int):
    out.parent.mkdir(exist_ok=True, parents=True)

    print(f"write results to {out}")
    with out.open(mode="w") as fd:
        writer = None

        for flat in _run_all(datasets_config, models, splits, seed):
            if writer is None:
                header = ["seed"] + list(flat.keys())

                writer = csv.DictWriter(fd, fieldnames=header)
                writer.writeheader()

            writer.writerow(flat | {"seed": seed})


all_config = {
    "datasets_config": {
        # 'config_file': irt2.ENV.DIR.CONF / 'datasets' / 'original.yaml',
        "without": ["blp/*"],
        "config_file": irt2.ENV.DIR.CONF / "datasets" / "original-subsampled.yaml",
        # 'config_file': irt2.ENV.DIR.CONF / 'datasets' / 'full.yaml',
        # 'config_file': irt2.ENV.DIR.CONF / 'datasets' / 'full-subsampled.yaml',
        # 'without': ('blp/wikidata5m', )
    },
    "models": ["true-mentions"],
    "splits": [
        "validation",
        "test",
    ],
    "seed": 31189,
}


def main(config):
    name = config["datasets_config"]["config_file"].stem
    fcsv = f"control-experiments-{name}.csv"
    run_all(out=irt2.ENV.DIR.DATA / "evaluation" / fcsv, **config)


if __name__ == "__main__":
    main(all_config)
    print("done")
