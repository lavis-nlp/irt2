# -*- coding: utf-8 -*-
"""
IRT2 Evaluation.

We offer two evaluation scripts for reproducibility
and fair comparison with future models. You can use
the CLI (file based) and internal API for both the
open-world knowledge graph completion and the ranking
tasks.
"""

import csv
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from statistics import mean
from typing import Collection, Optional, Union

import click
import pretty_errors
import yaml
from ktz.filesystem import path as kpath

import irt2
from irt2.dataset import IRT2
from irt2.types import MID, RID, VID

#
# rank metrics
#

#
# The macro-averaged score (or macro score) is computed using the
# arithmetic mean (aka unweighted mean) of all the per-class scores.

# Micro averaging computes a global average score. Micro-averaging essentially
# computes the proportion of correctly classified observations
# out of all observations.
#
# A "class" in our setting is a task from the ranking or kgc tasks.
#
# see
# noqa https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
# noqa https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f


def rr(ranks: list[int]) -> list[float]:
    """Calculate the mrr for each rank in tfs."""
    assert all(rank >= 0 for rank in ranks)
    return [1 / rank if rank != 0 else 0 for rank in ranks]


def micro_mrr(rankcol: Collection[list[int]]) -> float:
    """Compute the MICRO Mean Reciprocal Rank (MRR)."""
    return mean(inv for ranks in rankcol for inv in rr(ranks))


def macro_mrr(rankcol: Collection[list[int]]) -> float:
    """Compute the MACRO Mean Reciprocal Rank (MRR)."""
    return mean(mean(rr(ranks)) for ranks in rankcol)


def hits_at_k(ranks: list[int], k: int) -> float:
    return mean(1 if rank > 0 and rank <= k else 0 for rank in ranks)


def micro_hits_at_k(rankcol: Collection[list[int]], k: int) -> float:
    """Compute hits@k for all tasks."""
    assert k > 0
    flat = (rank for ranks in rankcol for rank in ranks)
    return hits_at_k(flat, k=k)


def macro_hits_at_k(rankcol: Collection[list[int]], k: int) -> float:
    assert k > 0
    return mean(hits_at_k(ranks, k=k) for ranks in rankcol)


# kgc: MID, RID query and (closed-world) VID targets
# ranking: VID, RID query and (open-world) MID targets

Ent = Union[MID, VID]
Task = tuple[Ent, RID]
Scores = set[(Ent, float)]
Prediction = dict[Task, Scores]
GroundTruth = dict[Task, set[Ent]]
TaskTriple = Union[tuple[VID, RID, MID], tuple[MID, RID, VID]]


@dataclass(frozen=True, order=True)
class Rank:

    # order by target-filtered rank with
    # the score as tie-breaker

    filtered: int
    score: float
    value: int


class Ranks(dict):  # TaskTriple -> Rank
    """
    Input format for evaluation metrics.

    Reads datapoints either from csv or dictionary.
    Ranks are triples:
      - (vid, rid, mid) for ranking
      - (mid, rid, vid) for open-world kgc
    mapping to Rank objects.

    Provided predictions are sorted by their score.

    """

    @cached_property
    def tasks(self) -> set[(Union[VID, MID], RID)]:
        """Obtain all tasks from the datapoints."""
        return {(ent, rid) for (ent, rid, _) in self.keys()}

    def __init__(self, pred: Prediction, gt: GroundTruth):
        assert set(gt.keys()) == set(pred.keys()), "ground-truth/predictions mismatch!"

        for (ent, rid), pred_ents in pred.items():

            seen = 0
            gen = sorted(pred_ents, key=lambda t: t[1], reverse=True)

            for rank, (pred_ent, score) in enumerate(gen, start=1):
                if pred_ent in gt[(ent, rid)]:

                    self[(ent, rid, pred_ent)] = Rank(
                        value=rank,
                        filtered=rank - seen,
                        score=score,
                    )

                    seen += 1

    @classmethod
    def from_dict(Cls: "Ranks", *args, **kwargs):
        return Cls(*args, **kwargs)

    @classmethod
    def from_csv(Cls: "Ranks", path: str, gt: GroundTruth):
        """
        Load the evaluation data from csv file.

        CSV file format:

        For Ranking:
        [
            [ vid, rid, pred_mid1, score_for_mid1, pred_mid2, score_for_mid2, ... ],
            ...
        ]

        For kgc:
        [
            [ mid, rid, pred_vid1, score_for_vid1, pred_vid2, score_for_vid2, ... ],
            ...
        ]


        The order of the predictions do not matter as they are sorted by score
        before ranks are calculated.

        Parameters
        ----------
        path : str
            Where to load the csv file from.

        gt : RankTask
            The ground truth entities for each task.

        """
        preds = {}

        with kpath(path, is_file=True).open(mode="r") as fd:
            reader = csv.reader(fd, delimiter=",")

            for row in reader:
                ent, rid = map(int, row[:2])

                # convert [1, 2.3, 2, 4.3] -> [(2, 4.3), (1, 2.3)]
                scores = zip(map(int, row[2::2]), map(float, row[3::2]))
                preds[(ent, rid)] = set(scores)

        return Cls(preds, gt)


@dataclass(frozen=True)
class RankEvaluator:
    """IRT2 Ranking Evaluation."""

    ranks: Ranks
    gt: GroundTruth

    def compute_metrics(self, max_rank: int = None) -> dict:
        """
        Compute ranking evaluation metrics for IRT2.open_ranking*
        or IRT2.open_kgc*

        Parameters
        ----------
        max_rank : int
            If the predicted rank of a predicted MID is > `max_rank`,
            it is clipped from the metric computation.
            This means that the rank is set to 0
            (i.e. the MID was not correctly predicted by the model).

        """
        tf_ranks = self.tf_ranks(max_rank=max_rank)

        return {
            "micro": {
                "mrr": micro_mrr(tf_ranks.values()),
                "hits_at_k": "Not implemented yet",
            },
            "macro": {
                "mrr": macro_mrr(tf_ranks.values()),
                "hits_at_k": "Not implemented yet",
            },
        }

    def tf_ranks(self, max_rank: int = None) -> dict[Task, tuple[int]]:
        """
        Return the target-filtered rank for each ground truth value.

        If the MID/VID is not in pred or the rank is greater than
        `max_rank`, the rank is 0.

        Parameters
        ---------
        max_rank : int
            The max tf rank used. If the tf rank of a task
            is greater than `max_rank`, the returned rank is 0.

        """

        def get(rank: Rank) -> int:
            if rank is None:
                return 0
            if max_rank is not None and max_rank < rank.filtered:
                return 0

            return rank.filtered

        tf_ranks = defaultdict(list)
        for (ent, rid), gt_ents in self.gt.items():

            for gt_ent in gt_ents:
                rank = self.ranks.get((ent, rid, gt_ent), None)
                tf_ranks[(ent, rid)].append(get(rank))

        return {k: tuple(v) for k, v in tf_ranks.items()}


#
# command line interface
#

log = logging.getLogger(__name__)
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


pretty_errors.configure(
    filename_display=pretty_errors.FILENAME_EXTENDED,
    lines_after=2,
    line_number_first=True,
)


@click.group()
def main():
    """Use irt2m from the command line."""
    irt2.init_logging()

    print(
        """
              ┌─────────────────────────────┐
              │ IRT2 COMMAND LINE INTERFACE │
              └─────────────────────────────┘
        """
    )

    log.info(f"initialized root path: {irt2.ENV.DIR.ROOT}")
    log.info(f"executing from: {os.getcwd()}")


@main.command(name="evaluate-owkgc")
def cli_eval_owkgc():
    """Evaluate the open-world KGC task."""
    pass


@main.command(name="evaluate-ranking")
@click.option(
    "--irt2",
    type=str,
    required=True,
    help="path to irt2 data",
)
@click.option(
    "--predictions",
    type=str,
    required=True,
    help="file containing mid predictions",
)
@click.option(
    "--split",
    type=str,
    required=True,
    help="one of validation, test",
)
@click.option(
    "--max-rank",
    type=int,
    default=100,
    help="only consider the first n ranks (target filtered)",
)
@click.option(
    "--model-name",
    type=str,
    help="optional name of the model",
)
@click.option(
    "--out",
    type=str,
    help="optional output file for metrics",
)
def cli_eval_ranking(
    irt2: str,
    predictions: str,
    split: str,
    max_rank: int,
    model_name: Optional[str],
    out: Optional[str],
):
    """Evaluate the open-world ranking task."""
    print("loading IRT2 dataset")
    irt2 = IRT2.from_dir(kpath(irt2, is_dir=True))
    print(f"loaded: {irt2}")

    gt = dict(
        validation=irt2.open_ranking_val_heads,
        test=irt2.open_ranking_test_heads,
    )[split]

    print(f"reading predictions from {predictions}...")
    ranks = Ranks.from_csv(kpath(predictions, is_file=True), gt)

    print("running evaluation...")
    evaluator = RankEvaluator(ranks, gt)

    metrics = evaluator.compute_metrics(max_rank=max_rank)
    report = dict(
        model_name=model_name or "unknown",
        date=datetime.now().isoformat(),
    )

    report |= metrics

    print("\nreport:")
    print(yaml.safe_dump(report))

    if out:
        out = kpath(out, exists=False)
        with out.open(mode="w") as fd:
            yaml.safe_dump(report, fd)


@main.command(name="evaluate-kgc")
@click.option(
    "--irt2",
    type=str,
    required=True,
    help="path to irt2 data",
)
@click.option(
    "--predictions",
    type=str,
    required=True,
    help="file containing vid predictions",
)
@click.option(
    "--split",
    type=str,
    required=True,
    help="one of validation, test",
)
@click.option(
    "--max-rank",
    type=int,
    default=100,
    help="only consider the first n ranks (target filtered)",
)
@click.option(
    "--model-name",
    type=str,
    help="optional name of the model",
)
@click.option(
    "--out",
    type=str,
    help="optional output file for metrics",
)
def cli_eval_kgc(
    irt2: str,
    predictions: str,
    split: str,
    max_rank: int,
    model_name: Optional[str],
    out: Optional[str],
):
    """Evaluate the open-world ranking task."""
    print("loading IRT2 dataset")
    irt2 = IRT2.from_dir(kpath(irt2, is_dir=True))
    print(f"loaded: {irt2}")

    gt = dict(
        validation=irt2.open_kgc_val_heads,
        test=irt2.open_kgc_test_heads,
    )[split]

    print(f"reading predictions from {predictions}...")
    ranks = Ranks.from_csv(kpath(predictions, is_file=True), gt)

    print("running evaluation...")
    evaluator = RankEvaluator(ranks, gt)

    metrics = evaluator.compute_metrics(max_rank)
    report = dict(
        model_name=model_name or "unknown",
        date=datetime.now().isoformat(),
    )

    report |= metrics

    print("\nreport:")
    print(yaml.safe_dump(report))

    if out:
        out = kpath(out, exists=False)
        with out.open(mode="w") as fd:
            yaml.safe_dump(report, fd)
