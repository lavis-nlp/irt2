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
from typing import Collection, Optional

import click
import pretty_errors
import yaml
from ktz.filesystem import path as kpath

import irt2
from irt2.dataset import IRT2
from irt2.types import MID, RID, VID

#
# open world knowledge graph completion
#


#
# open world ranking task
#

RankTask = dict[(VID, RID), set[MID]]


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


RANK = Optional[int]
RANK_TF = Optional[int]


@dataclass(frozen=True)
class EvaluationDataPoints:
    """Input format for evaluation metrics."""

    datapoints: dict[(VID, RID, MID), (RANK, RANK_TF)]

    def get_rank(self, vid: VID, rid: RID, mid: MID, tf: bool) -> Optional[int]:
        """Obtain the rank for a specific triple."""
        assert (vid, rid, mid) in self.datapoints, f"{vid=} {rid=} {mid=} not found!"

        rank, rank_tf = self.datapoints[(vid, rid, mid)]
        return rank_tf if tf else rank

    @cached_property
    def tasks(self) -> set[(VID, RID)]:
        """Obtain all tasks from the datapoints."""
        return {(vid, rid) for (vid, rid, _mid) in self.datapoints.keys()}

    @staticmethod
    def ranks(
        gt: set[MID],
        pred: list[MID],
    ) -> dict[MID, tuple[RANK, RANK_TF]]:
        """
        Get target-filtered ranks based on the provided ground truth.

        Given a set of ground-truth MID values, for each correct mid in the
        prediction list, the rank and target-filtered rank are associated and
        returned.

        Parameters
        ----------
        gt : set[MID]
            The ground truth mids.

        pred : list[MID]
            The predicted mids (ordered by score).

        """
        result = {}

        for rank, pred_mid in enumerate(pred, start=1):
            if pred_mid in gt:
                assert pred_mid not in result, f"already encountered {pred_mid=}!"

                rank_tf = rank - len(result)
                result[pred_mid] = rank, rank_tf

        return result

    @staticmethod
    def from_predictions(
        gt: RankTask,
        pred: dict[(VID, RID), list[MID]],
    ):
        """
        Create input data for evaluation metrics.

        Parameters
        ----------
        gt : RankTask
            The ground truth MID values for each (VID, RID).

        pred : dict[(VID, RID), list[MID]]
            The predicted MIDs for each (VID, RID).
            The values should be ordered by their scores (i.e. highest rank first).

        """
        assert len(gt.keys()) == len(pred.keys()), "ground-truth/predictions mismatch!"

        datapoints = {}
        for (vid, rid), pred_mids in pred.items():

            rankdic = EvaluationDataPoints.ranks(
                gt=gt[(vid, rid)],
                pred=pred_mids,
            )

            for mid, ranks in rankdic.items():
                datapoints[(vid, rid, mid)] = ranks

        return EvaluationDataPoints(datapoints)

    @staticmethod
    def from_csv(path: str, gt: RankTask):
        """
        Load the evaluation data from csv file.

        CSV file format:
        [
            [ vid, rid, pred_mid1, score_for_mid1, pred_mid2, score_for_mid2, ... ],
            ...
        ]

        A score of 0 is handled as if the mid was not predicted for this (VID, RID).
        The order of the predictions do not matter as they are sorted by score before
        ranks are calculated.

        Parameters
        ----------
        path : str
            Where to load the csv file from.

        gt : RankTask
            The ground truth mids for each (VID, RID).

        """
        predictions = {}

        with kpath(path, is_file=True).open(mode="r") as fd:
            reader = csv.reader(fd, delimiter=",")

            for row in reader:
                vid, rid = map(int, row[:2])

                # convert [1, 2.3, 2, 4.3] -> [(2, 4.3), (1, 2.3)]
                predictions[(vid, rid)] = sorted(
                    zip(map(int, row[2::2]), map(float, row[3::2])),
                    key=lambda tup: tup[1],
                    reverse=True,
                )

        assert len(predictions.keys()) == len(gt.keys()), "missing predictions!"

        # transform csv format to EvaluationDataPoints

        datapoints = {}
        for (vid, rid), pred in predictions.items():

            # TBD disabled: what about negative scores?
            # pred_mids are those with a score > 0
            # pred_mids = [mid for (mid, score) in pred if score > 0]

            rankdic = EvaluationDataPoints.ranks(
                gt=gt[(vid, rid)],
                pred=[mid for mid, _ in pred],
            )

            for mid, ranks in rankdic.items():
                datapoints[(vid, rid, mid)] = ranks

        return EvaluationDataPoints(datapoints)


class RankingEvaluator:
    """IRT2 Ranking Evaluation."""

    gt: RankTask
    pred: EvaluationDataPoints

    def __init__(
        self,
        gt: RankTask,
        pred: EvaluationDataPoints,
    ):
        self.gt = gt
        self.pred = pred

    def compute_metrics(
        self,
        max_rank: int,
    ) -> dict:
        """
        Compute ranking evaluation metrics for IRT2.open_ranking*.

        Parameters
        ----------
        model_name : str
            The name of the model.
            Is used for identifying the computed metrics.

        gt : RankTask
            The ground truth MIDs for each (VID, RID).

        pred : EvaluationDataPoints
            The predicted rankings.

        max_rank : int
            If the predicted rank of a predicted MID is > `max_rank`,
            it is clipped from the metric computation.
            This means that the rank is set to 0
            (i.e. the MID was not correctly predicted by the model).

        """
        tf_ranks = self.get_tf_ranks(max_rank=max_rank)

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

    def get_tf_ranks(
        self,
        max_rank: int,
    ) -> dict[(VID, RID), list[RANK_TF]]:
        """
        Return the target-filtered rank in pred for each MID in gt.

        If the MID is not in pred or the rank is greater than `max_rank`, the rank is 0.

        Parameters
        ---------
        gt : RankTask
            The ground truth MID values for each (VID, RID).

        pred : dict[(VID, RID, MID), tuple[RANK, RANK_TF]]
            The rank and target-filtering rank for each predicted MID in (VID, RID).

        max_rank : int
            The max tf rank used. If the tf rank of an (VID, RID, MID) is greater
            than `max_rank`, the returned rank is 0.

        """
        tf_ranks = defaultdict(list)

        for (vid, rid), mids in self.gt.items():

            for mid in mids:
                _, tf = self.pred.datapoints.get((vid, rid, mid), (0, 0))
                tf = tf if tf <= max_rank else 0
                tf_ranks[(vid, rid)].append(tf)

        return dict(tf_ranks)


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
    help="file containing rank predictions",
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
    predictions = EvaluationDataPoints.from_csv(
        kpath(predictions, is_file=True),
        gt,
    )

    print("running evaluation...")
    evaluator = RankingEvaluator(gt, predictions)

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
