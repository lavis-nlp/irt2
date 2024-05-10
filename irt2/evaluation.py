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
import gzip
import logging
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from statistics import mean
from typing import Literal

import yaml
from ktz.filesystem import path as kpath

import irt2
from irt2.dataset import IRT2
from irt2.types import VID, Entity, GroundTruth, Task

log = logging.getLogger(__name__)
tee = irt2.tee(log)


# kgc: MID, RID query and (closed-world) VID targets
# ranking: VID, RID query and (open-world) MID targets

Scores = Iterable[tuple[Entity, float]]
Predictions = Iterable[tuple[Task, Scores]]
PredictionsDict = dict[Task, Scores]


#
# rank metrics
#

#
# The macro-averaged score (or macro score) is computed using the
# arithmetic mean (aka unweighted mean) of all the per-class scores.

# Micro averaging computes a global average score. Micro-averaging
# computes the proportion of correctly classified observations
# out of all observations.
#
# A "class" in our setting is a task from the ranking or kgc tasks.
#
# see
# noqa https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
# noqa https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f


def rr(ranks: Iterable[int]) -> list[float]:
    """Calculate the mrr for each rank in tfs."""
    assert all(rank >= 0 for rank in ranks)
    return [1 / rank if rank != 0 else 0 for rank in ranks]


def micro_mrr(rankcol: Iterable[Iterable[int]]) -> float:
    """Compute the MICRO Mean Reciprocal Rank (MRR)."""
    return mean(inv for ranks in rankcol for inv in rr(ranks))


def macro_mrr(rankcol: Iterable[Iterable[int]]) -> float:
    """Compute the MACRO Mean Reciprocal Rank (MRR)."""
    return mean(mean(rr(ranks)) for ranks in rankcol)


def hits_at_k(ranks: Iterable[int], k: int) -> float:
    return mean(1 if rank > 0 and rank <= k else 0 for rank in ranks)


def micro_hits_at_k(rankcol: Iterable[Iterable[int]], k: int) -> float:
    """Compute hits@k for all tasks."""
    assert k > 0
    flat = (rank for ranks in rankcol for rank in ranks)
    return hits_at_k(flat, k=k)


def macro_hits_at_k(rankcol: Iterable[Iterable[int]], k: int) -> float:
    assert k > 0
    return mean(hits_at_k(ranks, k=k) for ranks in rankcol)


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
    """

    gt: GroundTruth

    def tasks(self) -> set[Task]:
        """Obtain all tasks from the datapoints."""
        return {(ent, rid) for (ent, rid, _) in self}

    def __init__(
        self,
        gt: GroundTruth,
        *predictions: tuple[Task, Scores],
    ):
        super().__init__()
        self.gt = gt
        self._tasks_added = set()
        self.add(predictions)

    def _strip_raw(
        self,
        targets: set[Entity],
        raw: Iterable[tuple[Entity, float]],
    ):
        ordered = sorted(raw, key=lambda t: t[1], reverse=True)
        counted = enumerate(ordered, start=0)

        return [(ent, pos, score) for (pos, (ent, score)) in counted if ent in targets]

    def _add(self, task: Task, *samples: tuple[VID, int, float]):
        assert task in self.gt, f"{task=} not in ground truth"

        # a task may only be added once, otherwise target filtering
        # won't work (we remember how many TP are skipped here)
        assert (
            task not in self._tasks_added
        ), f"{task=} already added, target filtering violated"
        self._tasks_added.add(task)

        last = math.inf
        for skip, (eid, position, score) in enumerate(samples):
            # yaml fails with numpy and torch dtypes...
            # To alleviate all the headache, data is eventually
            # converted here to their corresponding primitive types
            eid, position, score = int(eid), int(position), float(score)

            assert eid in self.gt[task], f"{eid=} not in ground truth"
            assert position >= 0, "positions must be positive"
            assert score <= last, "predictions are not sorted"
            last = score

            triple = task + (eid,)
            assert triple not in self, f"{task=}: {triple=} already present"

            rank = position + 1
            filtered = rank - skip
            assert (
                filtered >= 0
            ), f"{task=}: filtered < 0 - are positions with equal score sorted?"

            self[triple] = Rank(
                value=rank,
                filtered=filtered,
                score=score,
            )

    def add(self, predictions: Predictions):
        for prediction in predictions:
            task, scores = prediction
            samples = self._strip_raw(self.gt[task], scores)
            self._add(task, *samples)

    def add_dict(self, pred: PredictionsDict):
        for task, raw in pred.items():
            predictions = self._strip_raw(self.gt[task], raw)
            self._add(task, *predictions)


class RankEvaluator:
    """IRT2 Ranking Evaluation."""

    data: dict[str, tuple[Ranks, GroundTruth]]

    def __init__(self, **kwargs: tuple[Ranks, GroundTruth]):
        # for name, (ranks, gt) in kwargs.items():
        assert "all" not in set(kwargs)
        assert all(len(kwarg) == 2 for kwarg in kwargs.values())

        self.data = kwargs

    def _compute_metrics(self, rank_col, ks) -> dict:
        """
        Compute ranking evaluation metrics for IRT2.open_ranking*
        or IRT2.open_kgc*
        """

        micro = {"mrr": micro_mrr(rank_col)}
        micro |= {f"hits_at_{k}": micro_hits_at_k(rank_col, k) for k in ks}

        macro = {"mrr": macro_mrr(rank_col)}
        macro |= {f"hits_at_{k}": macro_hits_at_k(rank_col, k) for k in ks}

        res = {
            "micro": micro,
            "macro": macro,
        }

        return res

    @cache
    def compute_metrics(
        self,
        max_rank: int | None = None,
        ks: Iterable[int] = (1, 10),
    ) -> dict:
        result = {}

        all_rank_col = []
        tf_ranks = self.tf_ranks(max_rank=max_rank)
        for name, _ in self.data.items():
            rank_col = list(tf_ranks[name].values())
            all_rank_col += rank_col
            result[name] = self._compute_metrics(rank_col, ks)

        result["all"] = self._compute_metrics(all_rank_col, ks)
        return result

    @cache
    def tf_ranks(
        self,
        max_rank: int | None = None,
    ) -> dict[str, dict[Task, tuple[int, ...]]]:
        """
        Return a tuple of target-filtered ranks for each ground truth item.

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

        result: dict[str, dict[Task, tuple[int, ...]]] = {}
        for name, (ranks, gt) in self.data.items():
            tf_ranks = defaultdict(list)
            for (ent, rid), gt_ents in gt.items():
                for gt_ent in gt_ents:
                    rank = ranks.get((ent, rid, gt_ent), None)
                    tf_ranks[(ent, rid)].append(get(rank))

            result[name] = {k: tuple(v) for k, v in tf_ranks.items()}

        return result


def load_gt(
    ds: IRT2,
    task: Literal["kgc", "ranking"],
    split: Literal["validation", "test"],
):
    assert task in {"kgc", "ranking"}
    assert split in {"validation", "test"}

    gt_head, gt_tail = dict(
        kgc=dict(
            validation=(
                ds.open_kgc_val_heads,
                ds.open_kgc_val_tails,
            ),
            test=(
                ds.open_kgc_test_heads,
                ds.open_kgc_test_tails,
            ),
        ),
        ranking=dict(
            validation=(
                ds.open_ranking_val_heads,
                ds.open_ranking_val_tails,
            ),
            test=(
                ds.open_ranking_test_heads,
                ds.open_ranking_test_tails,
            ),
        ),
    )[task][split]

    return gt_head, gt_tail


def create_report(
    metrics: dict,
    ds: IRT2,
    task: str,
    split: str,
    model: str | None,
    filenames: dict[str, str | Path] | None = None,
    out: str | Path | None = None,
):
    report = dict(
        date=datetime.now().isoformat(),
        dataset=ds.name,
        model=model or "unknown",
        task=task,
        split=split,
        metrics=metrics,
    )

    if filenames:
        report |= {key: str(val) for key, val in filenames.items()}

    irt2.console.log("\nreport:")
    irt2.console.log(yaml.safe_dump(report))

    if out:
        tee(f"write report to {out}")

        fp = kpath(out, exists=False)
        with fp.open(mode="w") as fd:
            yaml.safe_dump(report, fd)

    return report


def load_csv(path: str | Path) -> Iterable[tuple[Task, Scores]]:
    """
        Load the evaluation data from csv file.

    CSV file format:

    For Ranking:
      vid, rid, pred_mid1, score_for_mid1, pred_mid2, score_for_mid2, ...

    For kgc:
      mid, rid, pred_vid1, score_for_vid1, pred_vid2, score_for_vid2, ...

    The order of the predictions do not matter as they are sorted by score
    before ranks are calculated. If the csv file ends with .gz, it is assumed
    to be a gzipped file.

    Parameters
    ----------
    path : str
        Where to load the csv file from.

    """
    fp = kpath(path, is_file=True)
    fd = gzip.open(fp, mode="rt") if fp.suffix == ".gz" else fp.open(mode="r")

    with fd:
        reader = csv.reader(fd, delimiter=",")

        row: list[str]
        for line, row in enumerate(reader):
            assert len(row) % 2 == 0 and len(row) > 2, f"error in line {line}"

            task = tuple(map(int, row[:2]))
            scores = zip(map(int, row[2::2]), map(float, row[3::2]))

            assert len(task) == 2
            yield task, scores


def evaluate(
    ds: IRT2,
    task: Literal["kgc", "ranking"],
    split: Literal["validation", "test"],
    head_predictions: Predictions,
    tail_predictions: Predictions,
    max_rank: int = 100,
) -> dict:
    tee("running evaluation...")

    gt_head, gt_tail = load_gt(ds, task=task, split=split)
    ranks_head, ranks_tail = Ranks(gt_head), Ranks(gt_tail)

    ranks_head.add(head_predictions)
    ranks_tail.add(tail_predictions)

    evaluator = RankEvaluator(
        head=(ranks_head, gt_head),
        tail=(ranks_tail, gt_tail),
    )

    metrics = evaluator.compute_metrics(max_rank)
    return metrics
