import pytest

from irt2 import evaluate as eval
from irt2.evaluate import Rank, RankEvaluator, Ranks


def score(*col):
    return {(i, 1 / i) for i in col}


class TestEvaluationRanking:

    # ranks class

    def test_ranks_from_predictions(self):

        gt = {(100, 200): {1, 2, 3}}
        pred = {(100, 200): score(1, 3)}

        expected = {
            (100, 200, 1): Rank(value=1, filtered=1, score=1 / 1),
            (100, 200, 3): Rank(value=2, filtered=1, score=1 / 3),
        }

        actual = Ranks(pred, gt)
        assert expected == actual

    def test_ranks_from_predictions_wrong_pred(self):

        gt = {(100, 200): {2, 3, 4}}
        pred = {(100, 200): score(1, 2, 3)}

        expected = {
            (100, 200, 2): Rank(value=2, filtered=2, score=1 / 2),
            (100, 200, 3): Rank(value=3, filtered=2, score=1 / 3),
        }

        actual = Ranks(pred, gt)
        assert expected == actual

    # tf_ranks function

    def test_tf_ranks_basic(self):

        gt = {(100, 200): {1, 2, 3}}
        pred = {(100, 200): score(2, 3, 4)}

        # for each value in gt:
        # - if MID exists in pred: the target filtering rank should be returned
        # - else 0 should be returned

        ranks = Ranks(pred, gt)
        result = RankEvaluator(gt=gt, ranks=ranks).tf_ranks(max_rank=100)

        assert len(result) == 1
        assert sorted(result[(100, 200)]) == [0, 1, 1]

    def test_tf_ranks_max_rank(self):
        gt = {(100, 200): {1, 101}}  # 1 is in it, 101 is not

        pred = {(100, 200): score(*range(1, 101))}
        ranks = Ranks(pred=pred, gt=gt)
        result = RankEvaluator(gt=gt, ranks=ranks).tf_ranks(max_rank=100)

        assert sorted(result[(100, 200)]) == [0, 1]

    def test_get_tf_ranks_multiple(self):
        gt = {
            (100, 200): {1, 2, 3, 5},
            (100, 300): {1, 2},
        }

        pred = {
            (100, 200): score(2, 3, 4),
            (100, 300): score(2, 3, 4),
        }

        ranks = Ranks(pred=pred, gt=gt)
        result = RankEvaluator(gt=gt, ranks=ranks).tf_ranks()

        assert sorted(result[(100, 200)]) == [0, 0, 1, 1]
        assert sorted(result[(100, 300)]) == [0, 1]

    # mrr computation

    def test_mrr_simple(self):

        gt = {
            (100, 200): {1, 2, 3},
            (100, 300): {10, 20},
        }
        pred = {
            (100, 200): score(1, 2),
            (100, 300): score(10, 40),
        }

        ranks = Ranks(pred=pred, gt=gt)
        pred_tfs = RankEvaluator(ranks=ranks, gt=gt).tf_ranks()

        # fmt: off
        expected_micro = 1 / 5 * (
            1 + 1 + 0 +
            1 + 0
        )
        # fmt: on
        actual_micro = eval.micro_mrr(pred_tfs.values())
        assert actual_micro == pytest.approx(expected_micro)

        # fmt: off
        expected_macro = 1 / 2 * (
            1/3 * (1 + 1 + 0) +
            1/2 * (1 + 0)
        )
        # fmt: on
        actual_macro = eval.macro_mrr(pred_tfs.values())
        assert actual_macro == pytest.approx(expected_macro)

    def test_mrr(self):

        gt = {
            (100, 200): {1, 2, 3, 4, 5},
            (100, 300): {10, 20, 30},
            (100, 400): {50, 60},
        }

        pred = {
            (100, 200): score(2, 4, 10, 50),
            (100, 300): score(1),
            (100, 400): score(60, 50),
        }

        ranks = Ranks(pred=pred, gt=gt)
        pred_tfs = RankEvaluator(ranks=ranks, gt=gt).tf_ranks()

        # fmt: off
        ref_micro = 1 / 10 * (
            1 + 1 + 0 + 0 + 0 +
            0 + 0 + 0 +
            1 + 1
        )
        # fmt: on
        micro = eval.micro_mrr(pred_tfs.values())
        assert micro == pytest.approx(ref_micro)

        # fmt: off
        ref_macro = 1 / 3 * (
            1/5 * (1 + 1 + 0 + 0 + 0) +
            1/3 * (0 + 0 + 0) +
            1/2 * (1 + 1)
        )
        # fmt: on
        macro = eval.macro_mrr(pred_tfs.values())
        assert macro == pytest.approx(ref_macro)

    # hits
