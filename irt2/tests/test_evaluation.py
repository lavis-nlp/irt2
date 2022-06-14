import pytest

from irt2.evaluate import EvaluationDataPoints, RankingEvaluator


class TestEvaluationRanking:

    # EvaluationDataPoints class

    def test_evaluation_data_points_from_predictions(self):

        gt = {(100, 200): {1, 2, 3}}
        pred = {(100, 200): [1, 3]}

        expected = {(100, 200, 1): (1, 1), (100, 200, 3): (2, 1)}
        actual = EvaluationDataPoints.from_predictions(gt=gt, pred=pred).datapoints

        assert expected == actual

    def test_evaluation_data_points_from_predictions_wrong_pred(self):

        gt = {(100, 200): {1, 2, 3}}
        pred = {(100, 200): [1, 3, 10]}

        expected = {(100, 200, 1): (1, 1), (100, 200, 3): (2, 1)}
        actual = EvaluationDataPoints.from_predictions(gt=gt, pred=pred).datapoints

        assert expected == actual

    # get_tf_ranks function

    def test_get_tf_ranks_basic(self):

        gt = {
            (100, 200): {1, 2, 3},
        }
        pred = {
            (100, 200, 1): (1, 1),
            (100, 200, 3): (3, 2),
        }

        expected = {
            # for each value in gt:
            # - if MID exists in pred: the target filtering rank should be returned
            # - else 0 should be returned
            # mid in gt: 1, 2, 3
            (100, 200): [1, 0, 2]
        }
        actual = RankingEvaluator.get_tf_ranks(gt=gt, pred=pred, max_rank=100)

        assert expected == actual

    def test_get_tf_ranks_max_rank(self):
        gt = {
            (100, 200): {1, 2, 3},
        }
        pred = {
            (100, 200, 1): (100, 100),
            (100, 200, 3): (102, 101),
        }

        expected = {(100, 200): [100, 0, 0]}
        actual = RankingEvaluator.get_tf_ranks(gt=gt, pred=pred, max_rank=100)

        assert expected == actual

    def test_get_tf_ranks_multiple(self):
        gt = {
            (100, 200): {1, 2, 3, 4},
            (100, 300): {1, 20, 5},
        }
        pred = {
            (100, 200, 1): (1, 1),
            (100, 200, 2): (2, 1),
            (100, 200, 3): (3, 1),
            (100, 300, 1): (1, 1),
            (100, 300, 5): (5, 4),
        }

        expected = {(100, 200): [1, 1, 1, 0], (100, 300): [1, 0, 4]}
        actual = RankingEvaluator.get_tf_ranks(gt=gt, pred=pred, max_rank=100)

        assert expected == actual

    # mrr computation

    def test_mrr_simple(self):

        gt = {(100, 200): {1, 2, 3}, (100, 300): {10, 20}}
        pred = {
            (100, 200): [1, 2],
            (100, 300): [10, 40],
        }

        pred_dp = EvaluationDataPoints.from_predictions(gt=gt, pred=pred).datapoints
        pred_tfs = RankingEvaluator.get_tf_ranks(gt=gt, pred=pred_dp, max_rank=100)

        expected_micro = 1 / 5 * ((1 / 1 + 1 / 1 + 0) + (1 / 1 + 0))
        actual_micro = RankingEvaluator.micro_mrr(pred_tfs.values())
        assert actual_micro == pytest.approx(expected_micro)

        expected_macro = 1 / 2 * (1 / 3 * (1 / 1 + 1 / 1 + 0) + 1 / 2 * (1 / 1 + 0))
        actual_macro = RankingEvaluator.macro_mrr(pred_tfs.values())
        assert actual_macro == pytest.approx(expected_macro)

    def test_mrr(self):

        gt = {
            (100, 200): {1, 2, 3, 4, 5},
            (100, 300): {10, 20, 30},
            (100, 400): {50, 60},
        }

        pred = {
            (100, 200): [2, 10, 4, 50],
            (100, 300): [1],
            (100, 400): [60, 50],
        }

        pred_dp = EvaluationDataPoints.from_predictions(gt=gt, pred=pred).datapoints
        pred_tfs = RankingEvaluator.get_tf_ranks(gt=gt, pred=pred_dp, max_rank=100)

        expected_micro = 1 / 10 * ((1 / 1 + 0 + 1 / 2 + 0) + (0) + (1 / 1 + 1 / 1))
        actual_micro = RankingEvaluator.micro_mrr(pred_tfs.values())
        assert actual_micro == pytest.approx(expected_micro)

        # fmt: off
        expected_macro = 1 / 3 * (
            1 / 5 * (0 + 1 / 1 + 0 + 1 / 2 + 0) +   # noqa: W504
            1 / 3 * (0 + 0 + 0) +  # noqa: W504
            1 / 2 * (1 / 1 + 1 / 1)
        )
        # fmt: on

        actual_macro = RankingEvaluator.macro_mrr(pred_tfs.values())
        assert actual_macro == pytest.approx(expected_macro)
