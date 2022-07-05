import tempfile
import textwrap

import pytest

from irt2 import evaluation as eval
from irt2.evaluation import Rank, RankEvaluator, Ranks


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

        actual = Ranks(gt).add_dict(pred)
        assert expected == actual

    def test_ranks_from_predictions_wrong_pred(self):

        gt = {(100, 200): {2, 3, 4}}
        pred = {(100, 200): score(1, 2, 3)}

        expected = {
            (100, 200, 2): Rank(value=2, filtered=2, score=1 / 2),
            (100, 200, 3): Rank(value=3, filtered=2, score=1 / 3),
        }

        actual = Ranks(gt).add_dict(pred)
        assert expected == actual

    # tf_ranks function

    def test_tf_ranks_basic(self):

        gt = {(100, 200): {1, 2, 3}}
        pred = {(100, 200): score(2, 3, 4)}

        # for each value in gt:
        # - if MID exists in pred: the target filtering rank should be returned
        # - else 0 should be returned

        ranks = Ranks(gt).add_dict(pred)
        result = RankEvaluator(test=(ranks, gt)).tf_ranks(max_rank=100)["test"]

        assert len(result) == 1
        assert sorted(result[(100, 200)]) == [0, 1, 1]

    def test_tf_ranks_hole(self):

        gt = {(100, 200): {1, 3, 5}}
        pred = {(100, 200): score(2, 3, 4, 5)}

        ranks = Ranks(gt).add_dict(pred)
        result = RankEvaluator(test=(ranks, gt)).tf_ranks(max_rank=100)["test"]

        assert len(result) == 1
        assert sorted(result[(100, 200)]) == [0, 2, 3]

    def test_tf_ranks_max_rank(self):
        gt = {(100, 200): {1, 101}}  # 1 is in it, 101 is not

        pred = {(100, 200): score(*range(1, 101))}
        ranks = Ranks(gt).add_dict(pred)
        result = RankEvaluator(test=(ranks, gt)).tf_ranks(max_rank=100)["test"]

        assert sorted(result[(100, 200)]) == [0, 1]

    def test_tf_ranks_multiple_gt(self):
        gt = {
            (100, 200): {1, 2, 3, 5},
            (100, 300): {1, 2},
        }

        pred = {
            (100, 200): score(2, 3, 4),
            (100, 300): score(2, 3, 4),
        }

        ranks = Ranks(gt).add_dict(pred)
        result = RankEvaluator(test=(ranks, gt)).tf_ranks()["test"]

        assert sorted(result[(100, 200)]) == [0, 0, 1, 1]
        assert sorted(result[(100, 300)]) == [0, 1]

    def test_tf_ranks_multiple_splits(self):
        gt1 = {(100, 200): {1, 2}}
        gt2 = {(100, 200): {3, 4}}

        pred1 = {(100, 200): score(2, 3, 4)}
        pred2 = {(100, 200): score(3, 4)}

        ranks1 = Ranks(gt1).add_dict(pred1)
        ranks2 = Ranks(gt2).add_dict(pred2)

        result = RankEvaluator(
            name1=(ranks1, gt1),
            name2=(ranks2, gt2),
        )

        result1 = result.tf_ranks()["name1"]
        result2 = result.tf_ranks()["name2"]

        assert sorted(result1[(100, 200)]) == [0, 1]
        assert sorted(result2[(100, 200)]) == [1, 1]

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

        ranks = Ranks(gt).add_dict(pred)
        pred_tfs = RankEvaluator(test=(ranks, gt)).tf_ranks()["test"]

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
            (100, 200): {10, 2, 30, 4, 50},
            (100, 300): {10, 20, 30},
            (100, 400): {50, 60},
        }

        pred = {
            (100, 200): score(2, 3, 4, 1000),
            (100, 300): score(1),
            (100, 400): score(60, 50),
        }

        ranks = Ranks(gt).add_dict(pred)
        pred_tfs = RankEvaluator(test=(ranks, gt)).tf_ranks()["test"]

        # fmt: off
        ref_micro = 1 / 10 * (
            1 + 0 + 1/2 + 0 + 0 +
            0 + 0 + 0 +
            1 + 1
        )
        # fmt: on
        micro = eval.micro_mrr(pred_tfs.values())
        assert micro == pytest.approx(ref_micro)

        # fmt: off
        ref_macro = 1 / 3 * (
            1/5 * (1 + 0 + 1/2 + 0 + 0) +
            1/3 * (0 + 0 + 0) +
            1/2 * (1 + 1)
        )
        # fmt: on
        macro = eval.macro_mrr(pred_tfs.values())
        assert macro == pytest.approx(ref_macro)

    # hits

    def test_hits_at_k(self):

        gt = {
            #     rank:  0  1  0  3   6
            #  rank_tf:  0  1  0  2   4
            (100, 200): {1, 2, 4, 5, 10},
            (100, 300): {10, 20, 30},
            (100, 400): {50, 60},
        }

        pred = {
            (100, 200): score(2, 3, 5, 6, 7, 10, 50),
            (100, 300): score(1),
            (100, 400): score(60, 50),
        }

        ranks = Ranks(gt).add_dict(pred)
        pred_tfs = RankEvaluator(test=(ranks, gt)).tf_ranks()["test"]

        # HITS@1 aka Accuracy

        micro_hits = eval.micro_hits_at_k(pred_tfs.values(), k=1)
        # fmt: off
        expected = 1 / 10 * (
            1 + 0 + 0 + 0 + 0 +
            0 + 0 + 0 +
            1 + 1
        )
        # fmt: on
        assert micro_hits == pytest.approx(expected)

        macro_hits = eval.macro_hits_at_k(pred_tfs.values(), k=1)
        # fmt: off
        expected = 1 / 3 * (
            1 / 5 * (1 + 0 + 0 + 0 + 0) +
            1 / 3 * (0 + 0 + 0) +
            1 / 2 * (1 + 1)
        )
        # fmt: on
        assert macro_hits == pytest.approx(expected)

        # HITS@5

        micro_hits = eval.micro_hits_at_k(pred_tfs.values(), k=5)
        # fmt: off
        expected = 1 / 10 * (
            1 + 1 + 1 + 0 + 0 +
            0 + 0 + 0 +
            1 + 1
        )
        # fmt: on
        assert micro_hits == pytest.approx(expected)

        macro_hits = eval.macro_hits_at_k(pred_tfs.values(), k=5)
        # fmt: off
        expected = 1 / 3 * (
            1 / 5 * (1 + 1 + 1 + 0 + 0) +
            1 / 3 * (0 + 0 + 0) +
            1 / 2 * (1 + 1)
        )
        # fmt: on
        assert macro_hits == pytest.approx(expected)

    # csv

    def test_csv(self):

        # like test_get_tf_ranks_multiple

        gt = {
            (100, 200): {1, 2, 3, 5},
            (100, 300): {1, 2},
        }

        with tempfile.NamedTemporaryFile(mode="w") as fd:
            csv = textwrap.dedent(
                """
                100, 200, 2, 0.5, 3, 0.3, 4, 0.25
                100, 300, 2, 0.5, 3, 0.3, 4, 0.25
                """
            )

            fd.write(csv.strip())
            fd.flush()

            ranks = Ranks(gt).add_csv(path=fd.name, gt=gt)

        result = RankEvaluator(test=(ranks, gt)).tf_ranks()["test"]

        assert sorted(result[(100, 200)]) == [0, 0, 1, 1]
        assert sorted(result[(100, 300)]) == [0, 1]
