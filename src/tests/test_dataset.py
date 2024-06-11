import irt2
import pytest
from irt2 import loader
from irt2.dataset import IRT2
from irt2.types import Split

DATASETS = (
    "irt2/tiny",
    "irt2/small",
    "irt2/medium",
    "irt2/large",
    "blp/wn18rr",
    # "blp/wikidata5m",
    "blp/fb15k237",
)


@pytest.fixture(scope="class")
def datasets():
    fpath = irt2.ENV.DIR.CONF / "datasets" / "original.yaml"
    return dict(loader.from_config_file(fpath, only=DATASETS))


@pytest.fixture(params=DATASETS)
def ds(datasets, request):
    return datasets[request.param]


def disjoint(*sets):
    assert not set.intersection(*sets)

    # for debugging
    # for a, b in combinations(sets, r=2):
    #     assert not (a & b)


class TestDataset:
    def test_has_name(self, ds: IRT2):
        assert ds.name

    # idmap tests

    def test_vid2str_resolvable(self, ds: IRT2):
        for s in ds.idmap.vid2str.values():
            assert s in ds.idmap.str2vid

        for vid in ds.idmap.str2vid.values():
            assert vid in ds.idmap.vid2str

    def test_mid2str_resolvable(self, ds: IRT2):
        for s in ds.idmap.mid2str.values():
            assert s in ds.idmap.str2mids

        for mids in ds.idmap.str2mids.values():
            for mid in mids:
                assert mid in ds.idmap.mid2str

    def test_rid2str_resolvable(self, ds: IRT2):
        for s in ds.idmap.rid2str.values():
            assert s in ds.idmap.str2rid

        for rid in ds.idmap.str2rid.values():
            assert rid in ds.idmap.rid2str

    def test_mentions_disjoint(self, ds: IRT2):
        disjoint(
            set.union(*ds.closed_mentions.values()),
            set.union(*ds.open_mentions_val.values()),
            set.union(*ds.open_mentions_test.values()),
        )

        disjoint(
            set.union(*ds.idmap.vid2mids[Split.train].values()),
            set.union(*ds.idmap.vid2mids[Split.valid].values()),
            set.union(*ds.idmap.vid2mids[Split.test].values()),
        )

    def test_mid2vid_resolvable(self, ds: IRT2):
        for split in Split:
            for mids in ds.idmap.vid2mids[split].values():
                for mid in mids:
                    assert mid in ds.idmap.mid2vid[split]

    def _linking_resolvable(self, ds: IRT2, kgc, split):
        for (mid, rid), vids in kgc.items():
            assert mid in ds.idmap.mid2str
            assert rid in ds.idmap.rid2str
            assert mid in ds.idmap.mid2vid[split]

            for vid in vids:
                assert vid in ds.idmap.vid2str

                if split == Split.valid:
                    assert (
                        vid in ds.idmap.vid2mids[Split.train]
                        or vid in ds.idmap.vid2mids[Split.valid]
                    )

                if split == Split.test:
                    assert (
                        vid in ds.idmap.vid2mids[Split.train]
                        or vid in ds.idmap.vid2mids[Split.valid]
                        or vid in ds.idmap.vid2mids[Split.test]
                    )

    def test_open_kgc_val_resolvable(self, ds: IRT2):
        self._linking_resolvable(ds, ds.open_kgc_val_heads, Split.valid)
        self._linking_resolvable(ds, ds.open_kgc_val_tails, Split.valid)

    def test_open_kgc_test_resolvable(self, ds: IRT2):
        self._linking_resolvable(ds, ds.open_kgc_test_heads, Split.test)
        self._linking_resolvable(ds, ds.open_kgc_test_tails, Split.test)

    def _ranking_resolvable(self, ds: IRT2, ranking, split):
        for (vid, rid), mids in ranking.items():
            assert vid in ds.idmap.vid2str
            assert rid in ds.idmap.rid2str
            assert vid in ds.idmap.vid2mids[Split.train]

            for mid in mids:
                assert mid in ds.idmap.mid2str

                if split == Split.valid:
                    assert (
                        mid in ds.idmap.mid2vid[Split.train]
                        or mid in ds.idmap.mid2vid[Split.valid]
                    )

                if split == Split.test:
                    assert (
                        mid in ds.idmap.mid2vid[Split.train]
                        or mid in ds.idmap.mid2vid[Split.valid]
                        or mid in ds.idmap.mid2vid[Split.test]
                    )

    def test_open_ranking_val_resolvable(self, ds: IRT2):
        self._ranking_resolvable(ds, ds.open_ranking_val_heads, Split.valid)
        self._ranking_resolvable(ds, ds.open_ranking_val_tails, Split.valid)

    def test_open_ranking_test_resolvable(self, ds: IRT2):
        self._ranking_resolvable(ds, ds.open_ranking_test_heads, Split.test)
        self._ranking_resolvable(ds, ds.open_ranking_test_tails, Split.test)
