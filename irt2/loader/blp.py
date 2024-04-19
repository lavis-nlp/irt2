import csv
import logging
from collections import defaultdict
from dataclasses import field
from datetime import datetime
from functools import cached_property
from itertools import count
from pathlib import Path
from typing import Callable

import irt2
from irt2.dataset import IRT2, Context, Split, text_eager
from irt2.types import MID, RID, VID, Sample
from ktz.collections import buckets
from ktz.dataclasses import Builder
from ktz.filesystem import path as kpath
from pudb.theme import dataclass

log = logging.getLogger(__name__)
tee = irt2.tee(log)


SEP = "\t"


@dataclass
class IDMap:
    vid2mid: dict[VID, MID] = field(default_factory=dict)

    vid2str: dict[VID, str] = field(default_factory=dict)
    mid2str: dict[MID, str] = field(default_factory=dict)
    rid2str: dict[RID, str] = field(default_factory=dict)

    @cached_property
    def str2vid(self) -> dict[str, VID]:
        ret = {name: vid for vid, name in self.vid2str.items()}
        assert len(ret) == len(self.vid2str)
        return ret

    @cached_property
    def str2rid(self) -> dict[str, RID]:
        ret = {name: rid for rid, name in self.rid2str.items()}
        assert len(ret) == len(self.rid2str)
        return ret

    @cached_property
    def str2mid(self) -> dict[str, set[MID]]:
        ret = dict(
            buckets(
                col=self.rid2str.items(),
                key=lambda _, tup: (tup[1], tup[0]),
                mapper=set,
            )
        )

        return ret  # type: ignore TODO fix upstream


def _init_build(
    name: str,
    entity_path: Path,
    relation_path: Path,
    mention_mapper: Callable[[str], str] = str.strip,
    relation_mapper: Callable[[str], str] = str.strip,
) -> tuple[Builder, IDMap]:
    assert relation_path.parent == entity_path.parent
    path = relation_path.parent

    build = Builder(IRT2)
    build.add(path=path)

    build.add(
        config=dict(
            create=dict(
                name=name,
                seperator=SEP,
            ),
            created=datetime.now().isoformat(),
        ),
    )

    vid_gen, mid_gen, rid_gen = count(), count(), count()

    idmap = IDMap()

    # we interpret "text" as mention and "long text" as context
    with entity_path.open(mode="r") as fd:
        for vertex, mention in csv.reader(fd, delimiter=SEP):
            vid = next(vid_gen)
            mid = next(mid_gen)

            idmap.vid2str[vid] = vertex.strip()
            idmap.mid2str[mid] = mention_mapper(mention)
            idmap.vid2mid[vid] = mid

    with relation_path.open(mode="r") as fd:
        for relation in fd:
            idmap.rid2str[next(rid_gen)] = relation_mapper(relation)

    build.add(
        vertices=idmap.vid2str,
        relations=idmap.rid2str,
        mentions=idmap.mid2str,
    )

    return build, idmap


def _load_train_graph(
    path: Path,
    build: Builder,
    idmap: IDMap,
):
    with path.open(mode="r") as fd:
        vids, closed_triples = set(), set()

        mapped = (
            (idmap.str2vid[h], idmap.str2vid[t], idmap.str2rid[r])
            for h, r, t in csv.reader(fd, delimiter=SEP)
        )

        for h, r, t in mapped:
            closed_triples.add((h, r, t))
            vids |= {h, t}

    closed_mentions = {vid: {idmap.vid2mid[vid]} for vid in vids}

    build.add(
        closed_triples=closed_triples,
        closed_mentions=closed_mentions,
    )


def _load_ow(
    p_valid: Path,
    test_path: Path,
    build: Builder,
    idmap: IDMap,
) -> dict[VID, Split]:
    def load_ow(
        fpath: Path,
    ) -> tuple[set[Sample], set[Sample], dict[VID, set[MID]]]:
        heads, tails, mapping = set(), set(), defaultdict(set)
        with fpath.open(mode="r") as fd:
            for h, r, t in csv.reader(fd, delimiter=SEP):
                # add both directions
                h_vid, t_vid, rid = idmap.str2vid[h], idmap.str2vid[t], idmap.str2rid[r]
                h_mid, t_mid = idmap.vid2mid[h_vid], idmap.vid2mid[t_vid]

                heads.add((h_mid, rid, t_vid))
                tails.add((t_mid, rid, h_vid))

                mapping[h_vid].add(h_mid)
                mapping[t_vid].add(t_mid)

        return heads, tails, mapping

    val_heads, val_tails, val_mentions = load_ow(p_valid)
    build.add(
        _open_val_heads=val_heads,
        _open_val_tails=val_tails,
        open_mentions_val=val_mentions,
    )

    test_heads, test_tails, test_mentions = load_ow(test_path)
    build.add(
        _open_test_heads=test_heads,
        _open_test_tails=test_tails,
        open_mentions_test=test_mentions,
    )

    misses = 0
    vid2split: dict[VID, Split] = {}
    for vid in idmap.vid2str:
        split = None
        if vid in build.get("closed_mentions"):
            split = Split.train
        elif vid in val_mentions:
            split = Split.valid
        elif vid in test_mentions:
            split = Split.test
        else:
            misses += 1
            # raise KeyError(f"not found: {vid} ({idmap.vid2str[vid]})")

        if split is not None:
            vid2split[vid] = split

    if misses:
        tee("could not assign {misses} vids for vid2split!")

    return vid2split


def _load_graphs(
    paths: tuple[Path, Path, Path],
    build: Builder,
    idmap: IDMap,
):
    p_trian, p_valid, p_test = paths

    _load_train_graph(
        path=p_trian,
        build=build,
        idmap=idmap,
    )

    vid2split = _load_ow(
        p_valid=p_valid,
        test_path=p_test,
        build=build,
        idmap=idmap,
    )

    return vid2split


def _load_text_eager(
    path: Path,
    build: Builder,
    idmap: IDMap,
    vid2split: dict[VID, Split],
):
    ctxs: dict[Split, list[Context]] = {split: [] for split in Split}

    with path.open(mode="r") as fd:
        misses = 0
        for vertex, text in csv.reader(fd, delimiter=SEP):
            vid = idmap.str2vid[vertex]
            mid = idmap.vid2mid[vid]

            if vid not in vid2split:
                misses += 1
                continue

            ctx = Context(
                mid=mid,
                mention=idmap.mid2str[mid],
                origin="",
                data=text,
            )

            ctxs[vid2split[vid]].append(ctx)

    tee("could not assign {len(misses)} texts to splits")
    build.add(_text_loader=text_eager(mapping=ctxs))


# ---


def load_fb15k237(folder: str | Path) -> IRT2:
    """Load FB16K237 as provided by BLP"""
    path = kpath(folder, is_dir=True)
    build, idmap = _init_build(
        name="FB15K237 (BLP)",
        entity_path=path / "entity2text.txt",
        relation_path=path / "relations.txt",
    )

    vid2split = _load_graphs(
        paths=(
            path / "ind-train.tsv",
            path / "ind-dev.tsv",
            path / "ind-test.tsv",
        ),
        build=build,
        idmap=idmap,
    )

    _load_text_eager(
        path=path / "entity2textlong.txt",
        build=build,
        idmap=idmap,
        vid2split=vid2split,
    )

    return build()


def load_wn18rr(folder: str | Path) -> IRT2:
    """Load WN18RR as provided by BLP.

    All data uses tabstop as seperator. There is no long text
    description for entities. All entity descriptions have their
    mention named as the first part of the description before the
    first comma:

    14854262        stool, solid excretory product evacuated from the bowels

    So we take the left side as mention and the full text as
    description.

    """
    path = kpath(folder, is_dir=True)
    build, idmap = _init_build(
        name="WN18RR (BLP)",
        entity_path=path / "entity2text.txt",
        relation_path=path / "relations.txt",
        mention_mapper=lambda s: s.strip().split(",", maxsplit=1)[0],
    )

    vid2split = _load_graphs(
        paths=(
            path / "ind-train.tsv",
            path / "ind-dev.tsv",
            path / "ind-test.tsv",
        ),
        build=build,
        idmap=idmap,
    )

    _load_text_eager(
        path=path / "entity2text.txt",
        build=build,
        idmap=idmap,
        vid2split=vid2split,
    )

    return build()


def load_umls(folder: str | Path) -> IRT2:
    """Load UMLS as provided by BLP.

    All data uses tabstop as seperator.

    train.tsv, dev.tsv, test.tsv:
    -----------------------------
    acquired_abnormality    location_of     experimental_model_of_disease
    anatomical_abnormality  manifestation_of        physiologic_function

    entities.txt, relations.txt:
    ----------------------------
    idea_or_concept
    virus

    entity2text.txt, entity2textlong.txt, relation2text.txt
    -------------------------------------------------------
    idea_or_concept idea or concept
    virus   virus
    """

    # -- training data

    path = kpath(folder, is_dir=True)
    build, idmap = _init_build(
        name="UMLS (BLP)",
        entity_path=path / "entity2text.txt",
        relation_path=path / "relations.txt",
    )

    vid2split = _load_graphs(
        paths=(
            path / "train.tsv",
            path / "dev.tsv",
            path / "test.tsv",
        ),
        build=build,
        idmap=idmap,
    )

    _load_text_eager(
        path=path / "entity2textlong.txt",
        build=build,
        idmap=idmap,
        vid2split=vid2split,
    )

    return build()
