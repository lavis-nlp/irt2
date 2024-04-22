import csv
import logging
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Callable

from ktz.dataclasses import Builder
from ktz.filesystem import path as kpath

import irt2
from irt2.dataset import IRT2, Split
from irt2.loader.irt2 import text_eager
from irt2.types import Context, IDMap, Sample

log = logging.getLogger(__name__)
tee = irt2.tee(log)


SEP = "\t"


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
    build.add(idmap=idmap)

    # we interpret "text" as mention and "long text" as context
    with entity_path.open(mode="r") as fd:
        for vertex, mention in csv.reader(fd, delimiter=SEP):
            vid = next(vid_gen)
            mid = next(mid_gen)

            idmap.vid2str[vid] = vertex.strip()
            idmap.mid2str[mid] = mention_mapper(mention)
            idmap.vid2mids[vid] = {mid}

    with relation_path.open(mode="r") as fd:
        for relation in fd:
            idmap.rid2str[next(rid_gen)] = relation_mapper(relation)

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
            vids |= {h, t}
            closed_triples.add((h, r, t))

    idmap.split2vids[Split.train] = vids
    build.add(closed_triples=closed_triples)


def _load_ow(
    p_valid: Path,
    p_test: Path,
    build: Builder,
    idmap: IDMap,
):
    def load_ow(fpath: Path, split: Split) -> tuple[set[Sample], set[Sample]]:
        heads, tails = set(), set()
        with fpath.open(mode="r") as fd:
            for h, r, t in csv.reader(fd, delimiter=SEP):
                # add both directions
                h_vid, t_vid, rid = idmap.str2vid[h], idmap.str2vid[t], idmap.str2rid[r]

                # assert that there is a 1:1 mapping from vid to mid
                assert len(idmap.vid2mids[h_vid]) == 1
                assert len(idmap.vid2mids[t_vid]) == 1

                h_mid, t_mid = (
                    list(idmap.vid2mids[h_vid])[0],
                    list(idmap.vid2mids[t_vid])[0],
                )

                heads.add((h_mid, rid, t_vid))
                tails.add((t_mid, rid, h_vid))

                idmap.vid2mids[h_vid].add(h_mid)
                idmap.split2vids[split] |= {h_vid, t_vid}

        return heads, tails

    val_heads, val_tails = load_ow(p_valid, Split.valid)
    build.add(
        _val_heads=val_heads,
        _val_tails=val_tails,
    )

    test_heads, test_tails = load_ow(p_test, Split.test)
    build.add(
        _test_heads=test_heads,
        _test_tails=test_tails,
    )


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

    _load_ow(
        p_valid=p_valid,
        p_test=p_test,
        build=build,
        idmap=idmap,
    )


def _load_text_eager(
    path: Path,
    build: Builder,
    idmap: IDMap,
):
    ctxs: dict[Split, list[Context]] = {split: [] for split in Split}

    with path.open(mode="r") as fd:
        misses = 0
        for vertex, text in csv.reader(fd, delimiter=SEP):
            vid = idmap.str2vid[vertex]

            assert len(idmap.vid2mids[vid]) == 1
            mid = list(idmap.vid2mids[vid])[0]

            splits = {split for split in Split if vid in idmap.split2vids[split]}

            if not splits:
                misses += 1
                continue

            ctx = Context(
                mid=mid,
                mention=idmap.mid2str[mid],
                origin="",
                data=text,
            )

            for split in splits:
                ctxs[split].append(ctx)

        tee(f"could not assign {misses} texts to splits")
    build.add(_text_loader=text_eager(mapping=ctxs))


# ---


def load_fb15k237(folder: str | Path) -> IRT2:
    """Load FB16K237 as provided by BLP.

    All data uses tabstop as seperator. Format follows that of UMLS
    etc.
    """
    path = kpath(folder, is_dir=True)
    build, idmap = _init_build(
        name="FB15K237 (BLP)",
        entity_path=path / "entity2text.txt",
        relation_path=path / "relations.txt",
    )

    _load_graphs(
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

    _load_graphs(
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

    _load_graphs(
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
    )

    return build()
