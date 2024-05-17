import csv
import logging
import string
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Callable

import irt2
from irt2.dataset import IRT2, Split
from irt2.loader.irt import text_eager
from irt2.types import RID, VID, Context, ContextGenerator, IDMap, Sample
from ktz.dataclasses import Builder
from ktz.filesystem import path as kpath

log = logging.getLogger(__name__)
tee = irt2.tee(log)


SEP = "\t"


def _norm_str(s: str) -> str:
    return s.strip().lower()


def _init_build(
    name: str,
    entity_path: Path,
    relation_path: Path,
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

    vid_gen, rid_gen = count(), count()

    idmap = IDMap()
    build.add(idmap=idmap)

    with entity_path.open(mode="r") as fd:
        for entity in fd:
            idmap.vid2str[next(vid_gen)] = entity.strip()

    with relation_path.open(mode="r") as fd:
        for relation in fd:
            idmap.rid2str[next(rid_gen)] = relation.strip()

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
            # only happens with Wikidata5m
            if h in idmap.str2vid and t in idmap.str2vid
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
            # total, notfound = 0, 0
            for h, r, t in csv.reader(fd, delimiter=SEP):
                # add both directions
                h_vid, t_vid, rid = idmap.str2vid[h], idmap.str2vid[t], idmap.str2rid[r]
                idmap.split2vids[split] |= {h_vid, t_vid}

                # although there are multiple mids per vertex for wikidata5m
                # we only add a single instance to the task set to conform
                # to the evaluation protocol of daza et al.

                h_mid = list(idmap.vid2mids[h_vid])[0]
                heads.add((h_mid, rid, t_vid))
                idmap.vid2mids[h_vid].add(h_mid)

                t_mid = list(idmap.vid2mids[t_vid])[0]
                tails.add((t_mid, rid, h_vid))
                idmap.vid2mids[t_vid].add(t_mid)

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
    p_train, p_valid, p_test = paths

    _load_train_graph(
        path=p_train,
        build=build,
        idmap=idmap,
    )

    _load_ow(
        p_valid=p_valid,
        p_test=p_test,
        build=build,
        idmap=idmap,
    )


def _load_text_lazy(
    path: Path,
    idmap: IDMap,
):
    # we require a copy of the idmap with the original eids
    # before remapping their names
    eid2vid = idmap.str2vid.copy()

    def contexts(fd, split: Split):
        for line in fd:
            eid, *rest = line.split("\t", maxsplit=1)
            text = " ".join(map(str.strip, rest))

            # skip unassociated texts
            if eid not in eid2vid:
                continue

            vid = eid2vid[eid]

            # skip text not used for the current split
            if vid not in idmap.split2vids[split]:
                continue

            mid = list(idmap.vid2mids[vid])[0]

            yield Context(
                mid=mid,
                mention=idmap.mid2str[mid],
                origin=eid,
                data=text,
            )

    def wrapped(split: Split) -> ContextGenerator:
        with open(path, mode="r") as fd:
            yield contexts(fd, split)

    return wrapped


UNK = "unknown"


def _load_mentions(
    idmap: IDMap,
    entity_path: Path,
    mention_mapper: Callable[[str], str] = _norm_str,
):
    # vertices and mentions

    mid_gen = count()

    def _add_mention(vid: VID, mention: str):
        mid = next(mid_gen)
        idmap.mid2str[mid] = mention
        idmap.vid2mids[vid].add(mid)

    seen, misses = set(), 0
    with entity_path.open(mode="r") as fd:
        for eid, *mentions in csv.reader(fd, delimiter=SEP):
            if eid not in idmap.str2vid:
                misses += 1
                continue

            vid = idmap.str2vid[eid]
            seen.add(vid)

            mentions = [mention_mapper(mention) for mention in mentions]
            for mention in mentions:
                _add_mention(vid, mention)

    tee(f"a total of {misses} entities' mentions have been skipped")

    unseen = set(idmap.vid2str) - seen
    if unseen:
        tee(f"a total of {len(unseen)} vertices had no mention assigned")

    for vid in unseen:
        _add_mention(vid, UNK)


def _remap_names(
    idmap: IDMap,
    e2text_path: Path,
    r2text_path: Path,
    entity_mapper: Callable[[str], str] = _norm_str,
    relation_mapper: Callable[[str], str] = _norm_str,
):
    def _remap(
        path: Path,
        target: dict[VID | RID, str],
        lookup: dict[str, VID | RID],
        mapper: Callable[[str], str],
    ):
        seen = set()
        with path.open(mode="r") as fd:
            for line in fd:
                eid, text = line.split(SEP, maxsplit=1)

                if eid not in lookup:
                    continue

                irt_id = lookup[eid]
                seen.add(irt_id)

                name = string.capwords(mapper(text))
                target[irt_id] = f"{eid}:{name}"

        unseen = set(target) - seen
        if unseen:
            tee(f"remapping {len(unseen)} names unseen in {path.name}")

        for irt_id in unseen:
            target[irt_id] = f"{target[irt_id]}:{UNK}"

    _remap(
        path=e2text_path,
        target=idmap.vid2str,
        lookup=idmap.str2vid,
        mapper=entity_mapper,
    )

    _remap(
        path=r2text_path,
        target=idmap.rid2str,
        lookup=idmap.str2rid,
        mapper=relation_mapper,
    )

    # invalidate cache
    del idmap.str2vid
    del idmap.str2rid


def _load_generic(
    folder: str | Path,
    name: str,
    train_file: str = "ind-train.tsv",
    valid_file: str = "ind-dev.tsv",
    test_file: str = "ind-test.tsv",
    text_file: str = "entity2textlong.txt",
    mention_mapper: Callable[[str], str] = _norm_str,
) -> IRT2:
    path = kpath(folder, is_dir=True)

    build, idmap = _init_build(
        name=name,
        relation_path=path / "relations.txt",
        entity_path=path / "entities.txt",
    )

    _load_mentions(
        idmap,
        entity_path=path / "entity2text.txt",
        mention_mapper=mention_mapper,
    )

    _load_graphs(
        paths=(
            path / train_file,
            path / valid_file,
            path / test_file,
        ),
        build=build,
        idmap=idmap,
    )

    build.add(
        _text_loader=_load_text_lazy(
            path=path / text_file,
            idmap=idmap,
        )
    )

    # this remaps all vertex and relation strings which makes it no
    # longer possible to address the original data - hence it comes
    # last

    _remap_names(
        idmap=idmap,
        e2text_path=path / "entity2text.txt",
        r2text_path=path / "relation2text.txt",
        entity_mapper=mention_mapper,
    )

    return build()


def load_wikidata5m(folder: str | Path) -> IRT2:
    """Load FB16K237 as provided by BLP.

    All data uses tabstop as seperator. Format follows that of UMLS
    etc. There are multiple mentions per vertex, however.
    """

    # there are 5091 entities without a mention: we discard them
    return _load_generic(folder, "BLP/WIKIDATA5M")


def load_fb15k237(folder: str | Path) -> IRT2:
    """Load FB16K237 as provided by BLP.

    All data uses tabstop as seperator. Format follows that of UMLS
    etc.
    """
    return _load_generic(folder, "BLP/FB15K237")


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
    return _load_generic(
        folder,
        "BLP/UMLS",
        train_file="train.tsv",
        valid_file="dev.tsv",
        test_file="test.tsv",
    )


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
    return _load_generic(
        folder,
        "BLP/WN18RR",
        text_file="entity2text.txt",
        mention_mapper=lambda s: _norm_str(s).split(",", maxsplit=1)[0],
    )
