# -*- coding: utf-8 -*-

"""Contains the API for ipynb/create.ipynb."""

import irt2
from irt2.dataset import IRT2
from irt2.graph import Graph
from irt2.graph import Relation
from irt2.types import VID, RID, EID, MID, Triple, Mention

from ktz.dataclasses import Index
from ktz.dataclasses import Builder
from ktz.collections import buckets
from ktz.collections import unbucket
from ktz.collections import Incrementer
from ktz.filesystem import path as kpath
from ktz.string import encode_line as encode
from ktz.string import decode_line as decode

from tabulate import tabulate

import re
import csv
import gzip
import yaml
import random
import logging
import numpy as np

from pathlib import Path
from itertools import islice
from itertools import combinations
from datetime import datetime
from contextlib import ExitStack
from dataclasses import dataclass
from collections import Counter
from collections import defaultdict
from functools import partial
from functools import cached_property

from typing import Literal
from typing import Optional
from collections.abc import Iterable


log = logging.getLogger(__name__)


def norm(s: str):
    """
    Remove some special characters and lowercase.

    This functions is generally used to normalize mentions (who would
    otherwise be trivially matched).

    Parameters
    ----------
    s : str
        Input string

    """
    return " ".join(re.sub(r"[-_()\[\].]", " ", s.lower()).split())


@dataclass(frozen=True)
class Match:
    """
    Single mention plus context.

    Each entity is identified by an entity id (eid). This eid stems
    from whatever upstream data is used (e.g. for CodEx these are the
    Wikidata ids). The match denotes a (unique) page id whence it was
    found and character positions for the mention's start and end
    tokens. The idea is to offer the possibility to sample larger
    contexts (and not single sentences) if desired.
    """

    page: str
    eid: EID  # entity id (e.g. Wikidata ID for cde)

    entity: str
    mention: str
    norm_mention: str  # see norm()

    start: int
    end: int

    @classmethod
    def from_csv(cls, line: Iterable[str]):
        """Load data from a csv file.

        Parameters
        ----------
        Match : Match
            Match constructor
        line : Iterable[str]
            Source data

        """
        page, eid, entity, mention, start, end = line
        start, end = int(start), int(end)

        return cls(
            page=page,
            eid=eid,
            entity=entity,
            mention=mention,
            norm_mention=norm(mention),
            start=start,
            end=end,
        )


# TODO use database connection for both matches/pages
# (see bottom where text sampling takes place)
def index_matches(path: Path, n: Optional[int] = None):
    """
    Create an index of Match objects.

    Parameters
    ----------
    path : Path
        Source csv file
    n : int
        Limit to first n objects
    """
    path = kpath(path, is_file=True)
    index = Index(Match, includes={"page", "eid"})

    log.info(f"creating match index from {path}...")
    with path.open(mode="r") as fd:
        reader = islice(csv.reader(fd), n)

        matches = (Match.from_csv(row) for row in reader)
        filtered = filter(lambda m: m.norm_mention, matches)

        index.add(filtered)

    return index


@dataclass
class Mentions:
    """Mentions extracted from Matches."""

    eid2mentions: dict[EID, dict[Mention, int]]
    norm2mentions: tuple[dict[str, str]]


def get_mentions(
    index: Index[Match],
    prune: int,
) -> Mentions:
    """
    Prune, map and count mentions.

    Parameters
    ----------
    index : Index[Match]
        Match index
    prune : int
        At least N matched contexts required

    Returns
    -------
    tuple[dict[str, str], dict[EID, dict[Mention, int]]]

    """
    norm2mentions = defaultdict(set)
    eid2mentions = defaultdict(Counter)

    log.info(f"extracting mentions from {len(index.flat)} matches")

    for match in index.flat:
        eid2mentions[match.eid][match.norm_mention] += 1
        norm2mentions[match.norm_mention].add(match.mention)

    log.info(f"pruning mentions to have at least {prune} matches")

    eid2mentions = {
        eid: {mention: count for mention, count in counts.items() if count >= prune}
        for eid, counts in eid2mentions.items()
        if len(counts) > 0
    }

    log.info(f"retained mentions for {len(eid2mentions)} entities")

    return Mentions(
        eid2mentions=eid2mentions,
        norm2mentions=dict(norm2mentions),
    )


#
#  mention split
#

KINDS = "closed/training", "open/validation", "open/test"
CLOSED_WORLD, *OPEN_WORLD = KINDS
OPEN_VALIDATION, OPEN_TEST = OPEN_WORLD

Flat = set[tuple[VID, Mention]]
Kind = Literal[KINDS]


# used in Split.create


def _split_create_relations(
    graph: Graph,
    include_rels: list[str],
    exclude_rels: list[str],
):
    relations = Relation.from_graph(graph)

    includes = set(include_rels)
    assert len(includes) == len(include_rels)

    excludes = set(exclude_rels)
    assert len(excludes) == len(exclude_rels)

    if includes:
        relations = [rel for rel in relations if rel.name in includes]

    elif excludes:
        relations = [rel for rel in relations if rel.name not in excludes]

    return relations


def _split_create_concepts(
    graph: Graph,
    mentions: Mentions,
    relations: list[Relation],
    concept_rels: set[str],
):

    stats = Counter()
    removed = set()

    # map to upstream
    # eid=Q108946:A Few Good Men -> Q108946
    mapped = {(vid, eid, eid.split(":")[0]) for vid, eid in graph.source.ents.items()}
    mapping = mentions.eid2mentions

    # first pass, identify vids without mentions

    for vid, eid, link in mapped:
        if link not in mapping or not mapping.get(link, {}):
            stats["no mention"] += 1
            removed.add(vid)

    # all concept vids
    concept_vids = {
        vid for rel in relations if rel.name in concept_rels for vid in rel.concepts
    } - removed

    retained_vids = {
        vid for rel in relations for vid in rel.heads | rel.tails
    } - removed

    retained_rids = {rel.rid for rel in relations}

    # target data:
    concepts: Flat = set()
    candidates: Flat = set()

    # second pass: select and transform vid -> {(vid, mention), ...}

    for vid, eid, link in mapped:

        # not part of any retained relation
        if vid not in retained_vids:
            stats["not retained"] += 1
            removed.add(vid)
            continue

        # not part of any triple anymore
        if not len(
            {
                (h, t, r)
                for h, t, r in graph.find(heads={vid}, tails={vid})
                if h in retained_vids and t in retained_vids and r in retained_rids
            }
        ):
            stats["no triples"] += 1
            removed.add(vid)
            continue

        # from this point on, a "sample" is a unique
        # combination of vid and mention string
        flat = {(vid, mention) for mention in mentions.eid2mentions[link].keys()}
        assert flat

        # add sample to either concepts or split-candidates
        target = concepts if vid in concept_vids else candidates
        target |= flat

    total = set(graph.source.ents)
    log.info(f"retaining {len(total - removed)}/{len(total)} vertices")
    log.info(", ".join(": ".join(map(str, t)) for t in stats.items()))

    # trace

    # _vids = (
    #     {vid for vid, _ in concepts},
    #     {vid for vid, _ in candidates},
    #     removed,
    # )

    # for _v in {8231, 8473, 9057, 15837, 16764}:
    #     print(_v, _v in _vids[0], _v in _vids[1], _v in _vids[2])

    # /trace

    return concepts, candidates, removed


def _split_create_cwow_uniform(
    seed: int,
    concepts: set[Flat],
    candidates: set[Flat],
    ratio_train: float,
    relations: list[Relation],
    vid2rels: dict[str, dict[VID, set[RID]]],
):
    # draw the threshold between all mentions and then set aside the
    # open world mentions. the remaining mentions can be added to the
    # concept mentions and form the closed world mention set.

    total = len(concepts) + len(candidates)
    threshold = int((ratio_train) * total) - len(concepts)

    candidates = sorted(candidates)
    random.shuffle(candidates)

    log.info(f"set aside {len(concepts)} concept mentions")
    assert 0 <= threshold, "threshold too small"

    cw = concepts | set(candidates[:threshold])
    ow = set(candidates[threshold:])

    log.info(f"create initial open-world split at {len(cw)}/{total}")
    return cw, ow


def _split_create_cwow_weighted(
    seed: int,
    concepts: set[Flat],
    candidates: set[Flat],
    ratio_train: float,
    relations: list[Relation],
    vid2rels: dict[str, dict[VID, set[RID]]],
):

    # draw the threshold between all mentions and then set aside the
    # open world mentions. the remaining mentions can be added to the
    # concept mentions and form the closed world mention set.

    total = len(concepts) + len(candidates)
    threshold = int((ratio_train) * total) - len(concepts)

    # candidate selection is biased towards underrepresented
    # relations as otherwise they may not survive when splitting.
    # we use the reciprocal of the smallest total amount of involved
    # vertices of the respective vertex

    rids = {rel.rid for rel in relations}
    sizes = {rel.rid: len(rel.heads | rel.tails) for rel in relations}

    vid2reciprocal = {
        vid: 1 / min(sizes[rid] for rid in sub if rid in rids)
        for vid2relset in vid2rels.values()
        for vid, sub in vid2relset.items()
    }

    lis = sorted(candidates)
    weights = np.array([vid2reciprocal[vid] for vid, _ in lis])

    # create selection indexes and add them to the concepts

    rng = np.random.default_rng(seed)
    idxs = rng.choice(
        a=len(lis),
        size=threshold,
        p=weights / weights.sum(),
        replace=False,
    )

    cw = concepts | {lis[idx] for idx in idxs}
    ow = candidates - cw

    assert len(cw) and len(ow), f"{len(cw)=} and {len(ow)=}"
    return cw, ow


def _split_create_prune(
    cw: set[Flat],
    ow: set[Flat],
    concepts: set[Flat],
    relations: list[Relation],
    vid2rels: dict[str, dict[VID, set[RID]]],
    prune: int,
):
    log.info(f"pruning closed world to contain at most {prune} mentions per relation")
    log.info(f"before: {len(cw)=} and {len(ow)=}")

    concept_vids = set(vid for vid, _ in concepts)

    # counting the retained mentions per relation and direction

    counts_head, counts_tail = Counter(), Counter()
    stats = Counter()

    # take all closed-world mentions as candidates
    # to be redistributed

    candidates = sorted(cw)
    random.shuffle(candidates)

    while candidates:
        vid, _ = flat = candidates.pop()
        stats["total"] += 1

        # concept entities are never put in the
        # open-world split and may violate the prune
        # parameter (this is, for cde and the xs configuration
        # seldom the case anyway) - revisit this if larger
        # graphs need to be sub-sampled
        if vid in concept_vids:
            stats["retained concept"] += 1
            continue

        head_rels, tail_rels = vid2rels["heads"][vid], vid2rels["tails"][vid]

        # create a flat collection for convenience
        gen = (head_rels, counts_head), (tail_rels, counts_tail)
        gen = [(rel, counter) for rels, counter in gen for rel in rels]

        # check whether retaining the entity in the closed-world
        # part violates the pruning parameter for any of the relations

        assert gen, "vid not part of any relation"
        if not any(counter[rel] >= prune for rel, counter in gen):

            for rel, counter in gen:
                counter[rel] += 1

            # leave candidate in closed-world
            stats["retained"] += 1
            continue

        # move candidate to open-world
        stats["pruned"] += 1
        cw.remove(flat)
        ow.add(flat)

    log.info(" ".join(f"{name}: {count}" for name, count in stats.items()))
    log.info(f"after: {len(cw)=} and {len(ow)=}")
    return cw, ow


def _split_ow_val_test(cw, ow, ratio_val):
    ow = sorted(ow)
    random.shuffle(ow)

    threshold = int(ratio_val * len(ow))
    train, valid, test = map(set, [cw, ow[:threshold], ow[threshold:]])

    log.info(f"split ow with {ratio_val:.3f} at {threshold}")
    log.info(f" train={len(train)} valid={len(valid)} test={len(test)}")

    return train, valid, test


# --


@dataclass
class Split:
    """Open/Closed-world mention split."""

    graph: Graph
    relations: set[RID]  # only the retained ones

    concept_vertices: set[VID]
    removed_vertices: set[VID]

    mentions: dict[Kind, Flat]

    @cached_property
    def description(self) -> str:
        """Return table with key figures."""

        def _open_world_row(kind):

            head_triples = self.open_world_head_triples[kind]
            head_vertices = {h for h, _, _ in head_triples}

            tail_triples = self.open_world_tail_triples[kind]
            tail_vertices = {t for _, t, _ in tail_triples}

            both = head_vertices | tail_vertices
            mentions = self.mentions[kind]

            return (
                (f"{kind}", None, None, None),
                (
                    "heads",
                    len([vid for vid, _ in mentions if vid in head_vertices]),
                    len(head_vertices),
                    len(head_triples),
                ),
                (
                    "tails",
                    len([vid for vid, _ in mentions if vid in tail_vertices]),
                    len(tail_vertices),
                    len(tail_triples),
                ),
                (
                    "both",
                    len([vid for vid, _ in mentions if vid in both]),
                    len(both),
                    len(head_triples | tail_triples),
                ),
            )

        rows = (
            (
                "concept",
                len(self.concept_mentions),
                len(self.concept_vertices),
                len(self.concept_triples),
            ),
            (
                "closed world (training)",
                len(self.mentions[CLOSED_WORLD]),
                len(self.vertices[CLOSED_WORLD]),
                len(self.triples[CLOSED_WORLD]),
            ),
        )

        return tabulate(
            rows + _open_world_row(OPEN_VALIDATION) + _open_world_row(OPEN_TEST),
            headers=("", "mentions", "vertices", "triples"),
        )

    @cached_property
    def concept_mentions(self) -> Flat:
        """Return all mentions of concept vertices."""
        return {
            (vid, mention)
            for vid, mention in self.mentions[CLOSED_WORLD]
            if vid in self.concept_vertices
        }

    # vertex split

    @cached_property
    def vertices(self) -> dict[Kind, set[VID]]:
        """Return vertices involved in each split."""
        return {k: {vid for vid, _ in self.mentions[k]} for k in KINDS}

    @cached_property
    def open_world_vertices(self) -> dict[Kind, set[VID]]:
        """Return unseen vertices involved in each split."""
        ret = {k: v.copy() for k, v in self.vertices.items()}

        ret[OPEN_VALIDATION] -= ret[CLOSED_WORLD]
        ret[OPEN_TEST] -= ret[CLOSED_WORLD] | ret[OPEN_VALIDATION]

        return ret

    # triple split

    def _filter_triples(self, triples: set[Triple]):
        """Remove all triples with removed vertices."""
        # there may be vertices without any mentions
        retained = {vid for vids in self.vertices.values() for vid in vids}

        return {
            (h, t, r)
            for h, t, r in triples
            if h in retained and t in retained and r in self.relations
        }

    @cached_property
    def triples(self) -> dict[Kind, Triple]:
        """Return triple sets."""
        closed = {
            CLOSED_WORLD: self._filter_triples(
                self.graph.select(
                    heads=self.vertices[CLOSED_WORLD],
                    tails=self.vertices[CLOSED_WORLD],
                )
            )
        }

        open = {
            k: self.open_world_head_triples[k] | self.open_world_tail_triples[k]
            for k in OPEN_WORLD
        }

        return closed | open

    @cached_property
    def concept_triples(self) -> set[Triple]:
        """Return all triples where both vertices are concepts."""
        return self._filter_triples(
            self.graph.select(
                heads=self.concept_vertices,
                tails=self.concept_vertices,
            )
        )

    @cached_property
    def open_world_head_triples(self) -> dict[Kind, set[Triple]]:
        """Return triples where the head vertex is open-world."""
        return {
            k: self._filter_triples(
                self.graph.find(heads=self.vertices[k]),
            )
            for k in OPEN_WORLD
        }

    @cached_property
    def open_world_tail_triples(self) -> dict[Kind, set[Triple]]:
        """Return triples where the tail vertex is open-world."""
        return {
            k: self._filter_triples(
                self.graph.find(tails=self.vertices[k]),
            )
            for k in OPEN_WORLD
        }

    # --  tests

    def _check_mentions(self, test):
        # the mention split is disjoint
        for kind1, kind2 in combinations(KINDS, r=2):
            test(
                lambda shared: not shared,
                f"{{shared}} mentions shared between {kind1} and {kind2}",
                shared=len(self.mentions[kind1] & self.mentions[kind2]),
            )

        # every mention is associated with a vertex and for each vertex
        # at least one head or tail task exists
        for kind in OPEN_WORLD:
            heads = {head for head, _, _ in self.open_world_head_triples[kind]}
            tails = {tail for _, tail, _ in self.open_world_tail_triples[kind]}
            mentions = {vid for vid, _ in self.mentions[kind]}

            for x in mentions - (heads | tails):
                print("!!", x)

            test(
                lambda excess, kind: excess == 0,
                "{kind}: {excess} vertices of mentions have no task triples",
                excess=len(mentions - (heads | tails)),
                kind=kind,
            )

    def _check_vertices(self, test):

        test(
            lambda vs, owvs: vs == owvs,
            "vertex mismatch",
            vs=self.vertices[CLOSED_WORLD],
            owvs=self.open_world_vertices[CLOSED_WORLD],
        )

        test(
            lambda vertices, owv: vertices == owv,
            "vertex set mismatch",
            vertices=set.union(*self.vertices.values()),
            owv=set.union(*self.open_world_vertices.values()),
        )

        # concept vertices are a subset of the closed-world vertex set
        test(
            lambda concepts, closed: concepts <= closed,
            "concept vertices are not a subset of closed-world vertices",
            concepts=self.concept_vertices,
            closed=self.vertices[CLOSED_WORLD],
        )

        # open-world vertices are not shared between splits
        for kind1, kind2 in combinations(KINDS, r=2):
            shared = self.open_world_vertices[kind1] & self.open_world_vertices[kind2]

            test(
                lambda shared: shared == 0,
                f"{{shared}} vertices shared between {kind1} and {kind2}",
                shared=len(shared),
            )

    def _check_triples(self, test):
        # note: the triple split may not be disjoint!
        # the same triple spawns multiple tasks for each
        # of its mentions!

        for kind in KINDS:
            test(
                bool,
                f"there are not triples for {kind}",
                triples=len(self.triples[kind]),
            )

        test(
            lambda concepts, closed: concepts <= closed,
            "concept triples are no subset of closed world",
            concepts=self.concept_triples,
            closed=self.triples[CLOSED_WORLD],
        )

        for rid in self.relations - {r for h, t, r in self.triples[CLOSED_WORLD]}:
            print(rid, self.graph.source.rels[rid])

        test(
            lambda found, retained: found == retained,
            "found {found} relations in closed world triples, expecting {retained}",
            found=len({r for h, t, r in self.triples[CLOSED_WORLD]}),
            retained=len(self.relations),
        )

    def check(self):
        """Run a self-check."""

        def test(cond, msg, **kwargs):
            if not cond(*kwargs.values()):
                raise Exception(msg.format(**kwargs))

        self._check_mentions(test)
        self._check_vertices(test)
        self._check_triples(test)

    @classmethod
    def create(
        Split,
        graph: Graph,
        mentions: Mentions,
        seed: int,
        ratio_train: float,
        ratio_val: float,
        concept_rels: list[str],  # manually selected concept relations
        include_rels: list[str],  # upstream names (eids)
        exclude_rels: list[str],  # upstream names (eids)
        prune: Optional[int] = None,
        sampling: Literal["uniform", "weighted reciprocal"] = "uniform",
    ):
        """Divide mentions by a given ratio and concepts."""
        assert not (bool(include_rels) and bool(exclude_rels)), "mutex!"

        log.info(f"setting seed to {seed}")
        random.seed(seed)

        # configure split

        build = Builder(Split)
        build.add(graph=graph)

        # select relations

        relations = _split_create_relations(
            graph=graph,
            include_rels=include_rels,
            exclude_rels=exclude_rels,
        )

        assert relations
        build.add(relations={rel.rid for rel in relations})

        # identify concepts

        concepts, candidates, removed = _split_create_concepts(
            graph=graph,
            mentions=mentions,
            relations=relations,
            concept_rels=concept_rels,
        )
        build.add(removed_vertices=removed)

        #

        # for each vertex, we'd like to know in which relations it occurs

        vid2rels = dict(heads=defaultdict(set), tails=defaultdict(set))
        for rel in relations:

            for vid in rel.heads:
                vid2rels["heads"][vid].add(rel.rid)

            for vid in rel.tails:
                vid2rels["tails"][vid].add(rel.rid)

        # create initial open/closed-world split
        log.info(f"sampling {sampling} for closed/open-world")

        sampler = {
            "uniform": _split_create_cwow_uniform,
            "weighted reciprocal": _split_create_cwow_weighted,
        }

        cw, ow = sampler[sampling](
            seed=seed,
            concepts=concepts,
            candidates=candidates,
            ratio_train=ratio_train,
            relations=relations,
            vid2rels=vid2rels,
        )

        # re-order split if subsampling is required

        if prune is not None:
            assert prune > 0
            cw, ow = _split_create_prune(
                cw=cw,
                ow=ow,
                concepts=concepts,
                relations=relations,
                vid2rels=vid2rels,
                prune=prune,
            )

        # divide open-world in validation and test

        train, valid, test = _split_ow_val_test(cw, ow, ratio_val)

        build.add(
            concept_vertices=set(vid for vid, _ in concepts),
            mentions={
                CLOSED_WORLD: train,
                OPEN_VALIDATION: valid,
                OPEN_TEST: test,
            },
        )

        return build()


def _split2irt2_create_ids(build, split):
    vids = Incrementer()
    rids = Incrementer()
    mids = Incrementer()

    build.add(
        vertices={
            vids[old_vid]: split.graph.source.ents[old_vid]
            for subset in split.vertices.values()
            for old_vid in subset
        },
    )

    vids.freeze()
    log.info(f"vid mapping: retained {len(vids)}")

    build.add(
        relations={
            rids[old_rid]: split.graph.source.rels[old_rid]
            for old_rid in split.relations
        }
    )

    rids.freeze()
    log.info(f"rid mapping: retained {len(rids)}")

    build.add(
        mentions={
            mids[(old_vid, mention)]: mention
            for subset in split.mentions.values()
            for old_vid, mention in subset
        },
    )

    mids.freeze()
    log.info(f"mid mapping: retained {len(mids)}")

    mentions: dict[VID, set[MID]] = {
        kind: buckets(
            col=split.mentions[kind],
            key=lambda _, t: (vids[t[0]], mids[t]),
            mapper=set,
        )
        for kind in split.mentions
    }

    build.add(
        closed_mentions=mentions[CLOSED_WORLD],
        open_mentions_val=mentions[OPEN_VALIDATION],
        open_mentions_test=mentions[OPEN_TEST],
    )

    return mentions, vids, rids, mids


def _split2irt2_create_triples(build, split, vids, rids, mentions):
    def idmap(triple):
        h, t, r = triple
        return vids[h], vids[t], rids[r]

    # create set[Triple]
    build.add(closed_triples=set(map(idmap, split.triples[CLOSED_WORLD])))

    for kind in OPEN_WORLD:

        # heads

        task_heads = defaultdict(set)
        head_triples = split.open_world_head_triples[kind]
        for h, t, r in map(idmap, head_triples):
            assert len(mentions[kind][h]) > 0
            for mid in mentions[kind][h]:
                task_heads[(mid, r)].add(t)

        # tails

        task_tails = defaultdict(set)
        tail_triples = split.open_world_tail_triples[kind]
        for h, t, r in map(idmap, tail_triples):
            assert len(mentions[kind][t]) > 0
            for mid in mentions[kind][t]:
                task_tails[(mid, r)].add(h)

        key = "val" if kind == OPEN_VALIDATION else "test"

        build.add(
            **{
                f"open_task_{key}_heads": task_heads,
                f"open_task_{key}_tails": task_tails,
            }
        )

        _count = sum(map(len, task_heads.values()))
        log.info(f"{kind}: added {_count} head tasks from {len(head_triples)} triples")

        _count = sum(map(len, task_tails.values()))
        log.info(f"{kind}: added {_count} tail tasks from {len(tail_triples)} triples")


# this functions assigns new gapless IDS starting at 0
def split2irt2(config, split) -> IRT2:
    """Transform a create.Split to an dataset.IRT2."""
    build = Builder(IRT2)
    build.add(path=irt2.ENV.DIR.DATA / "irt2" / "cde" / "large")
    build.add(
        config={
            "create": config,
            "created": datetime.now().isoformat(),
        }
    )

    mentions, vids, rids, mids = _split2irt2_create_ids(build, split)
    _split2irt2_create_triples(build, split, vids, rids, mentions)

    # check

    # assert that triple split on mention level are disjoint
    def _triples_from_head_task(task):
        return {(mid, rid, vid) for (mid, rid), vids in task.items() for vid in vids}

    def _triples_from_tail_task(task):
        return {(vid, rid, mid) for (mid, rid), vids in task.items() for vid in vids}

    def _head_triples_from_closed_world(triples):
        return {
            (mid, rid, vid)
            for head, rid, vid in triples
            for mid in mentions[CLOSED_WORLD][head]
        }

    def _tail_triples_from_closed_world(triples):
        return {
            (vid, rid, mid)
            for vid, rid, tail in triples
            for mid in mentions[CLOSED_WORLD][tail]
        }

    assert not set.intersection(
        *(
            _tail_triples_from_closed_world(build.get("closed_triples")),
            _triples_from_tail_task(build.get("open_task_val_tails")),
            _triples_from_tail_task(build.get("open_task_test_tails")),
        )
    ), "test leakage for tail triples"

    assert not set.intersection(
        *(
            _head_triples_from_closed_world(build.get("closed_triples")),
            _triples_from_head_task(build.get("open_task_val_heads")),
            _triples_from_head_task(build.get("open_task_test_heads")),
        )
    ), "test leakage for head triples"

    return build()


def _copy_text(out, dataset, counts):
    config = dataset.config["create"]

    # the text sampling requires some link to be implemented between
    # the original EID value which identifies the original entity and
    # the new vid/vertex name (e.g. the EID of any CodEx entity is the
    # wikidata id and the vertex name has the EID prepended))
    assert config["graph name"].startswith("CodEx"), "different id matching required"
    eid2vid = {name.split(":")[0]: vid for vid, name in dataset.vertices.items()}

    # constains mapping: (vid, mentions) -> mid
    # used to look up to which split a text context belongs
    mentions = {
        kind: {(vid, dataset.mentions[mid]): mid for vid, mid in unbucket(source)}
        for kind, source in [
            ("training", dataset.closed_mentions),
            ("validation", dataset.open_mentions_val),
            ("test", dataset.open_mentions_test),
        ]
    }

    sep = config["separator"]
    seen = defaultdict(set)

    with ExitStack() as stack:

        def was_seen(kind, sentence):
            if kind == "validation" and sentence in seen["training"]:
                return True

            if kind == "test":
                if sentence in seen["training"] or sentence in seen["validation"]:
                    return True

            return False

        fds = {
            k: stack.enter_context(gzip.open(name, mode=mode))
            for k, name, mode in [
                ("source", irt2.ENV.DIR.ROOT / config["source sentences"], "rb"),
                ("training", out / "closed.train-contexts.txt.gz", "wb"),
                ("validation", out / "open.validation-contexts.txt.gz", "wb"),
                ("test", out / "open.test-contexts.txt.gz", "wb"),
            ]
        }

        log.info("distributing text, this might take a while...")
        for line in islice(fds["source"], None):
            eid, origin, norm, mention, sentence = decode(line, sep=sep)
            counts["text total"] += 1

            if eid not in eid2vid:
                counts["text skipped eid"] += 1
                continue

            key = eid2vid[eid], norm

            mid = None
            for kind, lookup in mentions.items():
                if key in lookup:
                    mid = lookup[key]
                    break

            # no duplicate texts in the open world
            if was_seen(kind, sentence):
                counts["text skipped seen"] += 1

            # pruned mentions
            elif not mid:
                # (see create.get_mentions())
                counts["text skipped mid"] += 1

            # checks passed and retained
            else:
                # corresponds to irt2.dataset.Context
                fds[kind].write(
                    encode((mid, mention, origin, sentence), sep=sep, fn=str)
                )

                seen[kind].add(sentence)
                counts[f"text retained {kind}"] += 1

    log.info(f"distributed {counts['text total']} sentences")


def write_dataset(
    out: Path,
    dataset: IRT2,
    overwrite: bool = False,
):
    """Write IRT2 to disk."""
    out = kpath(out)
    if not overwrite and out.exists():
        raise irt2.IRT2Error(f"{out} already exists.")

    config = dataset.config["create"]
    counts = Counter()

    def path_norm(target, pov: Path):
        return str(target.relative_to(pov))

    # config
    with (out / "config.yaml").open(mode="w") as fd:
        yaml.safe_dump(dataset.config, fd)

    tup2bytes = partial(encode, fn=str, sep=config["separator"])

    # vertices
    with (out / "vertices.txt").open(mode="wb") as fd:
        fd.write(b"# unique vertex identifier\n")
        fd.write(b"# vertex id:vid | name:str\n")
        for line in map(tup2bytes, dataset.vertices.items()):
            fd.write(line)
            counts["ids vertices"] += 1

    # relations
    with (out / "relations.txt").open(mode="wb") as fd:
        fd.write(b"# unique relation identifier\n")
        fd.write(b"# relation id:rid | name:str\n")
        for line in map(tup2bytes, dataset.relations.items()):
            fd.write(line)
            counts["ids relations"] += 1

    # mentions
    with (out / "mentions.txt").open(mode="wb") as fd:
        fd.write(b"# unique mention identifier\n")
        fd.write(b"# mention id:mid | name:str\n")
        for line in map(tup2bytes, dataset.mentions.items()):
            fd.write(line)
            counts["ids mentions"] += 1

    def _write_mentions(kind, split, mentions):
        with (out / f"{kind}.{split}-mentions.txt").open(mode="wb") as fd:
            fd.write(b"# {kind}-world mentions (" + split.encode() + b")\n")
            fd.write(b"# vertex id:vid | mention id: mid\n")
            for line in map(tup2bytes, unbucket(mentions)):
                fd.write(line)
                counts[f"{kind} mentions"] += 1

    _write_mentions(
        kind="closed",
        split="train",
        mentions=dataset.closed_mentions,
    )

    # write triples
    with (out / "closed.train-triples.txt").open(mode="wb") as fd:
        fd.write(b"# closed world graph\n")
        fd.write(b"# head:vid | tail:vid | relation:rid\n")
        for line in map(tup2bytes, dataset.closed_triples):
            fd.write(line)
            counts["triples closed-world"] += 1

    def _write_open_task(split, direction, col):
        assert direction in {"head", "tail"}
        opposite = "tail" if direction == "head" else "head"

        with (out / f"open.{split}-{direction}.txt").open(mode="wb") as fd:
            fd.write(
                (
                    f"# {direction} is known, {opposite} mentions are queries\n"
                    f"# {opposite} mention id:mid | relation:rid"
                    f" | target {direction} vertex:vid\n"
                ).encode("utf-8")
            )

            for line in map(
                tup2bytes,
                ((mid, rid, vid) for (mid, rid), vids in col.items() for vid in vids),
            ):
                fd.write(line)
                counts[f"open {split} {direction}"] += 1

    def _write_open(split, mentions, heads, tails):
        _write_mentions("open", split, mentions)
        _write_open_task(split, "head", heads)
        _write_open_task(split, "tail", tails)

    _write_open(
        split="validation",
        mentions=dataset.open_mentions_val,
        heads=dataset.open_task_val_heads,
        tails=dataset.open_task_val_tails,
    )

    _write_open(
        split="test",
        mentions=dataset.open_mentions_test,
        heads=dataset.open_task_test_heads,
        tails=dataset.open_task_test_tails,
    )

    _copy_text(out, dataset, counts)
    return counts


def create_dataset(
    out: Path,
    config: dict,
    split: Split,
    overwrite: bool = False,
):
    """
    Create distribution package.

       - initialize a dataset.IRT2 instance
         (this assigns new ids - to have continuous ids starting at 0!)
       - select all associated sentences and write them to their respective files

    This step requires sampled sentences:
       - see scripts/create_text.py

    """
    assert overwrite or not out.exists()
    out.mkdir(parents=True, exist_ok=True)

    dataset = split2irt2(config, split)
    counts = write_dataset(out, dataset, overwrite)

    return dataset, counts
