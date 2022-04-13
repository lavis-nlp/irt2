# -*- coding: utf-8 -*-

"""Contains the API for ipynb/create.ipynb."""

import irt2
from irt2.dataset import IRT2
from irt2.graph import Graph
from irt2.graph import Relation
from irt2.types import VID, RID, EID, Triple, Mention

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
    def from_csv(Match, line: Iterable[str]):
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

        return Match(
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
def index_matches(path: Path, n: int = None):
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


Flat = set[tuple[VID, Mention]]
Kind = Literal["closed/train", "open/validation", "open/test"]


@dataclass
class Split:
    """Open/Closed-world mention split."""

    KINDS = "closed/train", "open/validation", "open/test"

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
                (f"{kind} (inklusive)", None, None, None),
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

        ow_vertices = self.open_world_vertices
        ow_triples = self.open_world_triples

        rows = (
            (
                "concept",
                len(self.concept_mentions),
                len(self.concept_vertices),
                len(self.concept_triples),
            ),
            (
                "closed world",
                len(self.mentions["closed/train"]),
                len(self.vertices["closed/train"]),
                len(self.triples["closed/train"]),
            ),
            (
                "open world (exclusive)",
                len(self.mentions["open/validation"] | self.mentions["open/test"]),
                len(ow_vertices["open/validation"] | ow_vertices["open/test"]),
                len(ow_triples["open/validation"] | ow_triples["open/test"]),
            ),
            (
                "open world (inklusive)",
                len(self.mentions["open/validation"] | self.mentions["open/test"]),
                len(self.vertices["open/validation"] | self.vertices["open/test"]),
                len(self.triples["open/validation"] | self.triples["open/test"]),
            ),
        )

        return tabulate(
            rows + _open_world_row("open/validation") + _open_world_row("open/test"),
            headers=("", "mentions", "vertices", "triples"),
        )

    @cached_property
    def concept_mentions(self) -> Flat:
        """Return all mentions of concept vertices."""
        return {
            (vid, mention)
            for vid, mention in self.mentions["closed/train"]
            if vid in self.concept_vertices
        }

    # vertex split

    @cached_property
    def vertices(self) -> dict[Kind, set[VID]]:
        """Return vertices involved in the splits."""
        return {k: {vid for vid, _ in self.mentions[k]} for k in self.KINDS}

    @cached_property
    def open_world_vertices(self) -> dict[Kind, set[VID]]:
        """Return open-world vertices."""
        ret = {k: v.copy() for k, v in self.vertices.items()}

        ret["open/validation"] -= ret["closed/train"]
        ret["open/test"] -= ret["closed/train"] | ret["open/validation"]

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
            "closed/train": self._filter_triples(
                self.graph.find(
                    heads=self.open_world_vertices["closed/train"],
                    tails=self.open_world_vertices["closed/train"],
                )
            )
        }

        open = {
            k: self.open_world_head_triples[k] | self.open_world_tail_triples[k]
            for k in ("open/validation", "open/test")
        }
        return closed | open

    @cached_property
    def concept_triples(self) -> set[Triple]:
        """Return all triples where both vertices are concepts."""
        return self.graph.select(
            heads=self.concept_vertices,
            tails=self.concept_vertices,
        )

    @cached_property
    def open_world_triples(self) -> set[Triple]:
        """Return triples where both head and tail are open world vertices."""
        open = {
            k: self._filter_triples(
                self.graph.select(
                    heads=self.open_world_vertices[k],
                    tails=self.open_world_vertices[k],
                )
            )
            for k in ("open/validation", "open/test")
        }

        return open

    @cached_property
    def open_world_head_triples(self) -> dict[Kind, set[Triple]]:
        """Return triples where the head vertex is open-world."""
        return {
            k: self._filter_triples(
                self.graph.find(heads=self.vertices[k]),
            )
            for k in ("open/validation", "open/test")
        }

    @cached_property
    def open_world_tail_triples(self) -> dict[Kind, set[Triple]]:
        """Return triples where the tail vertex is open-world."""
        return {
            k: self._filter_triples(
                self.graph.find(tails=self.vertices[k]),
            )
            for k in ("open/validation", "open/test")
        }

    # --  tests

    def _check_mentions(self, test):

        # the mention split is disjoint
        for kind1, kind2 in combinations(Split.KINDS, r=2):
            test(
                lambda shared: not shared,
                f"{{shared}} mentions shared between {kind1} and {kind2}",
                shared=len(self.mentions[kind1] & self.mentions[kind2]),
            )

        # every mention is associated with a vertex and for each vertex
        # at least one head or tail task exists
        for kind in ("open/validation", "open/test"):
            heads = {head for head, _, _ in self.open_world_head_triples[kind]}
            tails = {tail for _, tail, _ in self.open_world_tail_triples[kind]}
            mentions = {vid for vid, _ in self.mentions[kind]}

            test(
                lambda excess: excess == 0,
                "{excess} vertices of mentions do not have task triples assigned",
                excess=len(mentions - (heads | tails)),
            )

    def _check_vertices(self, test):

        # concept vertices are a subset of the closed-world vertex set
        test(
            lambda concepts, closed: concepts <= closed,
            "concept vertices are not a subset of closed-world vertices",
            concepts=self.concept_vertices,
            closed=self.vertices["closed/train"],
        )

        # open-world vertices are not shared between splits
        for kind1, kind2 in combinations(Split.KINDS, r=2):
            vertices = self.open_world_vertices

            test(
                lambda shared: not shared,
                "{{shared}} open-world vertices between {kind1} and {kind2}",
                shared=len(vertices[kind1] & vertices[kind2]),
            )

    def _check_triples(self, test):
        for kind in Split.KINDS:
            test(
                bool,
                f"there are not triples for {kind}",
                triples=len(self.triples[kind]),
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
    ):
        """Divide mentions by a given ratio and concepts."""
        assert not bool(include_rels) and bool(exclude_rels), "mutex!"

        # configure split

        build = Builder(Split)
        build.add(graph=graph)

        # select relations

        relations = Relation.from_graph(graph)

        includes = set(include_rels)
        assert len(includes) == len(include_rels)

        excludes = set(exclude_rels)
        assert len(excludes) == len(exclude_rels)

        if includes:
            relations = [rel for rel in relations if rel.name in includes]

        elif excludes:
            relations = [rel for rel in relations if rel.name not in excludes]

        assert relations
        build.add(relations={rel.rid for rel in relations})

        # all concept vids, regardless whether they have mentions assigned
        concept_vids = set.union(
            *[rel.concepts for rel in relations if rel.name in concept_rels]
        )

        # remove vertices which have no mentions assigned

        concepts: Flat = set()
        candidates: Flat = set()
        removed_vids = set()

        for vid, eid in graph.source.ents.items():

            # map to upstream
            # eid=Q108946:A Few Good Men -> link=Q108946
            link = eid.split(":")[0]
            if link not in mentions.eid2mentions:
                removed_vids.add(vid)
                continue

            flat = {(vid, mention) for mention in mentions.eid2mentions[link].keys()}
            target = concepts if vid in concept_vids else candidates

            target |= flat

        build.add(removed_vertices=removed_vids)
        _all_vids = set(graph.source.ents)
        log.info(f"retaining {len(_all_vids - removed_vids)}/{len(_all_vids)} vertices")

        # split remaining randomly

        random.seed(seed)
        candidates = sorted(candidates)
        random.shuffle(candidates)

        # draw the threshold between all mentions and then set aside the
        # open world mentions. the remaining mentions can be added to the
        # concept mentions and form the closed world mention set.
        total = len(concepts) + len(candidates)
        lower = int((1 - ratio_train) * total)
        upper = lower + int(len(candidates[lower:]) * ratio_val)

        log.info(f"set aside {len(concepts)} concept mentions")
        log.info(f"splitting mention candidates: 0:{lower}:{upper}:{len(candidates)}")

        assert lower < len(concepts), "threshold too small"

        # validation/test split

        build.add(
            concept_vertices=set(vid for vid, _ in concepts),
            mentions={
                "closed/train": concepts | set(candidates[:lower]),
                "open/validation": set(candidates[lower:upper]),
                "open/test": set(candidates[upper:]),
            },
        )

        return build()


# this functions assigns new gapless IDS starting at 0
def split2irt2(config, split) -> IRT2:
    """Transform a create.Split to an dataset.IRT2."""
    vids = Incrementer()
    rids = Incrementer()
    mids = Incrementer()
    build = Builder(IRT2)
    build.add(path=irt2.ENV.DIR.DATA / "irt2" / "cde" / "large")
    build.add(
        config={
            "create": config,
            "created": datetime.now().isoformat(),
        }
    )

    # re-map ids

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

    mentions = {
        kind: buckets(
            col=split.mentions[kind],
            key=lambda _, t: (vids[t[0]], mids[t]),
            mapper=set,
        )
        for kind in split.mentions
    }

    build.add(
        closed_mentions=mentions["closed/train"],
        open_mentions_val=mentions["open/validation"],
        open_mentions_test=mentions["open/test"],
    )

    # handle triples

    def idmap(triple):
        h, t, r = triple
        return vids[h], vids[t], rids[r]

    # create set[Triple]
    build.add(closed_triples=set(map(idmap, split.triples["closed/train"])))

    for kind in ("open/validation", "open/test"):

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

            assert mentions[kind][t]
            for mid in mentions[kind][t]:
                task_tails[(mid, r)].add(h)

        key = "val" if kind == "open/validation" else "test"

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

    # handle text

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
    with ExitStack() as stack:

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

            if mid:
                # corresponds to irt2.dataset.Context
                fds[kind].write(
                    encode((mid, mention, origin, sentence), sep=sep, fn=str)
                )
                counts[f"text retained {kind}"] += 1

            else:
                # this happens if a mention was pruned
                # (see create.get_mentions())
                counts["text skipped mid"] += 1

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
                counts[f"open {kind} mentions"] += 1

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
