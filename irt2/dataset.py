"""IRT2 data model."""

import enum
import gzip
import logging
import textwrap
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property, partial
from itertools import combinations
from pathlib import Path
from typing import Callable, Generator, Iterator, Union

import yaml
from ktz.collections import buckets
from ktz.dataclasses import Builder
from ktz.filesystem import path as kpath
from ktz.string import decode_line

import irt2
from irt2.graph import Graph, GraphImport, Relation
from irt2.types import MID, RID, VID, Sample, Triple

log = logging.getLogger(__name__)
tee = irt2.tee(log)


def _open(ctx):
    with ctx as fd:
        yield from filter(lambda bs: bs[0] != ord("#"), fd)


def _fopen(path):
    """Open regular file, read binary and skip comments."""
    return _open(kpath(path, is_file=True).open(mode="rb"))


def _gopen(path):
    """Open gzipped file, read binary and skip comments."""
    path = str(kpath(path, is_file=True))
    return _open(gzip.open(path, mode="rb"))


@dataclass(frozen=True)
class Context:
    """A single text context."""

    mid: MID
    mention: str  # it is asserted that 'mention in sentence'
    origin: str  # e.g. a Wikipedia page
    data: str  # e.g. a sentence

    def __str__(self):
        """Two-line content representation."""
        return f"{self.mention} (mid={self.mid}) [origin={self.origin}]\n>{self.data}<"

    @classmethod
    def from_line(cls, line: bytes, sep: str):
        """Transform context from line."""
        args = decode_line(line, fns=(int, str, str, str), sep=sep)
        return Context(*args)  # type: ignore FIXME upstream


# this is a contextmanager which yields a generator for Context objects
ContextGenerator = Iterator[Generator[Context, None, None]]


class Split(enum.Enum):
    train = enum.auto()
    valid = enum.auto()
    test = enum.auto()


def text_lazy(
    mapping: dict[Split, Path],
    seperator: str,
) -> Callable[[Split], ContextGenerator]:
    assert all(path for path in mapping.values())

    def wrapped(split: Split) -> ContextGenerator:
        gen = _gopen(mapping[split])
        yield (Context.from_line(line, sep=seperator) for line in gen)

    return wrapped


def text_eager(
    mapping: dict[Split, list[Context]]
) -> Callable[[Split], ContextGenerator]:
    def wrapped(split: Split) -> ContextGenerator:
        yield (ctx for ctx in mapping[split])

    return wrapped


@dataclass
class IRT2:
    """IRT2 data collection."""

    path: Path
    config: dict

    vertices: dict[VID, str]
    relations: dict[RID, str]
    mentions: dict[MID, str]

    closed_triples: set[Triple]

    closed_mentions: dict[VID, set[MID]]
    open_mentions_val: dict[VID, set[MID]]
    open_mentions_test: dict[VID, set[MID]]

    # internally used

    _text_loader: Callable[[Split], ContextGenerator]

    _open_val_heads: set[Sample]
    _open_val_tails: set[Sample]

    _open_test_heads: set[Sample]
    _open_test_tails: set[Sample]

    # --- convenience

    @property
    def name(self) -> str:
        """Get the dataset name - (e.g. IRT2/CDE-L)."""
        return self.config["create"]["name"]

    @cached_property
    def mid2vid(self) -> dict[MID, VID]:
        """Obtain a global MID->VID mapping."""
        mentions = (
            self.closed_mentions.items(),
            self.open_mentions_val.items(),
            self.open_mentions_test.items(),
        )

        gen = ((mid, vid) for col in mentions for vid, mids in col for mid in mids)
        return dict(gen)

    @cached_property
    def closed_vertices(self):
        """Closed-world vertices seen at training time."""
        return {vid for head, tail, _ in self.closed_triples for vid in (head, tail)}

    @cached_property
    def open_vertices_val(self):
        """Fully-inductive vertices seen at validation time.

        To obtain both semi-inductive and fully-inductive vertices,
        use open_mentions_val.
        """
        return set(self.open_mentions_val) - self.closed_vertices

    @cached_property
    def open_vertices_test(self):
        """Fully-inductive vertices seen at test time.

        To obtain both semi-inductive and fully-inductive vertices,
        use open_mentions_test.
        """
        return set(self.open_mentions_test) - self.closed_vertices

    # ---

    # open-world knowledge graph completion task
    # given a mention and relation as query, produce a ranking of vertices
    #   - you may use the mention
    #   - *_heads: do head-prediction, the given mid is a tail representation
    #   - *_tails: do tail-prediction, the given mid is a head representation
    #   - see ipynb/load-dataset.ipynb for examples

    def _open_kgc(self, source):
        task = defaultdict(set)

        for mid, rid, vid in source:
            task[(mid, rid)].add(vid)

        return dict(task)

    @cached_property
    def open_kgc_val_heads(self) -> dict[tuple[MID, RID], set[VID]]:
        """Get the kgc validation task."""
        return self._open_kgc(self._open_val_tails)

    @cached_property
    def open_kgc_val_tails(self) -> dict[tuple[MID, RID], set[VID]]:
        """Get the kgc validation task."""
        return self._open_kgc(self._open_val_heads)

    @cached_property
    def open_kgc_test_heads(self) -> dict[tuple[MID, RID], set[VID]]:
        """Get the kgc test task."""
        return self._open_kgc(self._open_test_tails)

    @cached_property
    def open_kgc_test_tails(self) -> dict[tuple[MID, RID], set[VID]]:
        """Get the kgc test task."""
        return self._open_kgc(self._open_test_heads)

    # ---

    # open-world ranking task
    # given a vertex and relation, produce a ranking of mentions
    #   - you must not use the mention
    #   - *_heads: find new head mentions for a given closed-world tail
    #   - *_tails: find new tail mentions for a given closed-world head
    #   - see ipynb/load-dataset.ipynb for examples

    def _open_ranking(self, source):
        task = defaultdict(set)

        for mid, rid, vid in source:
            task[(vid, rid)].add(mid)

        return dict(task)

    @cached_property
    def open_ranking_val_heads(self) -> dict[tuple[VID, RID], set[MID]]:
        """Get the ranking validation heads task."""
        return self._open_ranking(self._open_val_heads)

    @cached_property
    def open_ranking_val_tails(self) -> dict[tuple[VID, RID], set[MID]]:
        """Get the ranking validation tails task."""
        return self._open_ranking(self._open_val_tails)

    @cached_property
    def open_ranking_test_heads(self) -> dict[tuple[VID, RID], set[MID]]:
        """Get the ranking test heads task."""
        return self._open_ranking(self._open_test_heads)

    @cached_property
    def open_ranking_test_tails(self) -> dict[tuple[VID, RID], set[MID]]:
        """Get the ranking test tails task."""
        return self._open_ranking(self._open_test_tails)

    # --

    @contextmanager
    def closed_contexts(self) -> ContextGenerator:
        """Get a generator for closed-world contexts."""
        return self._text_loader(Split.train)

    @contextmanager
    def open_contexts_val(self) -> ContextGenerator:
        """Get a generator for open-world contexts (validation split)."""
        return self._text_loader(Split.valid)

    @contextmanager
    def open_contexts_test(self) -> ContextGenerator:
        """Get a generator for open-world contexts (test split)."""
        return self._text_loader(Split.test)

    # --

    @cached_property
    def description(self) -> str:
        """Summary of key figures."""
        cfg = self.config["create"]
        created = datetime.fromisoformat(self.config["created"])

        heading = textwrap.dedent(
            f"""
            {cfg["name"]}
            created: {created.ctime()}
            """
        )

        def mentions(dic):
            mids = {mid for mids in dic.values() for mid in mids}
            avg = len(mids) / len(dic) if len(dic) else 0
            return f"{len(mids)} (~{avg:2.3f} per vertex)"

        def contexts(mgr):
            with mgr() as contexts:
                return sum(1 for _ in contexts)

        def sumval(col):
            # which measures the amount of unique (mid, rid, vid)
            # triples of the tasks (and as such they have the same
            # size for both kgc and ranking)
            return sum(map(len, col.values()))

        semi_inductive_vertices_val = len(
            set(self.open_mentions_val) - self.open_vertices_val
        )

        semi_inductive_vertices_test = len(
            set(self.open_mentions_test) - self.open_vertices_test
        )

        body = textwrap.indent(
            textwrap.dedent(
                f"""
                vertices: {len(self.vertices)}
                relations: {len(self.relations)}
                mentions: {len(self.mentions)}

                closed-world
                  triples: {len(self.closed_triples)}
                  vertices: {len(self.closed_mentions)}
                  mentions: {mentions(self.closed_mentions)}
                  contexts: {contexts(self.closed_contexts)}

                open-world (validation)
                  mentions: {mentions(self.open_mentions_val)}
                  contexts: {contexts(self.open_contexts_val)}
                  tasks:
                    heads: {sumval(self.open_kgc_val_heads)}
                    tails: {sumval(self.open_kgc_val_tails)}
                  vertices: {len(self.open_mentions_val)}
                  semi inductive vertices: {semi_inductive_vertices_val}
                  fully inductive vertices: {len(self.open_vertices_val)}

                open-world (test)
                  mentions: {mentions(self.open_mentions_test)}
                  contexts: {contexts(self.open_contexts_test)}
                  task:
                    heads: {sumval(self.open_kgc_test_heads)}
                    tails: {sumval(self.open_kgc_test_tails)}
                  vertices: {len(self.open_mentions_test)}
                  semi inductive vertices: {semi_inductive_vertices_test}
                  fully inductive vertices: {len(self.open_vertices_test)}
                """
            ),
            prefix=" " * 2,
        )

        return heading + body

    def __repr__(self):
        # although semantically incorrect this returns
        # str(self) as it clogged my terminal all the
        # time when debugging ;)
        return str(self)

    def __str__(self):
        """Short description."""
        return (
            f"{self.config['create']['name']}:"
            f" {len(self.vertices)} vertices |"
            f" {len(self.relations)} relations |"
            f" {len(self.mentions)} mentions"
        )

    # --

    @classmethod
    def from_dir(cls, path: Union[str, Path]):
        """Load the dataset from a directory.

        Parameters
        ----------
        IRT2 : IRT2
            class constructor
        path : Path
            where to load the data from

        """
        build = Builder(cls)
        build.add(path=kpath(path, is_dir=True))
        fp: Path = build.get("path")

        with (fp / "config.yaml").open(mode="r") as fd:
            config = yaml.safe_load(fd)
            build.add(config=config)

        decode = partial(decode_line, sep=config["create"]["separator"])
        ints = partial(decode, fn=int)

        # -- ids

        def load_ids(fname) -> dict[int, str]:
            pairs = partial(decode, fns=(int, str))
            return dict(map(pairs, _fopen(fp / fname)))  # type: ignore FIXME upstream

        build.add(
            vertices=load_ids("vertices.txt"),
            relations=load_ids("relations.txt"),
            mentions=load_ids("mentions.txt"),
        )

        # -- triples

        def load_triples(fname) -> set[Triple]:
            return set(map(ints, _fopen(fp / fname)))  # type: ignore FIXME upstream

        build.add(closed_triples=load_triples("closed.train-triples.txt"))

        # -- mentions

        def load_mentions(fname) -> dict[VID, set[MID]]:
            items = map(ints, _fopen(fp / fname))
            return buckets(col=items, mapper=set)  # type: ignore FIXME upstream

        build.add(
            closed_mentions=load_mentions("closed.train-mentions.txt"),
            open_mentions_val=load_mentions("open.validation-mentions.txt"),
            open_mentions_test=load_mentions("open.test-mentions.txt"),
        )

        # -- open-world samples

        cw_vids = {v for h, t, _ in build.get("closed_triples") for v in (h, t)}

        def load_ow(fname) -> set[Sample]:
            triples = set(map(ints, _fopen(fp / fname)))
            filtered = {(m, r, v) for m, r, v in triples if v in cw_vids}  # type: ignore FIXME upstream

            log.info("loading {len(filtered)}/{len(triples)} triples from {fname}")
            return filtered  # type: ignore FIXME upstream

        build.add(
            _text_loader=text_lazy(
                mapping={
                    Split.train: fp / f"closed.train-contexts.txt.gz",
                    Split.valid: fp / f"open.validation-contexts.txt.gz",
                    Split.test: fp / f"open.test-contexts.txt.gz",
                },
                seperator=config["create"]["separator"],
            ),
            _open_val_heads=load_ow("open.validation-head.txt"),
            _open_val_tails=load_ow("open.validation-tail.txt"),
            _open_test_heads=load_ow("open.test-head.txt"),
            _open_test_tails=load_ow("open.test-tail.txt"),
        )

        return build()

    # --  utilities

    @cached_property
    def graph(self) -> Graph:
        """
        Create a Graph from the training triples.

        This property offers a irt2.graph.Graph instance which offers
        some convenience functionality (searching, pretty-printing) on
        top of networkx.

        Returns
        -------
        Graph
            Graph instance

        """
        return Graph(
            name=self.config["create"]["name"],
            source=GraphImport(
                triples=self.closed_triples,
                ents=self.vertices,
                rels=self.relations,
            ),
        )

    @cached_property
    def ratios(self) -> list[Relation]:
        """
        Create a list of relations sorted by ratio.

        Returns
        -------
        list[Relation]
            The relations

        """
        return Relation.from_graph(self.graph)

    def check(self):
        """Run self-checks."""

        def disjoint(*sets):
            for a, b in combinations(sets, r=2):
                assert not (a & b)

        # mentions are disjoint
        disjoint(
            set.union(*self.closed_mentions.values()),
            set.union(*self.open_mentions_val.values()),
            set.union(*self.open_mentions_test.values()),
        )
