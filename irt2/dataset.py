# -*- coding: utf-8 -*-

"""IRT2 data model."""


from irt2.types import MID, VID, RID, Triple

from ktz.collections import buckets
from ktz.dataclasses import Builder
from ktz.filesystem import path as kpath
from ktz.string import decode_line

import gzip
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from functools import cached_property
from contextlib import contextmanager

import yaml


# typedefs

from typing import Union
from typing import Generator


def _open(ctx):
    with ctx as fd:
        yield from filter(lambda bs: bs[0] != ord("#"), fd)


def _fopen(path):
    """Open file, read binary and skip comments."""
    return _open(kpath(path, is_file=True).open(mode="rb"))


def _gopen(path):
    """Open gzipped file, read binary and skip comments."""
    return _open(
        gzip.open(
            str(kpath(path, is_file=True)),
            mode="rb",
        )
    )


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
    def from_line(Context, line: bytes, sep: bytes):
        """Transform context from line."""
        args = decode_line(line, fns=(int, str, str, str), sep=sep)
        return Context(*args)


@dataclass
class IRT2:
    """IRT2 data collection."""

    path: Path
    config: dict

    vertices: dict[VID, str]
    relations: dict[RID, str]
    mentions: dict[MID, str]

    # training: closed-world

    closed_triples: set[Triple]
    closed_mentions: dict[VID, set[MID]]

    # validation: open-world

    open_mentions_val: dict[VID, set[MID]]
    open_task_val_heads: dict[tuple[MID, RID], set[VID]]
    open_task_val_tails: dict[tuple[MID, RID], set[VID]]

    # test: open-world

    open_mentions_test: dict[VID, set[MID]]
    open_task_test_heads: dict[tuple[MID, RID], set[VID]]
    open_task_test_tails: dict[tuple[MID, RID], set[VID]]

    # --

    @cached_property
    def description(self) -> str:
        """Summary of key figures."""
        raise NotImplementedError()

    # --

    def _contexts(self, kind: str):
        path = kpath(self.path / f"{kind}.contexts.txt.gz", is_file=True)
        constructor = partial(Context.from_line, sep=self.config["seperator"])
        yield map(constructor, _gopen(path))

    @contextmanager
    def closed_contexts(self) -> Generator[Context, None, None]:
        """Get a generator for closed-world contexts.

        Returns
        -------
        Generator[tuple[MID, Origin, Mention, Sentence], None, None]
        """
        yield from self._contexts(kind="closed_training")

    @contextmanager
    def open_contexts_validation(self) -> Generator[Context, None, None]:
        """Get a generator for open-world contexts (validation split).

        Returns
        -------
        Generator[tuple[MID, Origin, Mention, Sentence], None, None]
        """
        yield from self._contexts(kind="open_validation")

    @contextmanager
    def open_contexts_test(self) -> Generator[Context, None, None]:
        """Get a generator for open-world contexts (test split).

        Returns
        -------
        Generator[tuple[MID, Origin, Mention, Sentence], None, None]
        """
        yield from self._contexts(kind="open_test")

    # --

    def __str__(self):
        """Short description."""
        return (
            f"{self.config['create']['name']}:"
            f" {len(self.vertices)} vertices |"
            f" {len(self.relations)} relations |"
            f" {len(self.mentions)} mentions"
        )

    @classmethod
    def from_dir(IRT2, path: Union[str, Path]):
        """Load the dataset from a directory.

        Parameters
        ----------
        IRT2 : IRT2
            class constructor
        path : Path
            where to load the data from

        """
        build = Builder(IRT2)
        build.add(path=kpath(path, is_dir=True))
        path = build.get("path")

        with (path / "config.yaml").open(mode="r") as fd:
            config = yaml.safe_load(fd)
            build.add(config=config)

        decode = partial(decode_line, sep=config["create"]["separator"])
        ints = partial(decode, fn=int)

        # -- ids

        def load_ids(fname) -> dict[int, str]:
            pairs = partial(decode, fns=(int, str))
            return dict(map(pairs, _fopen(path / fname)))

        build.add(
            vertices=load_ids("vertices.txt"),
            relations=load_ids("relations.txt"),
            mentions=load_ids("mentions.txt"),
        )

        # -- triples

        def load_triples(fname) -> set[Triple]:
            return set(map(ints, _fopen(path / fname)))

        build.add(closed_triples=load_triples("closed.train-triples.txt"))

        # -- mentions

        def load_mentions(fname) -> dict[VID, set[MID]]:
            items = map(ints, _fopen(path / fname))
            return buckets(col=items, mapper=set)

        build.add(
            closed_mentions=load_mentions("closed.train-mentions.txt"),
            open_mentions_val=load_mentions("open.validation-mentions.txt"),
            open_mentions_test=load_mentions("open.test-mentions.txt"),
        )

        # -- tasks

        def load_task(fname) -> dict[tuple[MID, RID], set[VID]]:
            triples = map(ints, _fopen(path / fname))
            return buckets(
                col=triples,
                # t: (0=MID, 1=RID, 2=VID)
                key=lambda _, t: ((t[0], t[1]), t[2]),
                mapper=set,
            )

        build.add(
            open_task_val_heads=load_task("open.validation-head.txt"),
            open_task_val_tails=load_task("open.validation-tail.txt"),
            open_task_test_heads=load_task("open.test-head.txt"),
            open_task_test_tails=load_task("open.test-tail.txt"),
        )

        return build()
