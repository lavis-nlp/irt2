# -*- coding: utf-8 -*-

"""IRT2 data model."""


from irt2.types import MID, VID, RID, Triple

from ktz.string import decode_line
from ktz.dataclasses import Builder
from ktz.filesystem import path as kpath

import gzip
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from collections import defaultdict
from contextlib import contextmanager

import yaml


# typedefs

from typing import Union
from typing import Generator


def _skip_comments(it):
    return filter(lambda bs: bs[0] != ord("#"), it)


def _load_mentions(path, mentions: dict, decode):
    with path.open(mode="rb") as fd:
        gen = (decode(line, fns=(int, int, str)) for line in _skip_comments(fd))

        split_mentions = defaultdict(set)
        for mid, vid, mention in gen:
            mentions[mid] = mention
            split_mentions[vid].add(mid)

    return dict(split_mentions)


def _load_task(path, decode):
    with path.open(mode="rb") as fd:

        gen = (decode(line, fns=(int, int, int)) for line in _skip_comments(fd))

        open_task = {}
        for mid, rid, vid in gen:
            open_task[(mid, rid)] = vid

    return open_task


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

    # closed-world data
    closed_triples: set[Triple]
    closed_mentions: dict[VID, set[MID]]

    # open-world data
    open_mentions: dict[VID, set[MID]]
    open_task_heads: dict[tuple[MID, RID], VID]
    open_task_tails: dict[tuple[MID, RID], VID]

    # --

    def _contexts(self, kind: str):
        path = kpath(self.path / f"{kind}.contexts.txt.gz", is_file=True)
        with gzip.open(path, mode="r") as fd:
            ctxs = (
                Context.from_line(line, sep=self.config["seperator"])
                for line in _skip_comments(fd)
            )

            # TODO make sure this is not necessary
            yield (ctx for ctx in ctxs if ctx.mid in self.mentions)

    @contextmanager
    def closed_contexts(
        self,
    ) -> Generator[Context, None, None]:
        """Get a generator for closed-world contexts.

        Returns
        -------
        Generator[tuple[MID, Origin, Mention, Sentence], None, None]
        """
        yield from self._contexts(kind="closed")

    @contextmanager
    def open_contexts(
        self,
    ) -> Generator[Context, None, None]:
        """Get a generator for open-world contexts.

        Returns
        -------
        Generator[tuple[MID, Origin, Mention, Sentence], None, None]
        """
        yield from self._contexts(kind="open")

    # --

    def __str__(self):
        """Short description."""
        return (
            f"{self.config['name']}:"
            f" {len(self.vertices)} vertices |"
            f" {len(self.relations)} relations |"
            f" {len(self.mentions)} mentions"
        )

    @classmethod
    def from_dir(IRT2, path: Union[str, Path]):
        """Load the dataset from directory.

        Parameters
        ----------
        IRT2 : IRT2
            class constructor
        path : Path
            where to load the data from

        """

        def _open(path, mode="rb"):
            return kpath(path, is_file=True).open(mode=mode)

        build = Builder(IRT2)
        build.add(path=kpath(path, is_dir=True))
        path = build.get("path")

        with _open(path / "config.yml", mode="r") as fd:
            build.add(config=yaml.load(fd, Loader=yaml.FullLoader))

        config = build.get("config")
        decode = partial(decode_line, sep=config["seperator"])

        # load entities
        with _open(path / "vertices.txt") as fd:
            lines = (decode(line, fns=(int, str)) for line in _skip_comments(fd))
            build.add(vertices=dict(lines))

        # load relations
        with _open(path / "relations.txt") as fd:
            lines = (decode(line, fns=(int, str)) for line in _skip_comments(fd))
            build.add(relations=dict(lines))

        # load closed-world triples
        with _open(path / "closed.triples.txt") as fd:
            lines = (decode(line, fn=int) for line in _skip_comments(fd))
            build.add(closed_triples=set(lines))

        # load mentions
        mentions = {}
        build.add(
            closed_mentions=_load_mentions(
                path=path / "closed.mentions.txt",
                mentions=mentions,
                decode=decode,
            )
        )

        build.add(
            open_mentions=_load_mentions(
                path=path / "open.mentions.txt",
                mentions=mentions,
                decode=decode,
            )
        )

        build.add(mentions=mentions)

        # load tasks
        build.add(
            open_task_heads=_load_task(
                path=(path / "open.task.heads.txt"),
                decode=decode,
            )
        )

        build.add(
            open_task_tails=_load_task(
                path=(path / "open.task.tails.txt"),
                decode=decode,
            )
        )

        return build()
