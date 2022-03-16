# -*- coding: utf-8 -*-

"""IRT2 data model."""


import irt2
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
            f"{self.config['create']['name']}:"
            f" {len(self.vertices)} vertices |"
            f" {len(self.relations)} relations |"
            f" {len(self.mentions)} mentions"
        )

    # --
    # persistence
    def to_dir(self, path: Union[str, Path], overwrite: bool = False):
        """Write IRT2 to disk."""
        path = kpath(path)
        if not overwrite and path.exists():
            raise irt2.IRT2Error(f"{path} already exists.")

        raise NotImplementedError()

        # def path_norm(target, pov: Path):
        #     return str(target.relative_to(pov))

        # # write config
        # with (path / "config.yaml").open(mode="w") as fd:
        #     yaml.dump(
        #         {
        #             "source graph": path_norm(GRAPH_PATH_CDE, pov=irt.ENV.ROOT_DIR),
        #             "source matches": path_norm(PATH_MATCHES_CDE, pov=irt.ENV.ROOT_DIR),
        #             "source pages": path_norm(DB_CDE, pov=irt.ENV.ROOT_DIR),
        #             "spacy model": SPACY_MODEL,
        #             "created": datetime.now().isoformat(),
        #             "match count threshold": MATCHES_PRUNE,
        #             "mention split ratio": MENTION_SPLIT_RATIO,
        #             "seed": SEED,
        #             "seperator": SEP,
        #         },
        #         fd,
        #     )

        # # write triples
        # with (path / "closed.triples.txt").open(mode="wb") as fd:
        #     fd.write(b"# closed world graph\n")
        #     fd.write(b"# head:vid | tail:vid | relation:rid\n")
        #     for triple in sel.cw:
        #         fd.write(encode(triple))

        # # write vertices
        # with (path / "vertices.txt").open(mode="wb") as fd:
        #     fd.write(b"# unique vertex identifier\n")
        #     fd.write(b"# vertex id:vid | name:str\n")
        #     for vid, name in sel.g.source.ents.items():
        #         fd.write(encode((vid, name)))

        # # write vertices
        # with (path / "relations.txt").open(mode="wb") as fd:
        #     fd.write(b"# unique relation identifier\n")
        #     fd.write(b"# relation id:rid | name:str\n")
        #     for rid, name in sel.g.source.rels.items():
        #         fd.write(encode((rid, name)))

        # # write mentions
        # with (path / "closed.mentions.txt").open(mode="wb") as fd:
        #     fd.write(b"# mention:mid | vertex:vid | name:str\n")
        #     _write_mentions(
        #         fd=fd,
        #         sel=sel,
        #         mids=mids,
        #         eids=sel.eids,
        #         items=sel.mention_split["cw"].items(),
        #     )

        # with (path / "open.mentions.txt").open(mode="wb") as fd:
        #     fd.write(b"# mention:mid | vertex:vid | name:str\n")
        #     _write_mentions(
        #         fd=fd,
        #         sel=sel,
        #         mids=mids,
        #         eids=sel.eids,
        #         items=sel.mention_split["ow"].items(),
        #     )

        # vid2eid = {v: k for k, v in sel.eid2vid.items()}

        # # write task
        # with (path / "open.task.heads.txt").open(mode="wb") as fd:
        #     fd.write(b"# head is known, tail mentions are queries\n")
        #     fd.write(b"# tail mention id:mid | relation:rid | target head vertex:vid")

        #     # heads are queries
        #     for h, t, r in sel.ow_heads:
        #         eid = vid2eid[h]
        #         assert sel.mention_split["ow"][eid], eid
        #         for mention in sel.mention_split["ow"][eid]:
        #             mid = mids[(eid, mention)]
        #             fd.write(encode((mid, r, t)))

        # with (path / "open.task.tails.txt").open(mode="wb") as fd:
        #     fd.write(b"# tail is known, head mentions are queries\n")
        #     fd.write(b"# head mention id:mid | relation:rid | target tail vertex:vid")

        #     # tails are queries
        #     for h, t, r in sel.ow_tails:
        #         eid = vid2eid[t]
        #         assert sel.mention_split["ow"][eid], eid
        #         for mention in sel.mention_split["ow"][eid]:
        #             mid = mids[(eid, mention)]
        #             fd.write(encode((mid, r, h)))

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
