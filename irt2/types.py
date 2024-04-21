"""Project wide types and models."""

import enum
from dataclasses import dataclass, field
from functools import cached_property
from typing import Generator, Iterator

from ktz.collections import buckets
from ktz.string import decode_line

VID = int  # vertex id
MID = int  # mention id
RID = int  # relation id
EID = str  # upstream entity id (e.g. Wikidata ID for CodEx)

Mention = str

Triple = tuple[VID, VID, RID]  # used for graphs
Sample = tuple[MID, RID, VID]  # used for tasks

Head, Tail = VID, VID


class Split(enum.Enum):
    train = enum.auto()
    valid = enum.auto()
    test = enum.auto()


@dataclass
class IDMap:
    vid2split: dict[VID, Split] = field(default_factory=dict)
    vid2mids: dict[VID, set[MID]] = field(default_factory=dict)

    vid2str: dict[VID, str] = field(default_factory=dict)
    mid2str: dict[MID, str] = field(default_factory=dict)
    rid2str: dict[RID, str] = field(default_factory=dict)

    @cached_property
    def split2vids(self) -> dict[Split, set[VID]]:
        return buckets(
            col=self.vid2split.items(),
            key=lambda _, tup: (tup[1], tup[0]),
            mapper=set,
        )  # type: ignore TODO fix upstream

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
