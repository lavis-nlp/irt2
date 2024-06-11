"""Project wide types and models."""

import enum
from dataclasses import astuple, dataclass, field
from functools import cached_property
from typing import Generator, Iterator

from ktz.collections import buckets
from ktz.string import decode_line, encode_line

VID = int  # vertex id
MID = int  # mention id
RID = int  # relation id
EID = str  # upstream entity id (e.g. Wikidata ID for CodEx)

Mention = str
Entity = MID | VID  # used for tasks
Task = tuple[Entity, RID]  # are derived from Sample

Triple = tuple[VID, VID, RID]  # used for graphs
Sample = tuple[MID, RID, VID]  # used for tasks
GroundTruth = dict[Task, set[Entity]]

Head, Tail = VID, VID


class Split(enum.Enum):
    train = enum.auto()
    valid = enum.auto()
    test = enum.auto()


@dataclass
class IDMap:
    # --- global

    # each mid is unique per vertex, but not their string representation!
    # len(mid2str) =/= len(str2mid)

    vid2str: dict[VID, str] = field(default_factory=dict)
    rid2str: dict[RID, str] = field(default_factory=dict)
    mid2str: dict[MID, str] = field(default_factory=dict)

    # reverse lookups

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
    def str2mids(self) -> dict[str, set[MID]]:
        return dict(
            buckets(
                col=self.mid2str.items(),
                key=lambda _, tup: (tup[1], tup[0]),
                mapper=set,
            )
        )  # type: ignore TODO fix upstream

    @cached_property
    def mid2vid_global(self) -> dict[MID, VID]:
        return {mid: vid for split in Split for mid, vid in self.mid2vid[split].items()}

    # --- split specific

    # vid2mids: vertex to mention mapping per split
    vid2mids: dict[Split, dict[VID, set[MID]]] = field(
        default_factory=lambda: {
            Split.train: {},
            Split.valid: {},
            Split.test: {},
        }
    )

    @cached_property
    def mid2vid(self) -> dict[Split, dict[MID, VID]]:
        return {
            split: {
                mid: vid for vid, mids in self.vid2mids[split].items() for mid in mids
            }
            for split in Split
        }


@dataclass(frozen=True)
class Context:
    """A single text context."""

    mid: MID
    mention: str  # it is asserted that 'mention in sentence' (for irt2 datasets only)
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

    def to_line(self, sep: str):
        return encode_line(astuple(self), fn=str, sep=sep)


# this is a contextmanager which yields a generator for Context objects
ContextGenerator = Iterator[Generator[Context, None, None]]
