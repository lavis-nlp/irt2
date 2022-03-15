# -*- coding: utf-8 -*-

"""Contains the API for ipynb/create.ipynb."""

import irt2
from irt2.graph import Graph
from irt2.types import VID, EID, Triple, Mention

from ktz.string import args_hash
from ktz.dataclasses import Index
from ktz.filesystem import path as kpath

import re
import csv
import pickle
from pathlib import Path
from itertools import islice
from collections import Counter
from dataclasses import dataclass
from collections import defaultdict

from collections.abc import Iterable


#
# helper
#


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


#
# text data
#


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

    index = Index(
        Match,
        includes={"page", "eid"},
    )

    with path.open(mode="r") as fd:
        reader = islice(csv.reader(fd), n)

        matches = (Match.from_csv(row) for row in reader)
        filtered = filter(lambda m: m.norm_mention, matches)

        index.add(filtered)

    return index


def load_index_matches(
    path: Path,
    invalidate_cache: bool = False,
    n: int = None,
):
    """
    Load (cached) index of Match objects.

    Parameters
    ----------
    path : Path
        Source csv file
    invalidate_cache : bool
        Discard cache if present
    n : int
        Load at most n matches

    """
    path = kpath(path, is_file=True)

    hash = args_hash(path, n)
    cachefile = f"sampling.load_index_matches.{hash}.pkl"
    cache = irt2.ENV.DIR.CACHE / cachefile
    if not invalidate_cache and cache.exists():
        with cache.open(mode="rb") as fd:
            idx_matches = pickle.load(fd)

    else:
        idx_matches = index_matches(path=path, n=n)
        with cache.open(mode="wb") as fd:
            pickle.dump(idx_matches, fd)

    return idx_matches


def get_mentions(
    index: Index[Match],
    prune: int,
) -> tuple[dict[str, str], dict[EID, dict[Mention, int]]]:
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
    mentions = defaultdict(Counter)

    for match in index.flat:
        mentions[match.eid][match.norm_mention] += 1
        norm2mentions[match.norm_mention].add(match.mention)

    pruned = {
        eid: {mention: count for mention, count in counts.items() if count >= prune}
        for eid, counts in mentions.items()
        if len(counts) > 0
    }

    return dict(norm2mentions), pruned


#
#   triple data
#
@dataclass
class Selection:
    """Used to create an IRT2 instance."""

    graph: Graph
    closed_world: set[Triple]

    open_world_heads: set[Triple]
    open_world_tails: set[Triple]

    removed: set[EID]

    eid2vid: dict[EID, VID]
    mention_split: dict

    @property
    def eids(self) -> set[EID]:
        """Get all eids that have not been removed."""
        return set(self.eid2vid) - self.removed


def assign_triples(g, mention_split):
    """Create Selection based on a mention split and concept entities."""
    # vid: vertex id
    eid2vid = {v.split(":")[0]: k for k, v in g.source.ents.items()}

    cw = set()

    # ow_tails: head is known, tail is query (?, t, r)
    # ow_heads: tail is known, head is query (h, ?, r)
    ow_tails, ow_heads = set(), set()
    removed = set()

    for eid in eid2vid:
        vid = eid2vid[eid]

        # closed world entity
        if eid in mention_split["cw"]:
            cw |= g.find(heads={vid}, tails={vid})

        elif eid in mention_split["ow"]:
            ow_heads |= g.find(heads={vid})
            ow_tails |= g.find(tails={vid})

        else:
            removed.add(eid)

    print("assigned triples")
    print(f"  - cw: {len(cw)}")
    print(f"  - ow_heads: {len(ow_heads)}")
    print(f"  - ow_tails: {len(ow_tails)}")
    print(f"  - removed: {len(removed)}")

    return Selection(
        g=g,
        cw=cw,
        ow_heads=ow_heads,
        ow_tails=ow_tails,
        removed=removed,
        eid2vid=eid2vid,
        mention_split=mention_split,
    )
