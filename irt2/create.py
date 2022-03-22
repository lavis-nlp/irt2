# -*- coding: utf-8 -*-

"""Contains the API for ipynb/create.ipynb."""

import irt2
from irt2.dataset import IRT2
from irt2.graph import Graph
from irt2.graph import Relation
from irt2.types import VID, RID, EID, Triple, Mention

from ktz.string import encode_line
from ktz.dataclasses import Index
from ktz.dataclasses import Builder
from ktz.collections import buckets
from ktz.collections import Incrementer
from ktz.filesystem import path as kpath
from ktz.multiprocessing import Relay
from ktz.multiprocessing import Actor

import yaml
import spacy

import re
import csv
import gzip
import random
import sqlite3
import logging
from pathlib import Path
from itertools import islice
from datetime import datetime
from dataclasses import field
from dataclasses import dataclass
from collections import Counter
from collections import defaultdict
from functools import cached_property

from typing import Union
from collections.abc import Iterable


log = logging.getLogger(__name__)


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
        eid: {mention: count for mention,
              count in counts.items() if count >= prune}
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


@dataclass
class Split:
    """Aggregates mention split."""

    graph: Graph
    relations: set[RID]  # only the retained ones

    concept_vertices: set[VID]
    removed_vertices: set[VID] = field(default_factory=set)

    closed_world_mentions: Flat = field(default_factory=set)
    open_world_mentions: Flat = field(default_factory=set)

    @cached_property
    def mentions(self) -> Flat:
        """Return all known mentions."""
        return self.closed_world_mentions | self.open_world_mentions

    # vertex split

    @cached_property
    def vertices(self) -> set[VID]:
        """Return all closed- and open-world vertices."""
        return self.closed_world_vertices | self.open_world_vertices

    @cached_property
    def closed_world_vertices(self) -> set[VID]:
        """Return closed-world vertices."""
        return set(vid for vid, _ in self.closed_world_mentions)

    @cached_property
    def open_world_vertices(self) -> set[VID]:
        """Return open-world vertices."""
        vertices = set(vid for vid, _ in self.open_world_mentions)
        return vertices - self.closed_world_vertices

    # triple split

    def _filter_triples(self, triples: set[Triple]):
        """Remove all triples with removed vertices."""
        # there may be vertices without any mentions
        return {
            (h, t, r)
            for h, t, r in triples
            if h in self.vertices and t in self.vertices and r in self.relations
        }

    @cached_property
    def concept_triples(self) -> set[Triple]:
        """Return all triples where both vertices are concepts."""
        return self.graph.select(
            heads=self.concept_vertices,
            tails=self.concept_vertices,
        )

    @cached_property
    def closed_world_triples(self) -> set[Triple]:
        """Return closed-world triples."""
        return self._filter_triples(
            self.graph.find(
                heads=self.closed_world_vertices,
                tails=self.closed_world_vertices,
            )
        )

    @cached_property
    def open_world_triples(self) -> set[Triple]:
        """Return triples where both vertices are open-world."""
        return self._filter_triples(
            self.graph.select(
                heads=self.open_world_vertices,
                tails=self.open_world_vertices,
            )
        )

    @cached_property
    def open_world_head_triples(self) -> set[Triple]:
        """Return triples where the head vertex is open-world."""
        return self._filter_triples(
            self.graph.find(heads=self.open_world_vertices),
        )

    @cached_property
    def open_world_tail_triples(self) -> set[Triple]:
        """Return triples where the tail vertex is open-world."""
        return self._filter_triples(
            self.graph.find(tails=self.open_world_vertices),
        )

    # --

    def check(self):
        """Run a self-check."""
        assert len(self.concept_vertices) <= len(self.closed_world_vertices)


# select all concept entities
def split_mentions(
    graph: Graph,
    mentions: Mentions,
    seed: int,
    ratio: float,
    concept_rels: list[str],  # manually selected concept relations
    include_rels: list[str],  # upstream names (eids)
    exclude_rels: list[str],  # upstream names (eids)
) -> Split:
    """
    Divide mentions by a given ratio and concepts.

    TODO documentation
    TODO use Builder and remove field(default_factory) because of cached_property

    """
    assert not include_rels and exclude_rels, "mutex!"

    relations = Relation.from_graph(graph)

    # apply include/exclude

    includes = set(include_rels)
    assert len(includes) == len(include_rels)

    excludes = set(exclude_rels)
    assert len(excludes) == len(exclude_rels)

    if includes:
        relations = [rel for rel in relations if rel.name in includes]

    elif excludes:
        relations = [rel for rel in relations if rel.name not in excludes]

    assert relations

    # set aside concept vertices

    concept_rels = set(concept_rels)
    concept_vertices = set.union(
        *[rel.concepts for rel in relations if rel.name in concept_rels]
    )

    assert concept_vertices

    # determine mentions assigend to concepts

    split = Split(
        graph=graph,
        relations={rel.rid for rel in relations},
        concept_vertices=concept_vertices,
    )

    candidates: Flat = set()

    for vid, eid in graph.source.ents.items():

        # map to upstream
        # eid=Q108946:A Few Good Men -> link=Q108946
        link = eid.split(":")[0]
        if link not in mentions.eid2mentions:
            split.removed_vertices.add(vid)
            continue

        # create a flat list to be split later
        flat = {(vid, mention)
                for mention in mentions.eid2mentions[link].keys()}

        if vid in concept_vertices:
            split.closed_world_mentions.update(flat)
        else:
            candidates |= flat

    assert not candidates & split.closed_world_mentions

    # do the split

    random.seed(seed)
    candidates = sorted(candidates)
    random.shuffle(candidates)

    # draw the threshold between all mentions and then set aside the
    # open world mentions. the remaining mentions can be added to the
    # concept mentions and form the closed world mention set.
    total = len(split.closed_world_mentions) + len(candidates)
    threshold = int((1 - ratio) * total)

    assert threshold < len(
        split.closed_world_mentions
    ), "threshold too small: more concept entities exist"

    split.open_world_mentions.update(candidates[:threshold])
    split.closed_world_mentions.update(candidates[threshold:])

    assert (
        not split.open_world_mentions & split.closed_world_mentions
    ), "mentions are shared between open and closed world"
    assert (
        not split.open_world_vertices & split.closed_world_vertices
    ), "vertices are shared between open and closed world"

    return split


REL_SYNONYM = "IRT2:Synonym"


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

    # --- vertices

    # create dict[VID, str]
    build.add(
        vertices={
            vids[old_vid]: split.graph.source.ents[old_vid]
            for old_vid in split.vertices
        },
    )
    vids.freeze()

    # --- relations

    # create dict[RID, str]
    _irt2_rels = {rids[REL_SYNONYM]: REL_SYNONYM}
    _upstream_rels = {
        rids[old_rid]: split.graph.source.rels[old_rid] for old_rid in split.relations
    }

    build.add(relations=_irt2_rels | _upstream_rels)
    rids.freeze()

    # --- mentions

    # create dict[MID, str]
    build.add(
        mentions={
            mids[(old_vid, mention)]: mention for old_vid, mention in split.mentions
        },
    )
    mids.freeze()

    # create dict[VID, set[MID]]
    build.add(
        closed_mentions=buckets(
            col=split.closed_world_mentions,
            key=lambda _, t: (vids[t[0]], mids[t]),
            mapper=set,
        ),
        open_mentions=buckets(
            col=split.open_world_mentions,
            key=lambda _, t: (vids[t[0]], mids[t]),
            mapper=set,
        ),
    )

    # --- triples

    # create set[Triple]
    build.add(
        closed_triples={
            (vids[h], vids[t], rids[r]) for h, t, r in split.closed_world_triples
        },
    )

    # create dict[tuple[MID, RID], VID]
    open_task_heads = dict()
    for h, t, r in split.open_world_head_triples:
        h, t, r = vids[h], vids[t], rids[r]

        for mid in build.get("open_mentions")[h]:
            open_task_heads[(mid, r)] = h

    # create dict[tuple[MID, RID], VID]
    open_task_tails = dict()
    for h, t, r in split.open_world_tail_triples:
        h, t, r = vids[h], vids[t], rids[r]

        for mid in build.get("open_mentions")[t]:
            open_task_heads[(mid, r)] = t

    build.add(
        open_task_heads=open_task_heads,
        open_task_tails=open_task_tails,
    )

    return build()


#
# text sampling
#

# multiprocessing: using own multiprocessing
# because the spacy pipeline n_process was unreliable
# and slow (maybe I did it wrong...?)
#
#   how it works
#
# * dispatcher process creates
# * 1 reader process:
#   - iterate all pages from sqlite db
# * 1 writer process:
#   - open gzipped file
#   - poll on writer queue to write csv lines
# * n worker processes:
#   - process them with spacy to detect sentence boundaries
#   - match the sentences with an Index[Match] instance
#   - send sentences to writer queue
#
#   the writer process _does not_ match mid's yet
#   but builds a split-agnostic file which is then
#   used to select contexts for specific splits and
#   subsamples.

class Reader(Actor):

    db: str

    def __init__(self, db: Union[Path, str]):
        super().__init__()
        self.db = db

    def startup(self):
        log.info(f"[{self.name}] connecting to db {self.db}")
        self.con = sqlite3.connect(str(self.db))

    def shutdown(self):
        self.con.close()

    def loop(self):
        log.info(f"[{self.name}] reading from db")

        cur = self.con.cursor()
        rows = cur.execute(
            "select text, title from pages where text <> ''")
        # for filter(lambda t: t[1] in retained_pages, rows)

        for text, title in rows:
            self.send(text, title)

        cur.close()


# read only data (assuming copy-on-write semantics)
WORKER_MATCHES = None


class Worker(Actor):

    stats: dict[str, int]
    spacy_model: str

    def __init__(self, spacy_model):
        super().__init__()
        self.spacy_model = spacy_model
        self.stats = Counter()

        self._tmp_dir = kpath(irt2.ENV.DIR.DATA / 'tmp', create=True)
        errors = self._tmp_dir / f'{self.name}-errors.gz'
        self._fd_errors = errors.open(mode='wb')

    def startup(self):
        log.info(
            f'[{self.name}] loading spacy pipeline (model={self.spacy_model})')

        self.nlp = spacy.load(
            self.spacy_model,
            exclude=[
                "tagger",
                "scores",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ],
        )

    def shutdown(self):
        log.info(f'[{self.name}] writing stats')

        stats = self._tmp_dir / f'{self.name}-stats.yml'
        with stats.open(mode='w') as fd:
            yaml.safe_dump(dict(self.stats), fd)

        self._fd_errors.close()

    def recv(self, text, title):
        doc = self.nlp(text)
        self.stats['docs handled'] += 1

        # use spacy to detect sentence boundaries
        bounds, sentences = [], []
        for sent in doc.sents:
            if len(sent) <= 1:
                continue

            bounds.append(sent[-1].idx + len(sent[-1]))

            # joined = " ".join(map(str, sent))
            # sentences.append(" ".join(joined.split()))

            sentences.append(str(sent))

        # order matches by sentence boundaries
        matches = sorted(
            WORKER_MATCHES.get(page=title),
            key=lambda match: match.end,
            reverse=True,
        )

        sentences = list(zip(sentences, bounds))[::-1]
        if not sentences:
            self._fd_errors.write(encode_line(
                ('empty document', title), sep='|'))
            self.stats['empty documents'] += 1
            return

        sentence, bound = sentences.pop()

        # do not save multiple matches of a single sentence
        seen = set()
        while matches:
            self.stats['matches handled'] += 1

            match = matches.pop()
            while bound < match.end:
                sentence, bound = sentences.pop()

            uniq = (match.eid, match.mention, sentence)
            if uniq in seen:
                self.stats['duplicate sentences'] += 1
                continue

            if match.mention not in sentence:
                self._fd_errors.write(encode_line(
                    ('not in mention',) + uniq, sep='|'))
                self.stats['match not in sentence'] += 1
                continue

            self.stats['sentences sent'] += 1
            self.send(
                match.eid,
                match.norm_mention,
                match.mention,
                sentence,
            )

            seen.add(uniq)


class Writer(Actor):

    out: Union[str, Path]
    sep: str

    def __init__(self, out, sep):
        super().__init__()
        self.out = out
        self.sep = sep

    def startup(self):
        log.info(f'[{self.name}] opening {self.out}')
        self.fd = gzip.open(str(self.out), mode='wb')

    def shutdown(self):
        self.fd.close()

    def recv(self, *args):
        encoded = encode_line(args, sep=self.sep)
        self.fd.write(encoded)


def get_text(config, matches: Index[Match], procs: int = 1):
    global WORKER_MATCHES
    WORKER_MATCHES = matches

    # read

    reader = Reader(
        db=irt2.ENV.DIR.ROOT / config['source pages'],
    )

    # process

    worker = [
        Worker(spacy_model=config['spacy model'])
        for _ in range(procs)
    ]

    # write

    out = irt2.ENV.DIR.ROOT / config['source sentences']
    kpath(out.parent, create=True)
    writer = Writer(out=out, sep=config['separator'])

    # dispatch multiprocessing

    relay = Relay(maxsize=100)
    relay.connect(reader, worker, writer)
    relay.run()  # blocks
