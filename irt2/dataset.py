import logging
import textwrap
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from itertools import combinations
from pathlib import Path
from typing import Callable, Literal, Union

from ktz.filesystem import path as kpath

import irt2
from irt2.graph import Graph, GraphImport, Relation
from irt2.types import MID, RID, VID, ContextGenerator, IDMap, Sample, Split, Triple

log = logging.getLogger(__name__)
tee = irt2.tee(log)


@dataclass
class IRT2:
    """IRT2 data collection."""

    path: Path
    config: dict

    idmap: IDMap
    closed_triples: set[Triple]

    # internally used

    _text_loader: Callable[[Split], ContextGenerator]

    _val_heads: set[Sample]
    _val_tails: set[Sample]

    _test_heads: set[Sample]
    _test_tails: set[Sample]

    # --- convenience/aliases

    @property
    def vertices(self) -> dict[VID, str]:
        return self.idmap.vid2str

    @property
    def relations(self) -> dict[RID, str]:
        return self.idmap.rid2str

    @property
    def mentions(self) -> dict[MID, str]:
        return self.idmap.mid2str

    def _split_mentions(self, split: Split):
        return {vid: self.idmap.vid2mids[vid] for vid in self.idmap.split2vids[split]}

    @cached_property
    def closed_mentions(self) -> dict[VID, set[MID]]:
        return self._split_mentions(Split.train)

    @cached_property
    def open_mentions_val(self) -> dict[VID, set[MID]]:
        return self._split_mentions(Split.valid)

    @cached_property
    def open_mentions_test(self) -> dict[VID, set[MID]]:
        return self._split_mentions(Split.test)

    @property
    def name(self) -> str:
        """Get the dataset name - (e.g. IRT2/CDE-L)."""
        return self.config["create"]["name"]

    @cached_property
    def closed_vertices(self):
        """Closed-world vertices seen at training time."""
        return self.idmap.split2vids[Split.train]

    # tasks

    def task_filtered(
        self,
        prop: Literal["transductive", "semi-inductive", "fully-inductive"],
        split: Split,
        samples: set[Sample],
    ):
        train, valid = (
            self.idmap.split2vids[Split.train],
            self.idmap.split2vids[Split.valid],
        )

        assert split in {Split.valid, Split.test}
        ref = train if split == Split.valid else train | valid

        c_tr = lambda ref, v1, v2: (v1 in ref) & (v2 in ref)
        c_si = lambda ref, v1, v2: (v1 in ref) ^ (v2 in ref)
        c_fi = lambda ref, v1, v2: (v1 not in ref) & (v2 not in ref)

        cond = {
            "transductive": c_tr,
            "semi-inductive": c_si,
            "fully-inductive": c_fi,
        }[prop]

        prod = (
            (mid, v1, rid, v2)
            for mid, rid, v2 in samples
            for v1 in self.idmap.mid2vids.get(mid, set())
        )

        return {(mid, rid, v2) for mid, v1, rid, v2 in prod if cond(ref, v1, v2)}

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
        return self._open_kgc(self._val_tails)

    @cached_property
    def open_kgc_val_tails(self) -> dict[tuple[MID, RID], set[VID]]:
        """Get the kgc validation task."""
        return self._open_kgc(self._val_heads)

    @cached_property
    def open_kgc_test_heads(self) -> dict[tuple[MID, RID], set[VID]]:
        """Get the kgc test task."""
        return self._open_kgc(self._test_tails)

    @cached_property
    def open_kgc_test_tails(self) -> dict[tuple[MID, RID], set[VID]]:
        """Get the kgc test task."""
        return self._open_kgc(self._test_heads)

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
        return self._open_ranking(self._val_heads)

    @cached_property
    def open_ranking_val_tails(self) -> dict[tuple[VID, RID], set[MID]]:
        """Get the ranking validation tails task."""
        return self._open_ranking(self._val_tails)

    @cached_property
    def open_ranking_test_heads(self) -> dict[tuple[VID, RID], set[MID]]:
        """Get the ranking test heads task."""
        return self._open_ranking(self._test_heads)

    @cached_property
    def open_ranking_test_tails(self) -> dict[tuple[VID, RID], set[MID]]:
        """Get the ranking test tails task."""
        return self._open_ranking(self._test_tails)

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

    @staticmethod
    def from_dir(path: Union[str, Path]):
        """Load the dataset from a directory.

        Parameters
        ----------
        path : Path
            where to load the data from

        """
        from irt2.loader.irt2 import load_irt2

        return load_irt2(path=kpath(path, is_dir=True))

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

    # query tools

    def find_by_mention(
        self,
        query: str,
        splits: tuple[Split, ...] = (Split.train,),
    ) -> set[VID]:
        ...

    # -- only pretty output and statistics ahead

    table_header = (
        "name",
        "created",
        #
        "total vertices",
        "total relations",
        "total mentions",
        #
        "closed world triples",
        "closed world vertices",
        "closed world mentions",
        "closed world text contexts",
        #
        "open world validation head tasks",
        "open world validation head tasks transductive",
        "open world validation head tasks semi-inductive",
        "open world validation head tasks fully-inductive",
        "open world validation tail tasks",
        "open world validation tail tasks transductive",
        "open world validation tail tasks semi-inductive",
        "open world validation tail tasks fully-inductive",
        "open world validation mentions",
        "open world validation contexts",
        #
        "open world test head tasks",
        "open world test head tasks transductive",
        "open world test head tasks semi-inductive",
        "open world test head tasks fully-inductive",
        "open world test tail tasks",
        "open world test tail tasks transductive",
        "open world test tail tasks semi-inductive",
        "open world test tail tasks fully-inductive",
        "open world test mentions",
        "open world test contexts",
    )

    @cached_property
    def table_row(self):
        cfg = self.config["create"]
        row = (
            cfg["name"],
            self.config["created"],
            # total
            len(self.vertices),
            len(self.relations),
            len(self.mentions),
            # closed world
            len(self.closed_triples),
            len(self.closed_vertices),
            len({mid for mids in self.closed_mentions.values() for mid in mids}),
            self._contexts_count(self.closed_contexts),
        )

        it = (
            (
                Split.valid,
                (self._val_heads, self._val_tails),
                self.open_mentions_val,
                self.open_contexts_val,
            ),
            (
                Split.test,
                (self._test_heads, self._test_tails),
                self.open_mentions_test,
                self.open_contexts_test,
            ),
        )

        for split, samples_ht, mentions, contexts in it:
            for samples in samples_ht:
                row += (len(samples),)
                for prop in "transductive", "semi-inductive", "fully-inductive":
                    row += (len(self.task_filtered(prop, split, samples)),)

            row += (
                len({mid for mids in mentions.values() for mid in mids}),
                self._contexts_count(contexts),
            )

        # open world rest
        assert len(row) == len(
            self.table_header
        ), f"{len(row)=} == {len(self.table_header)=}"
        return row

    def _contexts_count(self, mgr) -> int:
        return -1
        with mgr() as contexts:
            return sum(1 for _ in contexts)

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

        body = textwrap.indent(
            textwrap.dedent(
                f"""
                vertices: {len(self.vertices)}
                relations: {len(self.relations)}
                mentions: {len(self.mentions)}

                closed-world
                  triples: {len(self.closed_triples)}
                  vertices: {len(self.closed_vertices)}
                  mentions: {mentions(self.closed_mentions)}
                  contexts: {self._contexts_count(self.closed_contexts)}

                open-world (validation)
                  mentions: {mentions(self.open_mentions_val)}
                  contexts: {self._contexts_count(self.open_contexts_val)}
                  tasks:
                    heads: {len(self._val_heads)}
                    tails: {len(self._val_tails)}

                open-world (test)
                  mentions: {mentions(self.open_mentions_test)}
                  contexts: {self._contexts_count(self.open_contexts_test)}
                  task:
                    heads: {len(self._test_heads)}
                    tails: {len(self._test_tails)}
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
