# -*- coding: utf-8 -*-

"""IRT2 graph abstraction."""


import json
import logging
import pathlib
from collections import defaultdict
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Union

import networkx
import numpy as np
import yaml
from ktz.filesystem import path as kpath

from irt2.types import RID, VID, Head, Tail, Triple

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphImport:
    """

    Unified data definition used by IRT2 for all upstream graphs.

    Graph triples are of the following structure: (head, tail, relation)
    You can provide any Iterable for the triples. They are converted
    to frozenset[triple]

    Currently the graph is defined by it's edges, which means each
    node is at least connected to one other node. This might change in
    the future.

    Order of provided triples is not preserved.

    The rels and ents dictionaries are filled with all missing
    information automatically such that e.g. rels[i] = f'{i}'.
    They cannot be changed afterwards.

    """

    # (head, tail, relation)
    triples: set[Triple]

    rels: dict[RID, str] = field(default_factory=dict)
    ents: dict[VID, str] = field(default_factory=dict)

    # --

    def _set(self, prop: str, *args, **kwargs):
        object.__setattr__(self, prop, *args, **kwargs)

    def _set_all(self, triples, ents, rels):
        self._set("triples", frozenset(triples))
        self._set("ents", dict(ents))
        self._set("rels", dict(rels))

    def _resolve(self, idx: int, mapping: dict[int, str]):
        if idx not in mapping:
            label = str(idx)
            mapping[idx] = label

    def __post_init__(self):  # noqa: D105
        triples = set(map(tuple, self.triples))

        for h, t, r in self.triples:
            self._resolve(h, self.ents)
            self._resolve(t, self.ents)
            self._resolve(r, self.rels)

        self._set_all(triples, self.ents, self.rels)

    # --

    def join(self, other: "GraphImport"):
        """Join two graph imports."""
        ents = {**self.ents, **other.ents}
        rels = {**self.rels, **other.rels}
        triples = self.triples | other.triples

        self._set_all(triples, ents, rels)

    def save(self, path: Union[str, pathlib.Path]):
        """Write graph import to disk."""
        path = kpath(path, create=True)

        with (path / "triples.txt").open(mode="w") as fd:
            fd.writelines(f"{h} {t} {r}\n" for h, t, r in self.triples)

        with (path / "entities.txt").open(mode="w") as fd:
            fd.writelines(f"{e} {name}\n" for e, name in self.ents.items())

        with (path / "relations.txt").open(mode="w") as fd:
            fd.writelines(f"{r} {name}\n" for r, name in self.rels.items())

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]):
        """Load graph import from disk."""
        path = kpath(path, create=True)

        with (path / "triples.txt").open(mode="r") as fd:
            triples: set[Triple] = set()
            for line in fd:
                h, t, r = map(int, line.split())
                triples.add((h, t, r))

        split = partial(str.split, maxsplit=1)

        def _load_dict(fd):
            lines = (split(line) for line in fd)
            return dict((int(i), name.strip()) for i, name in lines)

        with (path / "entities.txt").open(mode="r") as fd:
            ents = _load_dict(fd)

        with (path / "relations.txt").open(mode="r") as fd:
            rels = _load_dict(fd)

        return cls(triples=triples, ents=ents, rels=rels)


class Graph:
    """
    IRT2 Graph Abstraction.

    Create a new graph object which maintains a networkx graph.
    This class serves as a provider of utilities working on
    and for initializing the networkx graph.

    Design Decisions
    ----------------

    Naming:

    Naming nodes and edges: networkx uses "nodes" and "edges". To
    not confuse on which "level" you operate on the graph, everything
    here is called "ents" (for entities) and "rels" (for relations)
    when working with IRT code and "node" and "edges" when working
    with networkx instances.

    Separate Relation and Entitiy -Mapping:

    The reasoning for not providing (e.g.) the Graph.source.rels
    mapping directly on the graph is to avoid a false expectation
    that this is automatically in sync with the graph itself.
    Consider manipulating Graph.g (deleting nodes for example) -
    this would not update the .rels-mapping. Thus this is explicitly
    separated in the .source GraphImport.

    """

    name: str
    source: GraphImport

    nx: networkx.MultiDiGraph
    rnx: networkx.MultiDiGraph

    edges: dict[RID, set[tuple[Head, Tail]]]

    @property
    def description(self) -> str:
        """Verbose description of graph data."""
        s = (
            f"IRT2 GRAPH: {self.name}\n"
            f"  nodes: {self.nx.number_of_nodes()}\n"
            f"  edges: {self.nx.number_of_edges()}"
            f" ({len(self.source.rels)} types)\n"
        )

        try:
            degrees = np.array(list(self.nx.degree()))[:, 1]
            s += (
                f"  degree:\n"
                f"    mean {np.mean(degrees):.2f}\n"
                f"    median {int(np.median(degrees)):d}\n"
            )

        except IndexError:
            s += "  cannot measure degree\n"

        return s

    # --

    def __str__(self) -> str:
        return f"IRT graph: [{self.name}] ({len(self.source.ents)} entities)"

    def __init__(
        self,
        name: str | None = None,
        source: GraphImport | None = None,
    ):
        assert type(name) is str if name is not None else True, f"{name=}"

        # properties
        self.nx = networkx.MultiDiGraph()
        self.edges = defaultdict(set)

        self.name = "unknown" if name is None else name

        # GraphImport
        if source is not None:
            self.source = source
            self.add(source)
        else:
            self.source = GraphImport(triples=set())

        log.debug(f"created graph: \n{self.description}\n")

    # --

    def select(
        self,
        heads: set[VID] | None = None,
        tails: set[VID] | None = None,
        edges: set[RID] | None = None,
    ):
        """
        Select edges from the graph.

        An edge is a triple (h, t, r) and the selection is either
        the union or intersection of all edges containing the
        provided nodes and edges.

        The difference between Graph.find and Graph.select is that
        .find will select any edge containing any of the provided
        heads (union) or tails and .select will only choose those
        where their any combination of all provided entities occurs
        (intersection).

        Parameters
        ----------
        heads : Set[int]
          consider the provided head nodes
        tails : Set[int]
          consider the provided head nodes
        edges : Set[int]
          consider the provided edge classes

        Returns
        -------
        A set of edges adhering to the provided constraints.

        Notes
        -----
        Not using nx.subgraph as it would contain undesired edges
        (because nx.subgraph only works on node-level)

        """
        heads = set() if heads is None else heads
        tails = set() if tails is None else tails
        edges = set() if edges is None else edges

        def _gen(nxg, heads, tails, edges, rev=False):
            for h in heads:
                if h not in nxg:
                    continue

                for t, rs in nxg[h].items():
                    if tails and t not in tails:
                        continue

                    for r in rs:
                        if edges and r not in edges:
                            continue

                        yield (h, t, r) if not rev else (t, h, r)

        dom = set(_gen(self.nx, heads, tails, edges))
        rng = set(_gen(self.rnx, tails, heads, edges, rev=True))

        return dom | rng

    # --

    def find(
        self,
        heads: set[VID] | None = None,
        tails: set[VID] | None = None,
        edges: set[RID] | None = None,
    ) -> set[Triple]:
        """
        Find edges in the graph.

        An edge is a triple (h, t, r) and the selection is either
        the union or intersection of all edges containing one of the
        provided nodes and edges.

        The difference between Graph.find and Graph.select is that
        .find will select any edge containing any of the provided
        heads (union) or tails and .select will only choose those
        where their any combination of all provided entities occurs
        (intersection).

        Parameters
        ----------
        heads : Set[int]
          consider the provided head nodes

        tails : Set[int]
          consider the provided head nodes

        edges : Set[int]
          consider the provided edge classes

        Returns
        -------
        A set of edges adhering to the provided constraints.


        Notes
        -----
        Not using nx.subgraph as it would contain undesired edges
        (because nx.subgraph only works on node-level)

        """
        heads = set() if heads is None else heads
        tails = set() if tails is None else tails
        edges = set() if edges is None else edges

        def _gen(nxg, heads, rev=False):
            for h in heads:
                if h not in nxg:
                    continue

                for t, rs in nxg[h].items():
                    for r in rs:
                        yield (h, t, r) if not rev else (t, h, r)

        dom = set(_gen(self.nx, heads))
        rng = set(_gen(self.rnx, tails, rev=True))
        rel = {(h, t, r) for r in edges or [] for h, t in self.edges[r]}

        return dom | rng | rel

    #
    # --- | EXTERNAL SOURCES
    #

    def add(self, source: GraphImport) -> "Graph":
        """
        Add data to the current graph by using a GraphImport instance.

        Parameters
        ----------
        source : GraphImport
          Data to feed into the graph

        """
        for i, (h, t, r) in enumerate(source.triples):
            self.nx.add_node(h, label=source.ents[h])
            self.nx.add_node(t, label=source.ents[t])
            self.nx.add_edge(h, t, r, label=source.rels[r], rid=r)
            self.edges[r].add((h, t))

        self.source.join(source)
        self.rnx = self.nx.reverse()
        return self

    def save(self, path: Union[str, pathlib.Path]):
        """
        Persist graph to file.

        Parameters
        ----------
        path : Union[str, pathlib.Path]
          Folder to save the graph to

        """
        path = kpath(path, create=True)

        kwargs = dict(name=self.name)
        with (path / "config.yml").open(mode="w") as fd:
            yaml.dump(kwargs, fd)

        self.source.save(path)

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "Graph":
        """
        Load graph from file.

        Parameters
        ----------
        path : Union[str, pathlib.Path]
          File to load graph from

        """
        path = kpath(path, exists=True, message="loading graph from {path_abbrv}")

        with (path / "config.yml").open(mode="r") as fd:
            kwargs = yaml.load(fd, Loader=yaml.FullLoader)

        source = GraphImport.load(path)
        return cls(source=source, **kwargs)

    #
    # ---| SUGAR
    #

    def tabulate_triples(self, triples: Iterable[Triple]):
        """Table representation of triple sets."""
        from tabulate import tabulate

        src = self.source

        rows = [(h, src.ents[h], t, src.ents[t], r, src.rels[r]) for h, t, r in triples]

        return tabulate(rows, headers=("", "head", "", "tail", "", "relation"))

    def str_triple(self, triple: Triple):
        """Create string representation of single triple."""
        h, t, r = triple
        return (
            f"{self.source.ents[h]} | "
            f"{self.source.ents[t]} | "
            f"{self.source.rels[r]}"
        )


@dataclass
class Relation:
    """Determines concept entities."""

    rid: RID
    name: str
    triples: set[Triple]

    heads: set[VID]
    tails: set[VID]

    ratio: float

    def __str__(self):
        return (
            f"{self.name} ({self.rid}): ratio={self.ratio:.5f} "
            f"(heads={len(self.heads)}, tails={len(self.tails)}) "
            f"{len(self.triples)} triples"
        )

    @property
    def concepts(self) -> set[int]:
        """Either head or tail sets (whichever is smaller)."""
        reverse = len(self.heads) <= len(self.tails)
        return self.heads if reverse else self.tails

    @classmethod
    def from_graph(cls, g: Graph) -> list["Relation"]:
        """Create a (sorted) list of Relations based on ratio."""
        rels = []
        for rid, relname in g.source.rels.items():
            triples = g.find(edges={rid})
            assert triples, f"{relname} ({rid=}) has no triples assigned"

            heads, tails = map(set, zip(*((h, t) for h, t, _ in triples)))

            lens = len(heads), len(tails)
            ratio = min(lens) / max(lens)

            rels.append(
                cls(
                    rid=rid,
                    name=relname,
                    triples=triples,
                    heads=heads,
                    tails=tails,
                    ratio=ratio,
                )
            )

        return sorted(rels, key=lambda rel: rel.ratio)


#
#  loader
#


log = logging.getLogger(__name__)


# --- | CODEX IMPORTER
#       https://github.com/tsafavi/codex


def load_codex(
    *f_triples: str,
    f_rel2id: Union[str, pathlib.Path],
    f_ent2id: Union[str, pathlib.Path],
) -> GraphImport:
    """
    Load CoDEx-like benchmark files.

    Structure is as follows:

    f_triples: the graph as (
      head-wikidata-id, relation-wikidata-id, tail-wikidata-id
    ) triples

    f_rel2id: json file containing wikidata-id -> label mappings
    f_ent2id: json file containing wikidata-id -> label mappings

    """
    with kpath(f_rel2id, exists=True).open(mode="r") as fd:
        rel2label = json.load(fd)

    with kpath(f_ent2id, exists=True).open(mode="r") as fd:
        ent2label = json.load(fd)

    triples = set()
    refs = {
        "ents": {"counter": 0, "dic": {}},
        "rels": {"counter": 0, "dic": {}},
    }

    def _get(kind: str, key: str):
        dic = refs[kind]["dic"]

        if key not in dic:
            dic[key] = refs[kind]["counter"]
            refs[kind]["counter"] += 1

        return dic[key]

    for fname in f_triples:
        p_triples = kpath(fname, exists=True)

        with p_triples.open(mode="r") as fd:
            for line in fd:
                gen = zip(("ents", "rels", "ents"), line.strip().split())
                h, r, t = map(lambda a: _get(*a), gen)
                triples.add((h, t, r))  # mind the switch!

    gi = GraphImport(
        triples=triples,
        rels={
            idx: f"{wid}:{rel2label[wid]['label']}"
            for wid, idx in refs["rels"]["dic"].items()
        },
        ents={
            idx: f"{wid}:{ent2label[wid]['label']}"
            for wid, idx in refs["ents"]["dic"].items()
        },
    )

    return gi


# --- | OPEN KE IMPORTER
#       https://github.com/thunlp/OpenKE


def _oke_fn_triples(line: str):
    h, t, r = map(int, line.split())
    return h, t, r


def _oke_fn_idmap(line: str):
    name, idx = line.rsplit(maxsplit=1)
    return int(idx), name.strip()


def _oke_parse(path: str, fn) -> Generator[Any, None, None]:
    if path is None:
        return None

    with open(path, mode="r") as fd:
        fd.readline()
        for _, line in enumerate(fd):
            yield line if fn is None else fn(line)


def load_oke(
    *f_triples: str,
    f_rel2id: str,
    f_ent2id: str,
) -> GraphImport:
    """
    Load OpenKE-like benchmark files.

    Structure is as follows:

    f_triples: the graph as (eid-1, eid-2, rid) triples
    f_rel2id: relation names as (name, rid) tuples
    f_ent2id: entity labels as (label, eid) tuples

    The first line of each file is ignored (contains the number of
    data points in the original data set)

    """
    log.info(f"loading OKE-like graph from {f_triples}")

    triples: set[Triple] = set()
    for fname in f_triples:
        triples |= set(_oke_parse(fname, _oke_fn_triples))

    rels = dict(_oke_parse(f_rel2id, _oke_fn_idmap))
    ents = dict(_oke_parse(f_ent2id, _oke_fn_idmap))

    gi = GraphImport(triples=(triples), rels=rels, ents=ents)

    log.info(f"finished parsing {f_triples}")

    return gi


# --- | VILLMOW IMPORTER
#       https://github.com/villmow/datasets_knowledge_embedding
#       https://gitlab.cs.hs-rm.de/jvill_transfer_group/thesis/thesis


def load_vll(f_triples: list[str]) -> GraphImport:
    """
    Load Villmow's benchmark files.

    Structure is as follows:
    f_triples: the graph encoded as string triples (e1, r, e2)

    """
    log.info(f"loading villmow-like graph from {f_triples}")

    refs = {
        "ents": {"counter": 0, "dic": {}},
        "rels": {"counter": 0, "dic": {}},
    }

    def _get(kind: str, key: str):
        dic = refs[kind]["dic"]

        if key not in dic:
            dic[key] = refs[kind]["counter"]
            refs[kind]["counter"] += 1

        return dic[key]

    triples = set()
    for fname in f_triples:
        with open(fname, mode="r") as fd:
            for line in fd:
                gen = zip(("ents", "rels", "ents"), line.strip().split())
                h, r, t = map(lambda a: _get(*a), gen)
                triples.add((h, t, r))  # mind the switch

    gi = GraphImport(
        triples=triples,
        rels={idx: name for name, idx in refs["rels"]["dic"].items()},
        ents={idx: name for name, idx in refs["ents"]["dic"].items()},
    )

    log.info(f"finished parsing {f_triples}")

    return gi


LOADER = {
    "codex": load_codex,
    "oke": load_oke,
    "vll": load_vll,
}


def load_graph(loader: str, name: str, *args, **kwargs) -> Graph:
    """
    Load a graph from disk.

    See LOADER for defined loaders and their associated
    functions. Look at the specific functions to determine the
    required args and kwargs.

    Parameters
    ----------
    loader : str
        One of the support loaders (codex, oke, ...)
    name : str
        Graph name
    *args : Any
        Passed on to the specific loader
    **kwargs : Any
        Passed on to the specific loader

    Returns
    -------
    Graph
        New Graph instance

    """
    assert loader in LOADER, f"unknown loader: {loader}"
    source = LOADER[loader](*args, **kwargs)
    return Graph(name=name, source=source)
