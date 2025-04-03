#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract sentences using matches.

Using own multiprocessing
because the spacy pipeline n_process was unreliable
and slow (maybe I did it wrong...?)

  how it works

* dispatcher process creates
* 1 reader process:
  - iterate all pages from sqlite db
* n worker processes:
  - process them with spacy to detect sentence boundaries
  - match the sentences with an Index[Match] instance
  - send sentences to writer queue
* 1 writer process:
  - maintain gzipped file
  - write csv lines sent by the worker processes

  the writer process _does not_ match mid's yet
  but builds a split-agnostic file which is then
  used to select contexts for specific splits and
  subsamples.

"""


import irt2
from irt2.create import Match
from irt2.create import index_matches

from ktz.string import args_hash
from ktz.string import encode_line
from ktz.functools import Cascade
from ktz.dataclasses import Index
from ktz.multiprocessing import Relay
from ktz.multiprocessing import Actor
from ktz.multiprocessing import Handler
from ktz.multiprocessing import Control
from ktz.filesystem import path as kpath

import yaml
import spacy
from tqdm import tqdm as _tqdm

import enum
import gzip
import sqlite3
import logging
import logging.config
import argparse
from pathlib import Path
from itertools import islice
import multiprocessing as mp
from datetime import datetime
from functools import partial
from collections import Counter

from typing import Union
from typing import Optional
from collections.abc import Iterable


tqdm = partial(_tqdm, ncols=80)
log = logging.getLogger(__name__)


class Stats(enum.Enum):
    """
    Identifier for statistics communication.

    See Base.incr and TQDMHandler for more details.
    Each identifier value must be of form: prefix/unit
    """

    reader_docs = "reader/docs"
    reader_batches = "reader/batches"

    worker_batches = "worker/batches"
    worker_sentences = "worker/sentences"
    worker_errors = "worker/errors"

    writer_batches = "writer/batches"


class Base(Actor):
    """Base class shared by all pipeline actors."""

    statq: mp.Queue  # provide updates to the Handler

    def __init__(self, statq: mp.Queue, *args, **kwargs):  # noqa: D107
        super().__init__(*args, **kwargs)
        self.statq = statq

    def incr(self, identifier: enum.Enum, n: int = 1):
        """Message to update progress bar."""
        self.statq.put((identifier.value, n))


class Reader(Base):
    """Read Wikipedia pages from a matches.db."""

    db: str
    batchsize: int
    limit: Optional[int]

    def __init__(
        self,
        db: Union[Path, str],
        batchsize: int,
        limit: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Create Reader.

        Parameters
        ----------
        db : Union[Path, str]
            sqlite3 matches database

        limit : Optional[int]
            Only iterate N documents
        """
        super().__init__(*args, **kwargs)
        self.db = db
        self.limit = limit
        self.batchsize = batchsize

    def startup(self):
        """Before loop."""
        self.log(f"connecting to db {self.db}")
        self.con = sqlite3.connect(str(self.db))

    def shutdown(self):
        """After loop."""
        self.con.close()

    def loop(self):
        """Iterate database."""
        self.log("reading from db")

        cur = self.con.cursor()

        query = "select text, title from pages where text <> ''"
        if self.limit is not None:
            query += f" limit {self.limit}"

        rows = cur.execute(query)
        while True:

            batch = islice(rows, self.batchsize)
            batch = tuple((text, title) for text, title in batch)

            if not batch:
                break

            self.send(batch)

            self.incr(Stats.reader_docs, n=len(batch))
            self.incr(Stats.reader_batches)

        cur.close()


# read only data assuming copy-on-write semantics
# ...only works with fork() on unix
WORKER_MATCHES = None


class Worker(Base):
    """Extract sentences with matches."""

    # eliminates around 0.0016% of all data associated with cde
    SENTENCE_MAX_LEN = 1000

    sep: str
    stats: dict[str, int]
    spacy_model: str

    # set in own process (see startup())
    nlp: spacy.language.Language

    def __init__(self, spacy_model: str, sep: str, *args, **kwargs):
        """
        Create Worker.

        Parameters
        ----------
        spacy_model : str
            Something like en_core_web_lg
        sep : str
            CSV separator

        """
        super().__init__(*args, **kwargs)
        self.spacy_model = spacy_model
        self.stats = Counter()
        self.sep = sep

    def startup(self):
        """Before loop."""
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

    def handle_doc(self, doc: spacy.tokens.Doc, title: str):
        """
        Extract matched sentences from document.

        Parameters
        ----------
        doc : spacy.Doc
        title : str

        """
        bounds, sentences = [], []
        for sent in doc.sents:
            if len(sent) <= 1:
                continue

            bounds.append(sent[-1].idx + len(sent[-1]))
            sentences.append(str(sent).strip())

        # order matches by sentence boundaries
        matches = sorted(
            WORKER_MATCHES.get(page=title),
            key=lambda match: match.end,
            reverse=True,
        )

        sentences = list(zip(sentences, bounds))[::-1]
        if not sentences:
            return

        sentence, bound = sentences.pop()

        # do not save multiple matches of a single sentence
        seen = set()
        while matches:
            match = matches.pop()
            while bound < match.end:

                # TODO curious case, abort processing
                if not sentences:
                    self.incr(Stats.worker_errors)
                    return

                sentence, bound = sentences.pop()

            if match.mention not in sentence:
                self.incr(Stats.worker_errors)
                continue

            if len(sentence) > Worker.SENTENCE_MAX_LEN:
                continue

            line = match.eid, title, match.norm_mention, match.mention, sentence
            if line in seen:
                continue

            seen.add(line)
            yield line

    def recv(self, batch):
        """Extract sentences.

        Parameters
        ----------
        text : str
            Document to parse.
        title : str
            Identifier (EID)

        """
        docs = self.nlp.pipe(iter(batch), as_tuples=True)
        gen = (self.handle_doc(doc, title) for doc, title in docs)

        blob = b""
        for lines in map(tuple, gen):
            blob += b"".join(encode_line(line, self.sep) for line in lines)
            self.incr(Stats.worker_sentences, len(lines))

        self.send(blob)
        self.incr(Stats.worker_batches, 1)


class Writer(Base):
    """Writes blobs of sentences to gzipped file."""

    out: Union[str, Path]

    def __init__(self, out: str, *args, **kwargs):
        """Create Writer.

        Parameters
        ----------
        out : str
            Filename of gzipped csv

        """
        super().__init__(*args, **kwargs)
        self.out = out

    def startup(self):
        """Before loop."""
        self.log(f"opening {self.out}")
        self.fd = gzip.open(str(self.out), mode="wb")

    def shutdown(self):
        """After loop."""
        self.log(f"closing {self.fd.name}")
        self.fd.close()
        self.statq.put(Control.eol)

    def recv(self, encoded):
        """Write blob."""
        self.fd.write(encoded)
        self.incr(Stats.writer_batches)


class TQDMHandler(Handler):
    """Simple handler updating progress bars per group."""

    groups: dict

    def __init__(self, groups: Iterable[dict]):
        """Create Handler."""
        super().__init__()

        print("\n  IRT2 - SAMPLING\n")
        self.groups = {}
        for position, (name, val) in enumerate(groups.items(), 2):
            group, unit = name.split("/")

            self.groups[name] = tqdm(
                desc=f"{group} ",
                unit=f" {unit}",
                total=val.get("total", None),
            )

    def handle(self, stat: str, inc: int):
        """Update tqdm bar."""
        self.groups[stat].update(inc)

    def run(self):
        """While stats queue is active, update counter."""
        while True:
            msg = self.q.get()
            if msg == Control.eol:
                break

            self.handle(*msg)

    @classmethod
    def create(Self, db, config, batchsize, limit: Optional[int] = None):
        """Create and initialize a new TQDMHandler."""
        if not limit:
            con = sqlite3.connect(str(db))
            with con as cur:
                res = cur.execute("select count(*) from pages where text <> ''")
                limit = list(res)[0][0]

        groups = {key.value: {} for key in Stats}

        groups[Stats.reader_docs.value]["total"] = limit
        groups[Stats.reader_batches.value]["total"] = limit // batchsize

        return Self(groups=groups)


def get_text(
    config,
    matches: Index[Match],
    procs: int = 1,
    maxsize: Optional[int] = None,
    batchsize: Optional[int] = None,
    # used primarily for testing
    limit: Optional[int] = None,
    out: Optional[str] = None,
):
    """
    Start pipeline.

    Parameters
    ----------
    config : Config dictionary.
    matches : Index[Match]
        Retrieve match objects by EID
    procs : int
        Amount of concurrent worker processes
    maxsize : Optional[int]
        Applies backpressure
    limit : int
        Only process the first N documents
    out : str
        Overwrite 'source sentences' configuration option

    """
    global WORKER_MATCHES
    WORKER_MATCHES = matches

    maxsize = maxsize or 1000
    batchsize = batchsize or 100

    # real-time status updates

    db = irt2.ENV.DIR.ROOT / config["source pages"]
    handler = TQDMHandler.create(
        db=db,
        config=config,
        batchsize=batchsize,
        limit=limit,
    )

    # read

    reader = Reader(
        db=db,
        statq=handler.q,
        limit=limit,
        batchsize=batchsize,
    )

    # process

    worker = [
        Worker(
            spacy_model=config["spacy model"],
            statq=handler.q,
            sep=config["separator"],
        )
        for _ in range(procs)
    ]

    # write

    out = kpath(out) if out else irt2.ENV.DIR.ROOT / config["source sentences"]
    kpath(out.parent, create=True)
    writer = Writer(
        out=out,
        statq=handler.q,
    )

    # dispatch multiprocessing

    relay = Relay(maxsize=maxsize, log="irt2")
    relay.connect(reader, worker, writer)

    log.info("starting relay")
    relay.start(handler=handler)  # blocks
    log.info("finished relay")


def get_matches(config: dict):
    """Get Index[Match] (from cache)."""
    hash = args_hash(config)
    run = Cascade(
        path=irt2.ENV.DIR.CACHE,
        matches=f"create.ipynb-{hash}-matches",
    )

    @run.cache("matches")
    def load_matches():
        return index_matches(
            irt2.ENV.DIR.ROOT / config["source matches"],
            None,
        )

    load_matches()
    matches = run.get("matches")

    assert matches
    return matches


def main(args):
    """Bootstrap the pipeline."""
    with (irt2.ENV.DIR.CONF / "logging.yaml").open(mode="r") as fd:
        conf = yaml.safe_load(fd)

        # conf['handlers']['stdout']['formatter'] = 'plain'
        # conf["loggers"]["root"]["handlers"] = ["stdout"]
        # conf["loggers"]["ktz"] = {"loggers": ["root"]}
        conf["loggers"]["root"]["level"] = "DEBUG"

        logging.config.dictConfig(conf)

    log = logging.getLogger("irt2")
    log.info("create-text: sampling sentences from matches database")

    with (irt2.ENV.DIR.CONF / "create" / "cde-l.yaml").open(mode="r") as fd:
        config = yaml.safe_load(fd)

    matches = get_matches(config)

    start = datetime.now()
    log.info(f"starting processing at {start}")

    get_text(
        config,
        matches,
        procs=args.processes,
        maxsize=args.maxsize,
        batchsize=args.batchsize,
        limit=args.limit,
        out=args.out,
    )

    end = datetime.now()
    log.info(f"finished processing at {end}")
    log.info(f"total processing time: {end - start}")


def parse_args():
    """Configure and read CL args."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processes",
        type=int,
        required=True,
        help="number of worker processes to spawn",
    )

    parser.add_argument(
        "--maxsize",
        type=int,
        required=False,
        help="maximum queue items between processes",
    )

    parser.add_argument(
        "--batchsize",
        type=int,
        required=False,
        help="documents are processed in batches",
    )

    parser.add_argument(
        "--limit",
        type=int,
        required=False,
        help="only process the first N documents",
    )

    parser.add_argument(
        "--out",
        type=str,
        required=False,
        help="overwrite 'source sentences' configuration option",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
