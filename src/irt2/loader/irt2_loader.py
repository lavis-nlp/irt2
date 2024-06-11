import gzip
import logging
from functools import partial
from pathlib import Path
from typing import Callable, Literal

import irt2
import yaml
from irt2.dataset import IRT2
from irt2.types import MID, VID, Context, ContextGenerator, IDMap, Sample, Split, Triple
from ktz.collections import buckets
from ktz.dataclasses import Builder
from ktz.filesystem import path as kpath
from ktz.string import decode_line

log = logging.getLogger(__name__)
tee = irt2.tee(log)


def _open(ctx):
    with ctx as fd:
        yield from filter(lambda bs: bs[0] != ord("#"), fd)


def _fopen(path):
    """Open regular file, read binary and skip comments."""
    return _open(kpath(path, is_file=True).open(mode="rb"))


def _gopen(path):
    """Open gzipped file, read binary and skip comments."""
    path = str(kpath(path, is_file=True))
    return _open(gzip.open(path, mode="rb"))


def text_lazy(
    mapping: dict[Split, Path],
    seperator: str,
) -> Callable[[Split], ContextGenerator]:
    assert all(path for path in mapping.values())

    def wrapped(split: Split) -> ContextGenerator:
        gen = _gopen(mapping[split])
        yield (Context.from_line(line, sep=seperator) for line in gen)

    return wrapped


def text_eager(
    mapping: dict[Split, list[Context]]
) -> Callable[[Split], ContextGenerator]:
    def wrapped(split: Split) -> ContextGenerator:
        yield (ctx for ctx in mapping[split])

    return wrapped


def load_irt2(
    path: Path | str,
    mode: Literal["original", "full"] = "original",
):
    assert mode in {"original", "full"}

    path = Path(path)

    build = Builder(IRT2)
    build.add(path=path)

    with (path / "config.yaml").open(mode="r") as fd:
        config = yaml.safe_load(fd)
        build.add(config=config)

    decode = partial(decode_line, sep=config["create"]["separator"])
    ints = partial(decode, fn=int)

    # -- ids

    def load_ids(fname) -> dict[int, str]:
        pairs = partial(decode, fns=(int, str))
        return dict(map(pairs, _fopen(path / fname)))  # type: ignore FIXME upstream

    idmap = IDMap()
    build.add(idmap=idmap)

    idmap.vid2str = load_ids("vertices.txt")
    idmap.rid2str = load_ids("relations.txt")
    idmap.mid2str = load_ids("mentions.txt")

    # -- triples

    def load_triples(fname) -> set[Triple]:
        return set(map(ints, _fopen(path / fname)))  # type: ignore FIXME upstream

    build.add(closed_triples=load_triples("closed.train-triples.txt"))

    # -- mentions

    def load_mentions(fname, split: Split):
        items = map(ints, _fopen(path / fname))

        agg: dict[VID, set[MID]]
        agg = buckets(col=items, mapper=set)  # type: ignore FIXME upstream

        for vid, mids in agg.items():
            idmap.vid2mids[split][vid] = mids

    load_mentions("closed.train-mentions.txt", Split.train)
    load_mentions("open.validation-mentions.txt", Split.valid)
    load_mentions("open.test-mentions.txt", Split.test)

    assert set.union(*(set(d) for d in idmap.vid2mids.values())) == set(idmap.vid2str)

    # -- open-world samples

    cw_vids = {v for h, t, _ in build.get("closed_triples") for v in (h, t)}

    def load_ow(fname, split: Split) -> set[Sample]:
        samples: set[Sample]
        samples = set(map(ints, _fopen(path / fname)))  # type: ignore FIXME upstream

        filtered = {(m, r, v) for m, r, v in samples if mode == "full" or v in cw_vids}

        # --

        log.info(f"loaded {len(filtered)}/{len(samples)} samples for split {split}")
        return filtered

    tee(f"loading open world data using {mode.upper()} mode!")

    build.add(
        _text_loader=text_lazy(
            mapping={
                Split.train: path / f"closed.train-contexts.txt.gz",
                Split.valid: path / f"open.validation-contexts.txt.gz",
                Split.test: path / f"open.test-contexts.txt.gz",
            },
            seperator=config["create"]["separator"],
        ),
        _val_heads=load_ow("open.validation-head.txt", Split.valid),
        _val_tails=load_ow("open.validation-tail.txt", Split.valid),
        _test_heads=load_ow("open.test-head.txt", Split.test),
        _test_tails=load_ow("open.test-tail.txt", Split.test),
    )

    return build()
