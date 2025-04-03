#!/usr/bin/env python
# -*- coding: utf-8 -*-


import csv
from pathlib import Path

import yaml
from ktz.collections import dflat

fieldnames = [
    "task",
    "dataset",
    "split",
    "model",
    "macro hits_at_1",
    "macro hits_at_10",
    "macro mrr",
    "micro hits_at_1",
    "micro hits_at_10",
    "micro mrr",
    "filename",
    "date",
]

with Path("data/eval/summary.csv").open(mode="w") as fd:
    writer = csv.DictWriter(fd, fieldnames=fieldnames)

    for fname in Path("data/eval").glob("**/**/*yaml"):
        with fname.open(mode="r") as fd:
            report = yaml.safe_load(fd)

        print(f"{fname}")
        writer.writerow(dflat(report))
