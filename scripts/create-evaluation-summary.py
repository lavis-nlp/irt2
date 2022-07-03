#!/usr/bin/env python
# -*- coding: utf-8 -*-


import csv
from pathlib import Path

import yaml


def flatten(dic):
    def r(src, tar, trail):
        for k, v in src.items():
            k = f"{trail} {k}" if trail else k

            if isinstance(v, dict):
                r(v, tar, k)
            else:
                tar[k] = v

        return tar

    return r(dic, {}, None)


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
        writer.writerow(flatten(report))
