# Inductive Reasoning with Text - IRT2

[![IRT2 on PyPI](https://img.shields.io/pypi/v/irt2?style=for-the-badge)](https://pypi.org/project/irt2)
[![Code Style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)](https://github.com/psf/black)

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Inductive Reasoning with Text - IRT2](#inductive-reasoning-with-text---irt2)
    - [Download](#download)
    - [Installation](#installation)
    - [Getting started](#getting-started)
        - [Load a Dataset](#load-a-dataset)
        - [Datamodel](#datamodel)
        - [Evaluation](#evaluation)
    - [Cite](#cite)

<!-- markdown-toc end -->


This is the second iteration of the [IRT benchmark
dataset](https://github.com/lavis-nlp/irt2). This benchmark offers two
challenges: (1) **Ranking** sentences to reveal hidden entities of
interest, (2) and **Linking** sets of sentences containing new
mentions of entities into a knowledge graph (KG). For training, graphs
of varying size are given including (weakly) linked mentions of these
entities. For each entities' mention, a varying size of sentences is
provided to learn the entity-text relation from.

The dataset semantics, creation and uses are detailed in our paper:
[IRT2: Inductive Linking and Ranking in Knowledge Graphs of Varying Scale](https://recap.uni-trier.de/static/3be71a52f5af03e34adea05358817516/78.pdf) 
presented at the [Workshop on Text Mining and Generation (TMG)](https://recap.uni-trier.de/2022-tmg-workshop/) of the [KI2022](https://ki2022.gi.de/).

## Download

The datasets can be downloaded here:

* **[Download All](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde.tar.gz)**
* **[Download Tiny](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-tiny.tar.gz)**
* **[Download Small](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-small.tar.gz)**
* **[Download Medium](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-medium.tar.gz)**
* **[Download Large](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-large.tar.gz)**


We used a subset of all texts for testing in the paper. These subsets
can be downloaded here to reproduce the reported results (each archive
contains a readme.txt detailing how the text was sampled from the
original datasets):

| Variant | Ranking                                                                          | Linking                                                                          |
|---------|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Tiny    | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-tiny-ranking.tar.gz)   | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-tiny-linking.tar.gz)   |
| Small   | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-small-ranking.tar.gz)  | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-small-linking.tar.gz)  |
| Medium  | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-medium-ranking.tar.gz) | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-medium-linking.tar.gz) |
| Large   | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-large-ranking.tar.gz)  | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-large-linking.tar.gz)  |


## Installation

Python 3.11 is required.

```bash
poetry install [--with dev]
```


## Getting started

We offer an ipython notebook which details how to access the data. See
`ipynb/load-dataset.ipynb`. This repository offers the code necessary
to load the data and evaluate your models performance. See
[Datamodel](#datamodel) if you do not wish to use our interface.

### Load a Dataset

We offer access to the following datasets using our data interface:

- IRT2 (this repository)
  - Tiny
  - Small
  - Medium
  - Large
- [BLP (Daza et al.)](https://github.com/dfdazac/blp)
  - UMLS
  - WN18RR
  - FB15K237
  - Wikidata5m
- Coming soon (1.1.1) [OW (Shah et al.)](https://github.com/haseebs/OWE)
  - FB15K237


You can load a dataset using the CLI (`irt2 load`) like this:

```console
$ poetry run irt2 load --help

 Usage: irt2 load [OPTIONS] FOLDER

 Load a dataset for inspection.

╭─ Options ────────────────────────────────────────────────────────────────────────────────╮
│ --loader    [irt2|blp-umls|blp-wn18rr|blp-fb15k2  loader to use for foreign datasets     │
│             37|blp-wikidata5m]                                                           │
│ --debug                                           drop into debugger session             │
│ --attach                                          drop into ipython session              │
│ --table                                           print csv table data                   │
│ --help                                            Show this message and exit.            │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
```

For example:

```console
$ poetry run irt2 load data/irt2/irt2-cde-tiny --loader irt2 --attach
[12:17:27] IRT2/CDE-T: 12389 vertices | 5 relations | 23894 mentions               cli.py:74

local variable: 'ds': IRT2/CDE-T: 12389 vertices | 5 relations | 23894 mentions

Python 3.11.6 (main, Oct  8 2023, 05:06:43) [GCC 13.2.0]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.23.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: print(ds.description)

IRT2/CDE-T
created: Tue May 10 17:50:42 2022

  vertices: 12389
  relations: 5
  mentions: 23894

  closed-world
    triples: 2928
    vertices: 956
    mentions: 4590 (~4.801 per vertex)
    contexts: 9100422

  open-world (validation)
    mentions: 8111 (~2.262 per vertex)
    contexts: 850351
    tasks:
      heads: 12036
      tails: 5920

  open-world (test)
    mentions: 13336 (~1.956 per vertex)
    contexts: 6058557
    task:
      heads: 107768
      tails: 34819
```

If you provide the --attach flag, you are dropped into an interactive
IPython session to inspect the dataset on the fly.




### Datamodel

You are not constrained to our interface. If you like, you can also
simply process the data raw. When you open one of the dataset
variants, you are greeted with the follwing structure:


```console
$ tree irt2-cde-large
irt2-cde-large
├── closed.train-contexts.txt.gz
├── closed.train-mentions.txt
├── closed.train-triples.txt
├── config.yaml
├── log.txt
├── mentions.txt
├── open.test-contexts.txt.gz
├── open.test-head.txt
├── open.test-mentions.txt
├── open.test-tail.txt
├── open.validation-contexts.txt.gz
├── open.validation-head.txt
├── open.validation-mentions.txt
├── open.validation-tail.txt
├── relations.txt
└── vertices.txt
```

The `mentions.txt`, `relations.txt`, and `vertices.txt` contain the
respective ids and human readable names.

```console
$ head -n 5 vertices.txt
# unique vertex identifier
# vertex id:vid | name:str
0|Q108946:A Few Good Men
1|Q39792:Jack Nicholson
2|Q1041:Senegal
```

All other files then use the respective ids. For example, to see the
known vertex-mention mapping for the closed-world data, see the
`closed.train-mentions.txt`:

```console
# {kind}-world mentions (train)
# vertex id:vid | mention id: mid
1589|0
1589|6912
1589|1230
```

The associated text is found in the respective (gzipped) context
files:

```console
$ zcat closed.train-contexts.txt.gz | head -n 5
9805|United States|Alabama River|Its length as measured by the United States Geological Survey is ,U.S. Geological Survey.
9805|United States|Alabama River|Documented by Europeans first in 1701, the Alabama, Coosa, and Tallapoosa rivers were central to the homeland of the Creek Indians before their removal by United States forces to the Indian Territory in the 1830s.
20947|Theology|Alain de Lille|Alain spent many years as a professor of Theology at the University of Paris and he attended the Lateran Council in 1179.
360|University of Paris|Alain de Lille|Alain spent many years as a professor of Theology at the University of Paris and he attended the Lateran Council in 1179.
19913|France|Alain de Lille|Though the only accounts of his lectures seem to show a sort of eccentric style and approach, he was said to have been good friends with many other masters at the school in Paris, and taught there, as well as some time in southern France, into his old age.
```

Each line contains the mention id (`MID`), the originating Wikipedia
page and the text context. For all IRT2 datasets, we always assert
that the mention which is associated with the mention id can be found
literally in the provided sentence. Mention ids (MID) are always
uniquely assigned to vertices (VID), even if they have the same
phrase.


### Evaluation

To run an evaluation you can simply produce a csv file and invoke the IRT2 cli:

```csv
mention id, relation id, y_0, s(y_0), y_1, s(y_1)
...
```
Where y0 and s(y0) are the first prediction and associated score
respectively. Predictions can be provided in any order, they are
sorted by the evaluation script. Both the ranking
(`irt2 evaluate-ranking`) and the linking (`irt2 evaluate-kgc`)
tasks can be evaluated.


```console
$ poetry run irt2 evaluate-kgc --help

 Usage: irt2 evaluate-kgc [OPTIONS]

 Evaluate the open-world ranking task.
 It is possible to provide gzipped files: Just make sure the file suffix is *.gz.

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --head-task    TEXT                                                    all predictions from the head task [required]            │
│ *  --tail-task    TEXT                                                    all predictions from the tail task [required]            │
│ *  --irt2         TEXT                                                    path to irt2 data [required]                             │
│    --loader       [irt2|blp-umls|blp-wn18rr|blp-fb15k237|blp-wikidata5m]  loader to use for foreign datasets                       │
│ *  --split        TEXT                                                    one of validation, test [required]                       │
│    --max-rank     INTEGER                                                 only consider the first n ranks (target filtered)        │
│    --model        TEXT                                                    optional name of the model                               │
│    --out          TEXT                                                    optional output file for metrics                         │
│    --help                                                                 Show this message and exit.                              │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```


## Cite

If you find our work useful, please consider giving us a cite. You can
also always contact [Felix Hamann](https://github.com/kantholtz) for
any comments or questions!


```bibtex
@article{hamann2023irt2,
	title        = {Irt2: Inductive Linking and Ranking in Knowledge Graphs of Varying Scale},
	author       = {Hamann, Felix and Ulges, Adrian and Falk, Maurice},
	year         = 2023,
	journal      = {arXiv preprint arXiv:2301.00716}
}
```
