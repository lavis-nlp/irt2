# Inductive Reasoning with Text - IRT2

[![IRT2 on PyPI](https://img.shields.io/pypi/v/irt2?style=for-the-badge)](https://pypi.org/project/irt2)
[![Code Style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)](https://github.com/psf/black)

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


We used a subset of all texts for testing in the paper. These subsets can be
downloaded here to reproduce the reported results (each archive contains a readme.txt 
detailing how the text was sampled from the original datasets):

| Variant | Ranking  | Linking |
|---------|-|-|
| Tiny | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-tiny-ranking.tar.gz) | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-tiny-linking.tar.gz) |
| Small | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-small-ranking.tar.gz) | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-small-linking.tar.gz) |
| Medium | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-medium-ranking.tar.gz) | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-medium-linking.tar.gz) |
| Large | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-large-ranking.tar.gz) | [Download](http://lavis.cs.hs-rm.de/storage/irt2/irt2-cde-large-linking.tar.gz) |


## Installation

Python 3.9 is required.

```bash
pip install irt2
```

Or with all development dependencies:

```bash
pip install irt2[dev]
```

## Getting started

We offer an ipython notebook which details how to access the data. See
`ipynb/load-dataset.ipynb`. This repository offers the code necessary
to load the data and evaluate your models performance. However, you
can also simply process the data as you like as we tried to make it as
accessible as possible:

### Datamodel

When you open one of the dataset variants, you are greeted with the
follwing structure:

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
page and the text context. We always assert that the mention which is
associated with the mention id can be found literally in the provided
sentence.


### Evaluation

To run an evaluation you can simply produce a csv file and invoke the IRT2 cli:

```console
$ irt2 --help
Usage: irt2 [OPTIONS] COMMAND [ARGS]...

  Use irt2m from the command line.

Options:
  --help  Show this message and exit.

Commands:
  evaluate-kgc      Evaluate the open-world ranking task.
  evaluate-ranking  Evaluate the open-world ranking task.
```


You may either evaluate the KGC or the ranking objectives:

```
 $ irt2 evaluate-kgc --help

              ┌─────────────────────────────┐
              │ IRT2 COMMAND LINE INTERFACE │
              └─────────────────────────────┘

Usage: irt2 evaluate-kgc [OPTIONS]

  Evaluate the open-world ranking task.

  It is possible to provide gzipped files: Just make sure the file suffix is
  *.gz.

Options:
  --head-task TEXT    all predictions from the head task  [required]
  --tail-task TEXT    all predictions from the tail task  [required]
  --irt2 TEXT         path to irt2 data  [required]
  --split TEXT        one of validation, test  [required]
  --max-rank INTEGER  only consider the first n ranks (target filtered)
  --model TEXT        optional name of the model
  --out TEXT          optional output file for metrics
  --help              Show this message and exit.
```

The head and tail tasks are directly derived from the IRT2
dataset. For example, consider the tail task for KGC (given mentions
and a relation, predict suitable tail entities). A csv file for this
task must contain the following rows (we have some assertions in the
code to help out if the provided data is not suitable):


```csv
mention id, relation id, y_0, s(y_0), y_1, s(y_1)
...
```

See [irt2/evaluation.py#L204](irt2/evaluation.py#L204) where the files
are unpacked for more information.


## Cite

If you find our work useful, please give us a cite. You can also
always contact [Felix Hamann](https://github.com/kantholtz) for any
comments or questions!


```bibtex
@article{hamann2023irt2,
	title        = {Irt2: Inductive Linking and Ranking in Knowledge Graphs of Varying Scale},
	author       = {Hamann, Felix and Ulges, Adrian and Falk, Maurice},
	year         = 2023,
	journal      = {arXiv preprint arXiv:2301.00716}
}
```
