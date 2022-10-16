# Inductive Reasoning with Text - IRT2

This is the second iteration of the IRT benchmark dataset. This
benchmark offers two challenges: (1) **Ranking** sentences to reveal
hidden entities of interest, (2) and **Linking** sets of sentences
containing new mentions of entities into a knowledge graph (KG). For
training, graphs of varying size are given including (weakly) linked
mentions of these entities. For each entities' mention, a varying size
of sentences is provided to learn the entity-text relation from.

The dataset semantics, creation and uses are detailed in our paper:
[COMING SOON](https://github.com/lavis-nlp/irt2) presented at the
text-mining and generation workshop of the KI2022.


## Installation

Python 3.9 is required.

```bash
pip install irt2
```

Or with all development dependencies:

```bash
pip install irt2[dev]
```

The datasets can be downloaded here: [COMING
SOON](https://github.com/lavis-nlp/irt2)


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

## Cite

If you find our work useful, please give us a cite. You can also
always contact [Felix Hamann](https://github.com/kantholtz) for any
comments or questions!

```bibtex
TO BE ANNOUNCED
```
