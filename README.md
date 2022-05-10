n# Inductive Reasoning with Text - IRT2

TODO

## Installation

Python 3.9 is required.

```bash
pip install .
```

Or with all development dependencies:

```bash
pip install .[dev]
```


### Development Notes

Create a new distribution:

```fish
cd data/irt2/cde
for f in *uniform
    set -l name irt2-cde-(echo $f | cut -d '-' -f 1)
    tar cf ../../dist/$name.tgz --transform "s|.*/|$name/|" $f
end
```
