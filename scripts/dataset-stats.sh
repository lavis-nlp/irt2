#!/bin/bash

set -e

# poetry run irt2 -qd load data/blp/umls --loader blp-umls --table
# poetry run irt2 -qd load data/blp/WN18RR --loader blp-wn18rr --table
# poetry run irt2 -qd load data/blp/FB15k-237 --loader blp-fb15k237 --table
# poetry run irt2 -qd load data/blp/Wikidata5M --loader blp-wikidata5m --table

for size in tiny small medium large; do
    poetry run irt2 -qd load data/irt2/irt2-cde-$size --loader irt2 --table
done
