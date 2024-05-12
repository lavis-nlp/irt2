#!/bin/bash

set -e

load="poetry run irt2 -qd load"

$load data/blp/umls --loader blp/umls --table
$load data/blp/WN18RR --loader blp/wn18rr --table
$load data/blp/FB15k-237 --loader blp/fb15k237 --table
$load data/blp/Wikidata5M --loader blp/wikidata5m --table

for size in tiny small medium large; do
    $load data/irt2/irt2-cde-$size --loader irt2 --table
done
