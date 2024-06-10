#!/usr/bin/bash

# works when folder structure is like this:
#
# with:
#   task=kgc|ranking
#   irt2=CDE-L|CDE-M|CDE-S|CDE-T
#   model=<any>
#   split=validation|test
#


root=$1
if [ ! -d "$root" ]; then
   echo "usage $0 FOLDER"
   exit 2
fi


for path in $root/*; do
    name=$(basename $path)

    task=$(echo $name | cut -d . -f 1)
    irt2=$(echo $name | cut -d . -f 2)
    split=$(echo $name | cut -d . -f 3)

    if [ $task = kgc ]; then
        continue
    fi

    echo
    echo "--------------------"
    echo task=$task irt2=$irt2 split=$split

    for model in $path/*; do
        name=$(basename $model)
        echo name=$name

        in=$model/*.csv
        out=$(echo $in | sed -e 's/csv$/yaml/')

        irt2 evaluate-ranking \
             --irt2 data/irt2/irt2-cde-$irt2 \
             --split $split \
             --model $name \
             --predictions $in \
             --out $out
    done
done
