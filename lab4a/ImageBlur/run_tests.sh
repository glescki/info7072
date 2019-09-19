#!/bin/bash
## Script that automaticly run all images tests 10 times

TESTDIR="Tests/"
OUTPUT="$1"

# Check if TESTDIR exists, if not create it
[ ! -d $TESTDIR ] && mkdir -p "$TESTDIR"

if [ ! -f "a.out" ]; then
    echo "ERR: a.out file not found"
    exit 1;
fi

dt=$(date '+%Y-%m-%d %H:%M:%S')

echo "$dt Begin test run.."

for i in $(seq 0 9); do
    ./a.out -i Dataset/$i/input.ppm -o  Dataset/$i/output.ppm | grep "Compute" | cut -d " " -f 2 > $TESTDIR/Dataset$i.dat
    for j in $(seq 1 9); do
        ./a.out -i Dataset/$i/input.ppm -o  Dataset/$i/output.ppm | grep "Compute" | cut -d " " -f 2 >> $TESTDIR/Dataset$i.dat
    done
done

for i in $(seq 0 1); do
    ./a.out -i DatasetW/$i/input.ppm -o  DatasetW/$i/output.ppm | grep "Compute" | cut -d " " -f 2 > $TESTDIR/DatasetW$i.dat
    for j in $(seq 1 9); do
        ./a.out -i DatasetW/$i/input.ppm -o  DatasetW/$i/output.ppm | grep "Compute" | cut -d " " -f 2 >> $TESTDIR/DatasetW$i.dat
    done
done

echo "$dt End test run!"

echo "$dt Calculate mean values.."

cd $TESTDIR

echo "#x y " > $1

for i in $(seq 0 9); do
    mean=$(awk '{s+=$1} END {printf "%.9f\n", s/10}' Dataset$i.dat)
    echo "$i $mean" >> $1
done

for i in $(seq 0 1); do
    mean=$(awk '{s+=$1} END {printf "%.9f\n", s/10}' DatasetW$i.dat)
    echo "$((i + 10)) $mean" >> $1
done

cd ../

echo "$dt End mean run!"

echo "$dt Data file" 
