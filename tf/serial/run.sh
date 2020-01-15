#!/bin/bash

INPUTS_DIR=../inputs
OUTPUT_FILE=serial.dat

if [ -e "$OUTPUT_FILE" ];
then
    rm $OUTPUT_FILE
fi

for file in $(ls -v "$INPUTS_DIR"/*.dat);
do
    TIME=0

    reads=$(awk 'NR >= 2 {print}' < "$file")
    ref=$(head -n 1 "$file")

    printf 'Running %s\n' "$file"

    while read -r f;
    do
        NWOUT=$(./NeedlemanWunsch $f -r $ref)
        TIME=$(echo $TIME + $NWOUT | bc)

    done < <(sed 1d $file) 

    COUNT=$(wc -l $file)
    MEAN=$(echo "scale=2; $TIME / 10" | bc)
    IFS="_" read -ra ADDR <<< "$file"
    IFS="." read -ra OUT <<< "${ADDR[1]}"
    printf '%s\t%s\n' "${OUT[0]} $MEAN" >> $OUTPUT_FILE;

done


