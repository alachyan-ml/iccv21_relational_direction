#!/bin/bash

dir = extract_features/epic_kitchens
touch "$dir"/logs/sections_train.txt
if [ $# -eq 0 ]
then
    for i in { 0 .. 134 }
    do
        section = $i
        printf "Section: %s\n" $section >> "$dir"/logs/sections_train.txt
        python "$dir"/epic_extract_feature_map_ResNet_152_padding.py --section ${section} train | tee "$dir"/logs/output_extract_train_"$section".txt
    done
else
    section = $1
    printf "Section: %s\n" $section >> "$dir"/logs/sections_train.txt
    python "$dir"/epic_extract_feature_map_ResNet_152_padding.py --section $section train | tee $dir/logs/output_extract_train_$section.txt
fi

