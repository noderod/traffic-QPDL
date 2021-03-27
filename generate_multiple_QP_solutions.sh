#!/bin/bash

##################################
# SUMMARY
#
# Rapidly generates many QP solutions from a single set of OpenStreetMap's data and AADT values
##################################

# Filepath to OSM file
OSM_FILE_FILEPATH="$1"
# Filepath to JSON AADT data file
AADT_JSON_FILEPATH="$2"
# Path where original preprocessed output is to be generated
OUTPUT_JSON_FILEPATH="$3"
# Gets the number of copies
num_copies="$4"
# Directory to store the preprocessed results (must include the final '/')
multi_processed_dir="$5"


# Ensures that the number of copies must be 0 or positive
if (("$num_copies" <= 0)); then
    echo "The number of copies must be at least 1"
    exit
fi



# Preprocesses the data
python3 QP/preprocessing.py \
    --osm "$OSM_FILE_FILEPATH" \
    --aadt "$AADT_JSON_FILEPATH" \
    --output "$OUTPUT_JSON_FILEPATH"


# Deletes the last '/' from the directory name
output_dir_name_without_final_slash=${multi_processed_dir::-1}


# Generates as many maps with self-generated AADT values as needed
python3 QP/multi_map_generator.py --preprocessed_input "$OUTPUT_JSON_FILEPATH" --multiples "$num_copies" --output_dir "$multi_processed_dir"



# Creates the output directory if it does not exists already
output_QP_dir="$output_dir_name_without_final_slash"_output/

if [ ! -d "$output_QP_dir" ]; then
    mkdir "$output_QP_dir"
fi


# QP optimization

# Goes through all the files
for filename in "$multi_processed_dir"*; do

    output_location="$output_QP_dir"$(basename "$filename")

    python3 QP/QP.py \
        --input "$filename" \
        --output "$output_location"
done
