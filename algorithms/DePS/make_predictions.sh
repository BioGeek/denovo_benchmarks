#!/bin/bash

# Get dataset property tags
DSET_TAGS=$(python /algo/base/dataset_tags_parser.py --dataset "$@")
# Parse tags and set individual environment variables for each of them
# (variable names are identical to tag names
#  -- check DatasetTag values in dataset_config.py)
while IFS='=' read -r key value; do
    export "$key"="$value"
done <<< "$DSET_TAGS"

# Iterate through files in the dataset
for input_file in "$@"/*.mgf; do

    echo "Processing file: $input_file"

    # Convert input data to model format
    python main.py --spectrum "$input_file"

done