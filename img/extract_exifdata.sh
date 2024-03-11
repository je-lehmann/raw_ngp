#!/bin/bash

# Specify the directory containing your images
image_directory="/home/lehmann/scratch2/datasets/trooper_exp/raw"
output_directory="/home/lehmann/scratch2/datasets/trooper_exp/raw"
# Iterate over each image file and extract metadata
for image_file in "$image_directory"/*.exr; do
    # Extract the filename without path and extension
    filename=$(basename "$image_file")
    filename_without_extension="${filename%.*}"

    # Run exiftool to extract metadata from the image and save to JSON
    ./exiftool-src/exiftool -j "$image_file" > "$output_directory/${filename_without_extension}.json" 
    #exiftool -j "$image_file" > "${filename_without_extension}_metadata.json"

    echo "Metadata extracted from $image_file and saved to $output_directory/${filename_without_extension}_metadata.json"
done
