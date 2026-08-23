#!/bin/bash

for file in egress_pwr_*mHz_z0000km.fits
do
    # Remove the .fits extension
    base="${file%.fits}"

    # Name of the output file
    cropped="${base}_cropped.fits"

    echo "========================================"
    echo "Processing: $file"
    echo "Output:     $cropped"
    echo "========================================"

    # Create the rectangular cropped template
    rect_mask "$file" \
        150 150 844 844 \
        "$cropped"

    # Apply the original egression power map to the cropped template
    mask_mult \
        "$cropped" \
        "$file"

    echo "Finished: $cropped"
    echo
done

echo "All egression power maps have been processed."
