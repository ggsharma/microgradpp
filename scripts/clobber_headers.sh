#!/bin/bash

# Directory containing the header files
HEADER_DIR="../include"
OUTPUT_FILE="../microgradpp.h"

# Create or clear the output file
> $OUTPUT_FILE

# Loop through all header files and concatenate them into the output file
for header in $(find $HEADER_DIR -name "*.h" -o -name "*.hpp"); do
  echo "// Content from $header" >> $OUTPUT_FILE
  cat $header >> $OUTPUT_FILE
  echo -e "\n" >> $OUTPUT_FILE
done

echo "All headers have been concatenated into $OUTPUT_FILE"