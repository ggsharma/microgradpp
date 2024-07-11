import re
import os

# Define the headers in the required order
headers = [
    "../include/Value.hpp",
    "../include/Activation.hpp",
    "../include/Tensor.hpp",
    "../include/Neuron.hpp",
    "../include/Layer.hpp"
]

output_file = "microgradpp.h"

# Function to process each header file
def process_header(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    processed_content = []
    include_pattern = re.compile(r'#include\s*["<](.*)[">]')
    for line in content:
        match = include_pattern.match(line)
        if match:
            header = match.group(1)
            # Only keep standard library includes
            if not os.path.isfile(os.path.join(os.path.dirname(file_path), header)):
                processed_content.append(line)
        else:
            processed_content.append(line)

    return processed_content

# Create or clear the output file
with open(output_file, 'w') as outfile:
    # Process each header file in the specified order
    for header in headers:
        outfile.write(f"// Content from {header}\n")
        processed_content = process_header(header)
        outfile.writelines(processed_content)
        outfile.write("\n\n")

print(f"All headers have been concatenated into {output_file}")
