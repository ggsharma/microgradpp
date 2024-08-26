# import re
# import os
#
# # Define the headers in the required order
# headers = [
#     "../include/Value.hpp",
#     "../include/Activation.hpp",
#     "../include/Tensor.hpp",
#     "../include/Neuron.hpp",
#     "../include/Layer.hpp"
# ]
#
# output_file = "microgradpp.h"
#
# # Function to process each header file
# def process_header(file_path):
#     with open(file_path, 'r') as file:
#         content = file.readlines()
#
#     processed_content = []
#     include_pattern = re.compile(r'#include\s*["<](.*)[">]')
#     for line in content:
#         match = include_pattern.match(line)
#         if match:
#             header = match.group(1)
#             # Only keep standard library includes
#             if not os.path.isfile(os.path.join(os.path.dirname(file_path), header)):
#                 processed_content.append(line)
#         else:
#             processed_content.append(line)
#
#     return processed_content
#
# # Create or clear the output file
# with open(output_file, 'w') as outfile:
#     # Process each header file in the specified order
#     for header in headers:
#         outfile.write(f"// Content from {header}\n")
#         processed_content = process_header(header)
#         outfile.writelines(processed_content)
#         outfile.write("\n\n")
#
# print(f"All headers have been concatenated into {output_file}")

import re
import os
from collections import defaultdict, deque

# List of all header files to be processed
headers = [
    "../include/Value.hpp",
    "../include/Activation.hpp",
    "../include/Tensor.hpp",
    "../include/Neuron.hpp",
    "../include/Layer.hpp"
]

output_file = "microgradpp.h"

# Function to extract dependencies from a header file
def get_dependencies(file_path):
    include_pattern = re.compile(r'#include\s*["<](.*)[">]')
    dependencies = []
    with open(file_path, 'r') as file:
        for line in file:
            match = include_pattern.match(line)
            if match:
                header = match.group(1)
                # Only consider headers that are part of the project (ignore standard library includes)
                header_path = os.path.join(os.path.dirname(file_path), header)
                if os.path.isfile(header_path):
                    dependencies.append(header_path)
    return dependencies

# Perform topological sorting using DFS
def topological_sort(headers):
    graph = defaultdict(list)
    indegree = {header: 0 for header in headers}

    # Build the dependency graph
    for header in headers:
        dependencies = get_dependencies(header)
        for dep in dependencies:
            if dep in indegree:
                graph[dep].append(header)
                indegree[header] += 1

    # Perform topological sorting using Kahn's algorithm (BFS approach)
    queue = deque([header for header in headers if indegree[header] == 0])
    sorted_headers = []

    while queue:
        header = queue.popleft()
        sorted_headers.append(header)
        for neighbor in graph[header]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles
    if len(sorted_headers) != len(headers):
        raise ValueError("Cyclic dependency detected among headers.")

    return sorted_headers

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

# Perform topological sort on the headers based on their dependencies
sorted_headers = topological_sort(headers)

# Create or clear the output file
with open(output_file, 'w') as outfile:
    # Process each header file in the sorted order
    for header in sorted_headers:
        outfile.write(f"// Content from {header}\n")
        processed_content = process_header(header)
        outfile.writelines(processed_content)
        outfile.write("\n\n")

print(f"All headers have been concatenated into {output_file}")
