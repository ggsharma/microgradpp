import json
import os
import sys

# Calculate the relative path to the micrograd package
script_dir = os.path.dirname(os.path.abspath(__file__))
micrograd_path = os.path.join(script_dir, '3p/micrograd')

# Add the path to micrograd to the system path
sys.path.insert(0, micrograd_path)

from micrograd.engine import Value
from micrograd.nn import Neuron

def run_tests(test_file):
    # Calculate the full path to the test file
    test_file_path = os.path.join(script_dir, test_file)

    # Load the test definitions from the JSON file
    with open(test_file_path, 'r') as f:
        tests = json.load(f)

    for test in tests:
        test_name = test['name']
        commands = test['command']
        results_to_serialize = test.get('results', [])

        # Dictionary to store the output values
        output_values = {}

        # Execute each command in the test
        local_vars = {}
        for command in commands:
            exec(command, globals(), local_vars)

        # Serialize only the specified result variables
        for result in results_to_serialize:
            if result in local_vars:
                value = local_vars[result]
                if isinstance(value, Value):
                    # Initialize output_values[result] if it doesn't exist
                    if result not in output_values:
                        output_values[result] = {}
                    output_values[result]["data"] = value.data
                    output_values[result]["grad"] = value.grad
                else:
                    output_values[result] = value

        # Prepare the output file path
        output_file = f"{test_name}_output.json"

        # Save the output values to the JSON file
        with open(output_file, 'w') as f:
            json.dump(output_values, f, indent=4)

if __name__ == "__main__":
    # Example test file
    test_file = 'test_commands.json'
    run_tests(test_file)
