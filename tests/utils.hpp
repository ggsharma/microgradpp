#include <iostream>
#include <fstream>
#include <unordered_map>
#include "json.h"


struct props{
    float grad;
    float data;
};

namespace microgradpp {
    namespace utils {

        using json = nlohmann::json;

        // Function to read variables from a JSON file
        std::unordered_map<std::string, props> readVariablesFromJson(const std::string& filename) {
            std::unordered_map<std::string, props> output;
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Error opening file: " << filename << std::endl;
                return output;
            }

            json j;

            try {
                file >> j; // Parse JSON from file stream
            } catch (json::parse_error& e) {
                std::cerr << "Parse error while reading JSON: " << e.what() << std::endl;
                return output;
            }

            file.close();

            // Iterate over JSON object and read variables
            for (auto& el : j.items()) {
                std::string key = el.key();
                auto value = el.value();

                    if (value.is_object()) {
                    // Handle object values (if needed)
                    //std::cout << "Variable " << key << " is an object:" << std::endl;
                    // Process object fields as required
                    // Example: Read nested values
                    if (value.contains("grad")) {
                        auto nested_value = value["grad"];
                        if (nested_value.is_number()) {
                            double num_value = nested_value;
                            output[key].grad = num_value;
                            //std::cout << "Nested field value: " << num_value << std::endl;
                        }
                        // Handle other types as needed
                    }
                    if (value.contains("data")) {
                            auto nested_value = value["data"];
                            if (nested_value.is_number()) {
                                double num_value = nested_value;
                                output[key].data = num_value;
                                //std::cout << "Nested field value: " << num_value << std::endl;
                            }
                            // Handle other types as needed
                        }
                }
                // Add more types as needed (arrays, nested objects, etc.)
            }

            return output;
        }
    }
}
