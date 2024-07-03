#include <iostream>
#include <fstream>
#include <unordered_map>
#include "json.h"


struct props{
    float data;
    float grad;
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
                    else{
                        // Iterate over JSON object and read variables
                        for (auto& el : j["c"].items()) {
                            std::string key = el.key();
                            auto value = el.value();

                            if (value.is_array()) {
                                for (auto& arr : value) {
                                    if (arr.is_array() && arr.size() == 2 && arr[1].is_number()) {
                                        double data_value = arr[1];
                                        output[key].data = data_value;
                                    }
                                }
                            }
                        }
                    }

                }
                // Add more types as needed (arrays, nested objects, etc.)
            }

            return output;
        }
        std::unordered_map<std::string, std::vector<props>> readVariablesFromJson(const std::string& filename, bool flag) {
            std::unordered_map<std::string, std::vector<props>> output;
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
                props  out;
                std::string key = el.key();
                auto value = el.value();

                if (value.is_object()) {

                        // Iterate over JSON object and read variables
                        for (auto& el : j["c"].items()) {
                            std::string key = el.key();
                            auto value = el.value();
                            int idx = 0;
                            if (value.is_array()) {
                                for (auto& arr : value) {
                                    //if (arr.is_array() && arr.size() == 2 && arr[1].is_number()) {
                                        for(int jdx=0; jdx < arr.size(); jdx += 2){
                                            // data, grad
                                            output[key].push_back({arr[jdx], arr[jdx+1]});
                                        }

                                    //}
                                }
                            }
                        }
                    }
            }

            return output;
        }
    }
}
