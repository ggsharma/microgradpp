//
// Created by Gautam Sharma on 6/29/24.
//

#include "utils.hpp"
#include "Tester.hpp"

int main(){
    auto variables = microgradpp::utils::readVariablesFromJson("test2_output.json");
    // Example usage: iterate over variables
    for (const auto& pair : variables) {
        std::cout << "Variable: " << pair.first << ", Value: data " << pair.second.data <<  ", Grad: "<< pair.second.grad << std::endl;
    }

//    float c = 1;
    microgradpp::Tester::equals<float>(11.560670, variables["e"].data, "test1");
}
