//
// Created by Gautam Sharma on 6/30/24.
//


#include "utils.hpp"
#include "GradTester.hpp"
#include "Neuron.hpp"
#include <chrono>

using microgradpp::Neuron;

int main() {
    // Get the starting timestamp
    auto start = std::chrono::high_resolution_clock::now();

    // Size of neurons
    {
        constexpr size_t expectedSize = 1000;
        auto a = Neuron(expectedSize);
        // +1 for bias
        microgradpp::GradTester::equals<size_t>(a.getParametersSize(), expectedSize + 1, "Size of neurons");
    }

    // Get the ending timestamp
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;

    // Output the duration in seconds
    std::cout << "Time taken by testNeuron: " << duration.count() << " seconds" << std::endl;
}
