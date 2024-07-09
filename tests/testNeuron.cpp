//
// Created by Gautam Sharma on 6/30/24.
//


#include "utils.hpp"
#include "GradTester.hpp"
#include "Neuron.hpp"
#include <chrono>

using microgradpp::Value;
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

    // Zero grad
    {
        constexpr size_t expectedSize = 3;
        auto a = Neuron(expectedSize);
        const std::vector<std::shared_ptr<Value>> x = {Value::create(-1),Value::create(2),Value::create(3)};
        auto y = a(x);
        y->backProp();

        auto params = a.parameters();
        for(const auto& p : params){
            microgradpp::GradTester::notEquals<float>(p->grad, 0.0f, "Non Zero grad");
        }

        a.zeroGrad();
        params = a.parameters();
        for(const auto& p : params){
            microgradpp::GradTester::equals<float>(p->grad, 0.0f, "Zero grad");
        }
    }

    // test if neurons forward and backward pass give the same value as in micrograd -- num neurons = 3
    {
        auto variables = microgradpp::utils::readVariablesFromJson("test_neuron_equal_weights_output.json", true);
        constexpr size_t expectedSize = 3;
        auto a = Neuron(expectedSize, 12);
        const std::vector<std::shared_ptr<Value>> x = {Value::create(-1),Value::create(2),Value::create(3)};

        // Forward pass
        std::shared_ptr<Value> result;
        for(int idx = 0; idx< 10; ++idx){
            result = a(x);
            result->backProp();
            auto params = a.parameters();
            for(int jdx = 0; jdx< params.size(); ++jdx){
                microgradpp::GradTester::equals<float>(params[jdx]->grad, variables[std::to_string(idx)][jdx].grad, "Neuron grad value/Num neurons 3/idx:" + std::to_string(idx));
                microgradpp::GradTester::equals<float>(params[jdx]->data, variables[std::to_string(idx)][jdx].data, "Neuron data value/Num neurons 3/idx:" + std::to_string(idx));
            }
        }
    }

    // test if neurons forward and backward pass give the same value as in micrograd -- num neurons = 6
    {
        auto variables = microgradpp::utils::readVariablesFromJson("test_neuron_equal_weights_num_neurons_6_output.json", true);
        constexpr size_t expectedSize = 6;
        auto a = Neuron(expectedSize, -9);
        const std::vector<std::shared_ptr<Value>> x = {Value::create(-1),Value::create(2),Value::create(3), Value::create(7.89), Value::create(3.78), Value::create(6.78)};

        // Forward pass
        std::shared_ptr<Value> result;
        for(int idx = 0; idx< 10; ++idx){
            result = a(x);
            result->backProp();
            auto params = a.parameters();
            for(int jdx = 0; jdx< params.size(); ++jdx){
                microgradpp::GradTester::equals<float>(params[jdx]->grad, variables[std::to_string(idx)][jdx].grad, "Neuron grad value/Num neurons 6/idx:" + std::to_string(idx));
                microgradpp::GradTester::equals<float>(params[jdx]->data, variables[std::to_string(idx)][jdx].data, "Neuron data value/Num neurons 6/idx:" + std::to_string(idx));
            }
        }
    }




























    // Get the ending timestamp
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;

    // Output the duration in seconds
    std::cout << "Time taken by testNeuron: " << duration.count() << " seconds" << std::endl;
}
