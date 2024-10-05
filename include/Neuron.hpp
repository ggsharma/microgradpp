//
// Created by Gautam Sharma on 6/30/24.
//

#ifndef MICROGRADPP_NEURON_HPP
#define MICROGRADPP_NEURON_HPP

#include "Value.hpp"
#include "Activation.hpp"
#include "TypeDefs.hpp"

#include <memory>
#include <vector>
#include <random>
#include <stdio.h>
#include <numeric> // for std::inner_product
#include <thread>
#include <execution> // For parallel execution policies
#include <numeric> // For std::transform_reduce

#include <future>


namespace microgradpp{

    // Function to generate a random float between -1 and 1
    float getRandomFloat() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(-1, 1);
        return static_cast<float>(dis(gen));
    }

    class Neuron {
    private:
        std::vector<ValuePtr> weights;
        ValuePtr bias = Value::create(0.0f);
        const ActivationType activation_t;

    public:
        Neuron(size_t nin, const ActivationType& activation_t) : activation_t(activation_t) {
            for (size_t idx = 0; idx < nin; ++idx) {
                weights.emplace_back(Value::create(getRandomFloat()));
            }
        }

        // For testing
        Neuron(size_t nin, float val, const ActivationType& activation_t = ActivationType::SIGMOID)
                : activation_t(activation_t) {
            for (size_t idx = 0; idx < nin; ++idx) {
                weights.emplace_back(Value::create(getRandomFloat()));
            }
        }

        void zeroGrad() {
            for (auto& weight : weights) {
                weight->grad = 0;
            }
            bias->grad = 0;
        }

        // Dot product of a Neuron's weights with the input
        ValuePtr operator()(const std::vector<ValuePtr>& x) {
            if (x.size() != weights.size()) {
                throw std::invalid_argument("Error in micrograd::Neuron -> Vectors must be of the same length");
            }

            ValuePtr sum = Value::create(0.0);

            for (size_t idx = 0; idx < weights.size(); ++idx) {

                ValuePtr intermediateVal = Value::multiply(x[idx], weights[idx]);
                //std::cout << "Multiplying " << weights[idx] << " with " << x[idx] << " Results is: " << intermediateVal<<"\n";
                sum = Value::add(sum, intermediateVal);
                //std::cout <<  sum << std::endl;
            }

            // Add bias
            //sum->add_inplace(bias);
            sum = Value::add(sum, bias);

            // Apply activation function
            const auto& activationFcn = Activation::mActivationFcn.at(activation_t);
            return activationFcn(sum);
            //std::cout << "Results of actvation is: " << b << "\n";
            //return b;

        }

        __MICROGRADPP_NO_DISCARD__
        std::vector<ValuePtr> parameters() const {
            std::vector<ValuePtr> out;
            out.reserve(weights.size() + 1);

            out.insert(out.end(), weights.begin(), weights.end());
            out.push_back(bias);

            return out;
        }

        void printParameters() const {
            printf("Number of Parameters: %zu\n", weights.size() + 1);
            for (const auto& param : weights) {
                printf("%f, %f\n", param->data, param->grad);
            }
            printf("%f, %f\n", bias->data, bias->grad);
            printf("\n");
        }

        __MICROGRADPP_NO_DISCARD__
        size_t getParametersSize() const {
            return weights.size() + 1;
        }
    };

}

#endif //MICROGRADPP_NEURON_HPP
