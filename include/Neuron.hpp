/**
 *  @file Neuron.hpp
 *  @brief Defines the Neuron class, which represents a single neuron in a neural network,
 *  including its weights, bias, and activation functionalities.
 *
 *  This file is part of the microgradpp project, a lightweight C++ library for neural
 *  network training and inference.
 *
 *  @section License
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 *  @section Author
 *  Gautam Sharma
 *  Email: gautamsharma2813@gmail.com
 *  Date: October 12, 2024
 *
 *  @details
 *  This header file contains the definition of the Neuron class, which includes methods for
 *  weight initialization, forward propagation, gradient management, and parameter retrieval.
 */


#ifndef MICROGRADPP_NEURON_HPP
#define MICROGRADPP_NEURON_HPP

#include "Value.hpp"
#include "Activation.hpp"
#include "TypeDefs.hpp"
#include "Tensor.hpp"

#include <memory>
#include <vector>
#include <random>
#include <stdio.h>
#include <numeric> // for std::inner_product
#include <thread>
#include <execution> // For parallel execution policies
#include <numeric> // For std::transform_reduce

#include <future>

namespace microgradpp {

/**
 * @brief Generates a random float between -1 and 1.
 *
 * This function uses a random number generator to produce a floating-point
 * number uniformly distributed in the range [-1, 1].
 * It is used for initializing weights in the Neuron class.
 *
 * @return A random float between -1 and 1.
 */
    float getRandomFloat() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(-1, 1);
        return static_cast<float>(dis(gen));
    }

/**
 * @brief Represents a single neuron in a neural network.
 *
 * The Neuron class contains weights and a bias that are used for calculating
 * the output of the neuron given an input tensor. It also provides methods
 * for parameter management, gradient resetting, and output computation.
 */
    class Neuron {
    private:
        Tensor1D weights;      ///< Weights of the neuron.
        ValuePtr bias = Value::create(0.0f); ///< Bias of the neuron.

    public:
        /**
         * @brief Constructs a Neuron with a specified number of inputs.
         *
         * Initializes the weights with random values and the bias with zero.
         *
         * @param nin The number of input connections to the neuron.
         */
        Neuron(size_t nin) {
            for (size_t idx = 0; idx < nin; ++idx) {
                weights.emplace_back(Value::create(getRandomFloat()));
            }
        }

        /**
         * @brief Constructs a Neuron for testing purposes.
         *
         * Initializes the weights with random values. The second parameter is
         * currently unused but provided for potential testing customization.
         *
         * @param nin The number of input connections to the neuron.
         * @param val A float value (currently unused).
         */
        Neuron(size_t nin, float val) {
            for (size_t idx = 0; idx < nin; ++idx) {
                weights.emplace_back(Value::create(getRandomFloat()));
            }
        }

        /**
         * @brief Resets the gradients of the neuron.
         *
         * This method sets the gradients of all weights and the bias to zero,
         * preparing the neuron for a new forward/backward pass in training.
         */
        void zeroGrad() {
            for (auto& weight : weights) {
                weight->grad = 0;
            }
            bias->grad = 0;
        }

        /**
         * @brief Computes the output of the neuron for a given input.
         *
         * This operator overload calculates the dot product of the neuron's
         * weights and the input tensor, adds the bias, and returns the result.
         * If the input tensor size does not match the weights size, an exception
         * is thrown.
         *
         * @param x The input tensor (1D) to the neuron.
         * @return A pointer to the resulting Value after computation.
         * @throws std::invalid_argument If the input size does not match the
         *         weights size.
         */
        ValuePtr operator()(const Tensor1D& x) {
            if (x.size() != weights.size()) {
                throw std::invalid_argument("Error in micrograd::Neuron -> Vectors must be of the same length");
            }

            ValuePtr sum = Value::create(0.0);

            for (size_t idx = 0; idx < weights.size(); ++idx) {
                ValuePtr intermediateVal = Value::multiply(x[idx], weights[idx]);
                sum = Value::add(sum, intermediateVal);
            }

            // Add bias
            sum = Value::add(sum, bias);

            return sum;
        }

        /**
         * @brief Returns the parameters of the neuron as a tensor.
         *
         * This method combines the weights and the bias into a single Tensor1D
         * object and returns it. The method reserves space for efficiency.
         *
         * @return A Tensor1D containing the weights and bias of the neuron.
         */
        __MICROGRADPP_NO_DISCARD__
        Tensor1D parameters() const {
            Tensor1D out;
            out.reserve(weights.size() + 1);

            out.insert(out.end(), weights.begin(), weights.end());
            out.push_back(bias);

            return out;
        }

        /**
         * @brief Prints the parameters of the neuron.
         *
         * This method outputs the current values and gradients of the weights
         * and bias to the console for debugging purposes.
         */
        void printParameters() const {
            printf("Number of Parameters: %zu\n", weights.size() + 1);
            for (const auto& param : weights) {
                printf("%f, %f\n", param->data, param->grad);
            }
            printf("%f, %f\n", bias->data, bias->grad);
            printf("\n");
        }

        /**
         * @brief Returns the total number of parameters in the neuron.
         *
         * This method counts the number of weights plus the bias, which represents
         * the total parameters of the neuron.
         *
         * @return The total number of parameters in the neuron.
         */
        __MICROGRADPP_NO_DISCARD__
        size_t getParametersSize() const {
            return weights.size() + 1;
        }
    };

} // namespace microgradpp

#endif //MICROGRADPP_NEURON_HPP
