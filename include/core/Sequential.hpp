/**
 *  @file Sequential.hpp
 *  @brief Defines the `Sequential` class for sequentially stacking neural network layers.
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
 *  Date: October 14, 2024
 *
 *  @details
 *  The `Sequential` class facilitates the creation of a sequence of neural network layers,
 *  allowing for easy forward propagation through a series of layers and providing methods
 *  for gradient management and parameter updates. This structure is designed to simplify
 *  model construction and manipulation in the microgradpp framework.
 */

#pragma once

// Standard libraries
#include <vector>
#include <memory>

// microgradpp libraries
#include "CoreReLU.hpp"
#include "CoreLinear.hpp"
#include "TypeDefs.hpp"

namespace microgradpp::core {

    /**
     * @class Sequential
     * @brief Manages a sequence of neural network layers and supports forward propagation.
     *
     * The `Sequential` class allows users to stack neural network layers in a defined
     * sequence, providing functionality for forward propagation, parameter access,
     * gradient updates, and zeroing. This class is essential for constructing
     * feedforward neural networks in the microgradpp library.
     */
    class Sequential {
    private:
        /// Sequence of neural network layers.
        std::vector<std::shared_ptr<MppCore>> _layerSequence;

    public:

        /**
         * @brief Constructs a `Sequential` object with a specified layer sequence.
         *
         * @param layerSequence A vector of shared pointers to `MppCore` objects representing
         * the neural network layers to be stacked sequentially.
         */
        Sequential(std::vector<std::shared_ptr<MppCore>> layerSequence)
                : _layerSequence(std::move(layerSequence)) {};

        /**
         * @brief Performs forward propagation through the sequence of layers.
         *
         * Iteratively applies each layer in the sequence to the input, passing the result
         * from one layer as the input to the next.
         *
         * @param input The input tensor for the network.
         * @return Tensor1D The output tensor after passing through all layers.
         */
        Tensor1D operator()(const Tensor1D& input) {
            auto result = input;
            for(auto& layer : _layerSequence){
                auto out = layer->operator()(result);
                result = out;
            }
            return result;
        }

        /**
         * @brief Collects parameters from all layers in the sequence.
         *
         * Provides access to all parameters within the layers in the sequence, enabling
         * updates and gradient tracking across the entire network.
         *
         * @return std::vector<Value*> A vector of pointers to the parameters in all layers.
         */
        __MICROGRADPP_NO_DISCARD__
        std::vector<Value*> parameters() const {
            std::vector<Value*> params;
            if(params.empty()) {
                for (const auto &layerSeq: _layerSequence) {
                    for (const auto &p: layerSeq->parameters()) {
                        params.push_back(p);
                    }
                }
            }
            return params;
        }

        /**
         * @brief Updates each parameter in the network based on the specified learning rate.
         *
         * Adjusts each parameter by subtracting the product of the learning rate and
         * the parameter's gradient, supporting the training process.
         *
         * @param learningRate The learning rate to apply for each parameter update.
         */
        void update(float learningRate) {
            for (auto &p: this->parameters()) {
                p->data += (float)((float)-learningRate * (float)p->grad);
            }
        }

        /**
         * @brief Prints parameters for each layer in the sequence.
         *
         * Calls the `printParameters` function of each layer, displaying details
         * about the parameters across all layers.
         */
        void printParameters() {
            for(const auto& layerSeq: _layerSequence){
                layerSeq->printParameters();
            }
        }

        /**
         * @brief Resets gradients for all parameters in each layer.
         *
         * Calls the `zeroGrad` function of each layer, setting all parameter
         * gradients to zero for a new optimization step.
         */
        void zeroGrad() {
            for(const auto& layerSeq: _layerSequence){
                layerSeq->zeroGrad();
            }
        }

        /**
         * @brief Prints information about each layer in the sequence.
         *
         * Calls the `print` function of each layer, outputting information such
         * as layer types and sizes for each component in the sequence.
         */
        void print() {
            for(const auto& layerSeq: _layerSequence){
                layerSeq->print();
            }
        }
    };
}
