/**
 *  @file BaseMultiLayerPerceptron.hpp
 *  @brief Defines a base class for implementing Multi-Layer Perceptron (MLP) networks.
 *
 *  This file is part of the microgradpp project, a lightweight C++ library for neural
 *  network training and inference. It provides foundational MLP functionality that
 *  can be extended to create customized neural network architectures.
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
 *  Date: September 29, 2024
 *
 *  @details
 *  The `BaseMultiLayerPerceptron` class offers core functionality for MLP networks,
 *  including gradient zeroing, parameter updates, and basic parameter display methods.
 *  It manages a `Sequential` object which allows chaining of neural network layers,
 *  simplifying MLP structure and functionality.
 *
 *  Copyright (c) 2024 Gautam Sharma. All rights reserved.
 */


#pragma once

#include "core/Sequential.hpp"
#include "TypeDefs.hpp"

using microgradpp::core::Sequential;

namespace microgradpp::base {

/**
 * @class BaseMultiLayerPerceptron
 * @brief Abstract base class for Multi-Layer Perceptron (MLP) networks.
 *
 * This class provides a foundation for implementing multi-layer perceptron models.
 * It encapsulates a `Sequential` object, which contains a series of neural network layers,
 * and provides methods to print parameters, reset gradients, and update parameters.
 * Derived classes must implement the `forward` method to define the forward pass logic.
 */
    class BaseMultiLayerPerceptron {

        /// Internal `Sequential` object managing layers in the MLP.
        Sequential _baseSequential;

    public:
        /// Public reference to the underlying `Sequential` object, used to build layer sequences.
        Sequential& sequential = _baseSequential;

        /**
         * @brief Constructs the MLP base class with a given `Sequential` layer sequence.
         * @param sequential A `Sequential` object containing the layer sequence for the MLP.
         */
        BaseMultiLayerPerceptron(const Sequential& sequential) : _baseSequential(sequential) {}

        /**
         * @brief Prints information about the model's layer sequence.
         */
        void print() {
            this->_baseSequential.print();
        }

        /**
         * @brief Prints the parameters of each layer in the MLP model.
         */
        void printParameters() {
            this->_baseSequential.printParameters();
        }

        /**
         * @brief Resets the gradients for all layers to zero, preparing for a new training iteration.
         */
        void zeroGrad() {
            this->_baseSequential.zeroGrad();
        }

        /**
         * @brief Updates the model parameters based on computed gradients and the learning rate.
         */
        void update() {
            this->_baseSequential.update(this->learningRate);
        }

        /**
         * @brief Invokes the forward pass of the MLP for a given input.
         * @param input The input tensor of shape 1D.
         * @return The output tensor after the forward pass.
         */
        Tensor1D operator()(const Tensor1D& input) {
            return this->forward(input);
        }

        /**
         * @brief Pure virtual method for the forward pass, to be implemented in derived classes.
         * @param input Input tensor for the forward pass.
         * @return Output tensor after processing the input through the MLP layers.
         */
        virtual Tensor1D forward(Tensor1D input) = 0;

    protected:
        /// Learning rate used during the parameter update step.
        float learningRate = 0.001;
    };

} // namespace microgradpp::base
