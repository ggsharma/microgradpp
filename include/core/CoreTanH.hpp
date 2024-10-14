/**
 *  @file CoreTanH.hpp
 *  @brief Defines the CoreTanH class for applying TanH activation in neural networks.
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
 *  The `CoreTanH` class provides a TanH (Hyperbolic Tangent) activation layer,
 *  applying the TanH function element-wise to an input tensor. This class inherits from
 *  `MppCore` and overrides the necessary methods for a basic TanH layer in neural networks.
 */

#pragma once

// Standard libraries
#include <iostream>
#include <cassert>

// microgradpp libraries
#include "MppCore.hpp"
#include "Activation.hpp"

namespace microgradpp::core {

    /**
     * @class CoreReLU
     * @brief Implements the ReLU activation function as a layer in a neural network.
     *
     * The `CoreReLU` class is a layer that applies the ReLU activation function
     * element-wise to an input tensor, setting all negative values to zero.
     */
    class CoreTanH : public MppCore {
    public:

        /**
         * @brief Prints layer information, displaying that this is a ReLU layer.
         *
         * Outputs "ReLU Layer" to the standard console to indicate the type of layer.
         */
        void print() const override final {
            std::cout << "TanH Layer" << std::endl;
        }

        /**
         * @brief Applies the ReLU activation function to each element in the input tensor.
         *
         * Uses the ReLU function, defined in the `Activation` class, to process each
         * element in the input tensor. This is achieved by iterating through the tensor
         * and replacing negative values with zero while keeping positive values unchanged.
         *
         * @param in Input tensor for which ReLU activation is applied.
         * @return Tensor1D Output tensor where each element is the result of applying ReLU to the corresponding input element.
         */
        Tensor1D operator()(const Tensor1D& in) override {
            const auto& activationFcn = Activation::mActivationFcn.at(ActivationType::TANH);
            Tensor1D out;
            for(const auto& value : in) {
                out.push_back(activationFcn(value));
            }
            return out;
        }
    };
}
