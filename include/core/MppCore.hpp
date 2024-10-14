/**
 *  @file MppCore.hpp
 *  @brief Defines the abstract base class `MppCore` for core neural network layers.
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
 *  The `MppCore` class provides an abstract interface for core neural network layers,
 *  such as linear and activation layers. It defines virtual methods that derived classes
 *  must implement or can override, supporting core functionality such as parameter access,
 *  gradient management, and printing. This class enables polymorphism across different
 *  neural network components.
 */

#pragma once

// Standard libraries
#include <vector>

// microgradpp libraries
#include "Value.hpp"
#include "TypeDefs.hpp"

namespace microgradpp::core {

    /**
     * @class MppCore
     * @brief Abstract base class for core components of neural network layers.
     *
     * The `MppCore` class serves as an interface for all core neural network layers.
     * It defines essential functions for neural network layers, including parameter
     * management, gradient resetting, and forward computation, which must be implemented
     * by any class that inherits from `MppCore`.
     */
    class MppCore {
    public:

        /**
         * @brief Default constructor for `MppCore`.
         *
         * Provides a default constructor that allows derived classes to initialize
         * without specifying additional parameters.
         */
        MppCore() = default;

        /**
         * @brief Pure virtual function to print layer information.
         *
         * This function must be implemented by derived classes to print relevant
         * layer information, such as the type and size of the layer.
         */
        virtual void print() const = 0;

        /**
         * @brief Pure virtual function for forward computation.
         *
         * Processes the input tensor and returns the output tensor by performing
         * the layer's specific operation. Must be implemented by derived classes.
         *
         * @param in Input tensor.
         * @return Tensor1D Output tensor after applying the layer's computation.
         */
        virtual Tensor1D operator()(const Tensor1D& in) = 0;

        /**
         * @brief Resets gradients for all parameters in the layer.
         *
         * Sets the gradient values of all parameters in the layer to zero. This function
         * is optional for derived classes and can be overridden where necessary.
         */
        virtual void zeroGrad() {};

        /**
         * @brief Prints parameter information for the layer.
         *
         * This function is optional for derived classes and can be overridden
         * to print specific details about the layer's parameters.
         */
        virtual void printParameters() const {};

        /**
         * @brief Returns a list of pointers to the layer's parameters.
         *
         * This function is optional for derived classes and can be overridden
         * to provide access to the layer's parameters, enabling parameter updates
         * and gradient tracking.
         *
         * @return std::vector<Value*> A vector of pointers to the parameters in the layer.
         */
        __MICROGRADPP_NO_DISCARD__ virtual std::vector<Value*> parameters() const {
            return {};
        }
    };
}
