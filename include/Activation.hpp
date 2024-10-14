/**
 *  @file Activation.hpp
 *  @brief Defines activation functions and their types for neural networks.
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
 *  Created on: July 9, 2024
 *
 *  @details
 *  This header file contains the definition of the Activation class, which provides
 *  different activation functions commonly used in neural networks, including ReLU,
 *  Tanh, and Sigmoid. It also defines an enumeration for activation types and maps
 *  them to their respective functions.
 */

#ifndef MICROGRADPP_ACTIVATION_HPP
#define MICROGRADPP_ACTIVATION_HPP

// Standard libraries
#include <unordered_map>
#include <functional>

// Microgradpp headers
#include "Value.hpp"

namespace microgradpp {
    // Enumeration for different activation types
    enum ActivationType {
        RELU,    ///< Rectified Linear Unit
        TANH,    ///< Hyperbolic Tangent
        SIGMOID  ///< Sigmoid function
    };

    class Activation {
        // Static functions for activation computations
        static std::shared_ptr<Value> Relu(const std::shared_ptr<Value>& val) {
            return Value::relu(val);
        }

        static std::shared_ptr<Value> TanH(const std::shared_ptr<Value>& val) {
            return Value::tanh(val);
        }

        static std::shared_ptr<Value> Sigmoid(const std::shared_ptr<Value>& val) {
            return Value::sigmoid(val);
        }

    public:
        // Map of activation types to their corresponding functions
        static inline std::unordered_map<ActivationType, std::function<std::shared_ptr<Value>(const std::shared_ptr<Value>&)>> mActivationFcn = {
                {ActivationType::RELU, Relu},
                {ActivationType::TANH, TanH},
                {ActivationType::SIGMOID, Sigmoid}
        };
    };
}

#endif // MICROGRADPP_ACTIVATION_HPP

