/**
 *  @file Layers.hpp
 *  @brief Provides factory functions for creating neural network layers.
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
 *  This file contains factory functions for creating instances of neural network layers,
 *  specifically linear layers and ReLU activation layers. These functions simplify the
 *  creation and management of layer instances within the microgradpp framework.
 */

#pragma once

// Standard libraries
#include <memory>

// microgradpp core libraries
#include "core/CoreReLU.hpp"
#include "core/CoreTanH.hpp"
#include "core/CoreLinear.hpp"

namespace microgradpp::nn {

    /**
     * @brief Factory function to create a unique pointer to a CoreLinear layer.
     *
     * This function constructs a new instance of the `CoreLinear` class using perfect
     * forwarding for its constructor arguments, ensuring efficient and flexible 
     * initialization.
     *
     * @tparam T Variadic template parameter pack for constructor arguments.
     * @param args Arguments to be forwarded to the CoreLinear constructor.
     * @return std::unique_ptr<microgradpp::core::CoreLinear> A unique pointer
     *         to the created CoreLinear layer instance.
     */
    template <class... T>
    std::unique_ptr<microgradpp::core::CoreLinear> Linear(T&&... args) {
        return std::make_unique<microgradpp::core::CoreLinear>(std::forward<T>(args)...);
    }

    /**
     * @brief Factory function to create a unique pointer to a CoreReLU layer.
     *
     * This function constructs a new instance of the `CoreReLU` class without any
     * constructor arguments, returning a unique pointer to the created layer.
     *
     * @return std::unique_ptr<microgradpp::core::CoreReLU> A unique pointer
     *         to the created CoreReLU layer instance.
     */
    std::unique_ptr<microgradpp::core::CoreReLU> ReLU() {
        return std::make_unique<microgradpp::core::CoreReLU>();
    }

    /**
     * @brief Factory function to create a unique pointer to a CoreTanH layer.
     *
     * This function constructs a new instance of the `CoreTanH` class without any
     * constructor arguments, returning a unique pointer to the created layer.
     *
     * @return std::unique_ptr<microgradpp::core::CoreTanH> A unique pointer
     *         to the created CoreTanH layer instance.
     */
    std::unique_ptr<microgradpp::core::CoreTanH> TanH() {
        return std::make_unique<microgradpp::core::CoreTanH>();
    }
}
