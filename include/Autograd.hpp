/**
 *  @file Autograd.hpp
 *  @brief Defines the Autograd class for automatic differentiation and gradient computation.
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
 *  This header file contains the definition of the Autograd class, which facilitates automatic
 *  differentiation by recording operations on the Value class. It provides methods for adding
 *  entries to the computation tape and performing backward passes to compute gradients.
 */

#pragma once

#include <vector>
#include <functional>
#include <unordered_map>

namespace microgradpp{
    class Value;
    using ValuePtr = std::shared_ptr<Value>;

    // TapeEntry
    struct TapeEntry {
        ValuePtr output;                          // The output of the operation
        std::function<void()> backward_fn = nullptr;        // Function to compute gradients during backward pass
    };

    class Autograd {
    public:
        Autograd() = default;

        std::vector<TapeEntry> tape;              // Stores the sequence of operations
        std::unordered_map<size_t, std::function<void()>> TapeMap;

        /**
         * @brief Adds an operation to the computation tape.
         *
         * @param output The output of the operation to be recorded.
         * @param backward_fn The function to compute gradients during the backward pass.
         */
        void add_entry(ValuePtr& output, std::function<void()> backward_fn) {
            tape.push_back({output, std::move(backward_fn)});
        }

        /**
         * @brief Performs a backward pass through the computation tape,
         * executing the stored backward functions in reverse order.
         */
        void backward() {
            for (auto it = global_tape.tape.rbegin(); it != global_tape.tape.rend(); ++it) {
                if (it->backward_fn) {
                    it->backward_fn();
                }
            }
        }

        /**
         * @brief Clears the global computation tape.
         */
        static void clear() noexcept {
            global_tape.tape.clear();
        }

        static Autograd global_tape; // Global instance of Autograd for tracking operations
    };
}
