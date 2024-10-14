/**
 *  @file AbstractLoss.hpp
 *  @brief Defines an abstract base class for loss functions in neural networks.
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
 *  Created on: Sept 1, 2024
 *
 *  @details
 *  This header file contains the definition of the AbstractLoss template class. This class
 *  serves as a base for creating various loss functions in neural networks. Any class
 *  derived from AbstractLoss must implement the `operator()` method, which takes two
 *  parameters of type T (representing ground truth and predictions) and returns a ValuePtr
 *  representing the calculated loss.
 */

#pragma once
#include "Value.hpp"

namespace microgradpp {
    /**
     * @brief Abstract base class for loss functions.
     *
     * Any class that derives from AbstractLoss needs to define a `operator()`
     * method for calculating the loss between ground truth and predictions.
     */
    template<class T>
    class AbstractLoss {
    public:
        AbstractLoss() = default;

        /**
         * @brief Calculates the loss between ground truth and predictions.
         *
         * @param groundTruth The actual values.
         * @param prediction The predicted values.
         * @return A pointer to the calculated loss value.
         */
        virtual ValuePtr operator()(const T& groundTruth, const T& prediction) = 0;
    };
}
