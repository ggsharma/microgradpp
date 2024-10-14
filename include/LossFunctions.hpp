/**
 *  @file Loss.hpp
 *  @brief Defines loss functions for evaluating model performance, including Mean Squared Error.
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
 *  This header file contains the definitions of loss functions used in the microgradpp
 *  library for evaluating the performance of neural network models. It includes the
 *  MeanSquaredError and MeanSquaredErrorFor1DPixels classes, which compute the mean squared
 *  error between the predicted and ground truth outputs.
 */

#pragma once

// stdlibs
#include <cassert>

#include "AbstractLoss.hpp"
#include "Tensor.hpp"

namespace microgradpp::loss{

    class MeanSquaredError : public AbstractLoss<Tensor2D>{
    public:
        MeanSquaredError () = default;

        /**
         * @brief Computes the mean squared error loss between ground truth and predictions.
         * 
         * @param groundTruth The true values.
         * @param prediction The predicted values.
         * @return ValuePtr The computed mean squared error loss.
         */
        ValuePtr operator()(const Tensor2D& groundTruth, const Tensor2D& prediction) override{
            // Calculate loss
            auto loss = Value::create(0.0f);
            assert(groundTruth.size() == prediction.size());
            for (size_t i = 0; i < groundTruth.size(); ++i) {
                auto c = Value::subtract(groundTruth.at(i) , prediction.at(i));
                auto b = Value::multiply(c, c);
                loss = Value::add(loss, b);
            }
            return loss;
        }
    };


    class MeanSquaredErrorFor1DPixels : public AbstractLoss<Tensor2D>{
    public:
        MeanSquaredErrorFor1DPixels () = default;

        /**
         * @brief Computes the mean squared error loss for 1D pixel values.
         * 
         * @param groundTruth The true pixel values.
         * @param prediction The predicted pixel values.
         * @return ValuePtr The computed mean squared error loss.
         */
        ValuePtr operator()(const Tensor2D& groundTruth, const Tensor2D& prediction) override{
            // Calculate loss
            auto loss = Value::create(0.0f);
            const size_t  maxSize = prediction[0].size();
            assert(groundTruth.size() == prediction.size());
            for (size_t i = 0; i < maxSize; ++i) {
                auto c = Value::subtract(groundTruth.at(0,i) , prediction.at(0,i));
                auto b = Value::multiply(c, c);
                loss = Value::add(loss, b);
            }
            return loss;
        }
    };
}
