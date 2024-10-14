/**
 *  @file Algorithms.hpp
 *  @brief Defines common algorithms like Multi-Layer Perceptron (MLP).
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
 *  Date: September 29, 2024
 *
 *  @details
 *  This header file contains the definition of the MLP (Multi-Layer Perceptron) class,
 *  which implements a feedforward neural network with support for various layer sizes
 *  and activation functions. The class provides methods for forward propagation,
 *  updating parameters, and zeroing gradients.
 */

#pragma once

// Standard libraries
#include <iostream>
#include <algorithm>

// Microgradpp headers
#include "Layer.hpp"
#include "TypeDefs.hpp"

namespace microgradpp::algorithms {
    class MLP {
    private:
        std::vector<size_t> sizes;        // Sizes of each layer in the MLP
        std::vector<Layer> layers;        // Layers of the MLP
        float learningRate;               // Learning rate for parameter updates

    public:
        /**
         * @brief Constructs an MLP with specified input size and output sizes.
         *
         * @param nin Input size.
         * @param nouts Vector of output sizes for each layer.
         * @param learningRate Learning rate for parameter updates (default is 0.0025).
         */
        MLP(size_t nin, std::vector<size_t> nouts, const float learningRate = 0.0025)
                : learningRate(learningRate) {
            sizes.push_back(nin);
            std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes));
            for (size_t idx = 0; idx < sizes.size() - 1; ++idx) {
                layers.emplace_back(sizes[idx], sizes[idx + 1], ActivationType::TANH);
            }
        }

        /**
         * @brief Constructs an MLP with specified input size and two output sizes.
         *
         * @param nin Input size.
         * @param nout1 Size of the first output layer.
         * @param nout2 Size of the second output layer.
         * @param learningRate Learning rate for parameter updates (default is 0.0025).
         */
        MLP(size_t nin, size_t nout1, size_t nout2, const float learningRate = 0.0025)
                : learningRate(learningRate) {
            sizes.push_back(nin);
            std::vector<size_t> nouts = {nout1, nout2};
            std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes));
            for (size_t idx = 0; idx < sizes.size() - 1; ++idx) {
                layers.emplace_back(sizes[idx], sizes[idx + 1], ActivationType::TANH);
            }
        }

        /**
         * @brief Constructs an MLP with specified input size and three output sizes.
         *
         * @param nin Input size.
         * @param nout1 Size of the first output layer.
         * @param nout2 Size of the second output layer.
         * @param nout3 Size of the third output layer.
         * @param learningRate Learning rate for parameter updates (default is 0.0025).
         */
        MLP(size_t nin, size_t nout1, size_t nout2, size_t nout3, const float learningRate = 0.0025)
                : learningRate(learningRate) {
            sizes.push_back(nin);
            std::vector<size_t> nouts = {nout1, nout2, nout3};
            std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes));
            for (size_t idx = 0; idx < sizes.size() - 1; ++idx) {
                layers.emplace_back(sizes[idx], sizes[idx + 1], ActivationType::TANH);
            }
        }

        /**
         * @brief Destructor for MLP.
         */
        ~MLP() {
            std::cout << "microgradpp::algorithms::MLP destroyed" << std::endl;
        }

        /**
         * @brief Updates the parameters of the MLP using the gradients.
         */
        void update() const {
            for (auto& p : this->parameters()) {
                p->data += (float)((float)-this->learningRate * (float)p->grad);
            }
        }

        /**
         * @brief Resets the gradients of all layers to zero.
         */
        void zeroGrad() {
            for (auto& layer : this->layers) {
                layer.zeroGrad();
            }
        }

        /**
         * @brief Performs forward propagation with the given input.
         *
         * @param input Input tensor for the MLP.
         * @return Tensor1D Result of the forward pass.
         */
        Tensor1D forward(const Tensor1D& input) {
            return this->operator()(input);
        }

        /**
         * @brief Performs forward propagation through the MLP.
         *
         * @param input Input tensor for the MLP.
         * @return Tensor1D Result of the forward pass.
         */
        Tensor1D operator()(const Tensor1D& input) {
            Tensor1D x = input;
            for (auto& layer : this->layers) {
                auto y = layer(x);
                x = y;
            }
            return x;
        }

        /**
         * @brief Prints the parameters and their gradients.
         */
        void printParameters() const {
            const auto params = this->parameters();
            printf("Num parameters: %d\n", (int)params.size());
            for (const auto& p : params) {
                std::cout << &p << " ";
                printf("[data=%f, grad=%lf]\n", p->data, p->grad);
            }
            printf("\n");
        }

        /**
         * @brief Retrieves the parameters of the MLP.
         *
         * @return Vector of pointers to Value objects representing the parameters.
         */
        __MICROGRADPP_NO_DISCARD__
        std::vector<Value*> parameters() const {
            std::vector<Value*> params;
            if (params.empty()) {
                for (const auto& layer : this->layers) {
                    for (const auto& p : layer.parameters()) {
                        params.push_back(p);
                    }
                }
            }
            return params;
        }
    };
}
