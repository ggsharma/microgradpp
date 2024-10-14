/**
 *  @file CoreLinear.hpp
 *  @brief Defines the CoreLinear class for linear transformation in neural networks.
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
 *  The `CoreLinear` class provides functionality for a single linear (fully connected) layer
 *  in a neural network, using neurons as its basic units. It offers methods for forward
 *  computation, parameter management, and gradient resetting.
 */

#pragma once

// Standard libraries
#include <iostream>
#include <vector>
#include <algorithm>

// microgradpp libraries
#include "MppCore.hpp"
#include "Neuron.hpp"
#include "Value.hpp"
#include "TypeDefs.hpp"

namespace microgradpp::core {

    /**
     * @class CoreLinear
     * @brief Represents a linear (fully connected) layer with configurable input and output dimensions.
     *
     * The `CoreLinear` class is derived from `MppCore` and implements the forward operation,
     * zero gradient function, and parameter retrieval for a linear neural network layer.
     */
    class CoreLinear : public MppCore {
    private:
        size_t _nin;           /**< Number of input neurons */
        size_t _nout;          /**< Number of output neurons */
        std::vector<Neuron> _neurons; /**< Vector containing neuron objects for the layer */

    public:
        /**
         * @brief Constructs a CoreLinear layer with specified input and output sizes.
         * @param nin Number of inputs to each neuron.
         * @param nout Number of neurons (output size).
         */
        CoreLinear(size_t nin, size_t nout): _nin(nin), _nout(nout){
            for(size_t idx = 0; idx < nout; ++idx){
                this->_neurons.emplace_back(nin);
            }
        }

        /**
         * @brief Performs the forward pass of the linear layer using input tensor.
         * @param x Input tensor of size `nin`.
         * @return Tensor1D Output tensor of size `nout` after applying each neuron's forward pass.
         */
        Tensor1D operator()(const Tensor1D& x) override {
            Tensor1D out;
            out.reserve(this->_neurons.size());
            std::for_each(this->_neurons.begin(), this->_neurons.end(), [&out, x=x](   auto neuron)mutable{
                out.emplace_back(neuron(x));
            });
            return out;
        }

        /**
         * @brief Prints layer information, displaying the input-output dimensions.
         */
        void print() const override final{
            std::cout << _nin << " X " << _nout << " Linear Layer"<< std::endl;
        };

        /**
         * @brief Resets gradients for all neurons in the layer to zero.
         */
        void zeroGrad() override final{
            for(auto& neuron: this->_neurons){
                neuron.zeroGrad();
            }
        }

        /**
         * @brief Retrieves pointers to the parameters (weights and biases) of all neurons in the layer.
         * @return std::vector<Value*> Vector of pointers to `Value` objects representing the layer parameters.
         */
        __MICROGRADPP_NO_DISCARD__
        std::vector<Value*> parameters() const override{
            std::vector<Value*> params;
            if(params.empty()) {
                for(const auto& neuron : _neurons){
                    for(const auto& p : neuron.parameters()){
                        params.push_back(p.get());
                    }
                }
            }
            return params;
        }

        /**
         * @brief Prints each neuron's parameters, including data and gradient values.
         */
        void printParameters() const override{
            const auto params = this->parameters();
            printf("Num parameters: %d\n", (int)params.size());
            for(const auto& p : params){
                std::cout<< &p << " ";
                printf("[data=%f,grad=%lf]\n", p->data, p->grad);
            }
            printf("\n");
        }
    };
}
