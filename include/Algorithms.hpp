/*
 * @file: Algorithms.hpp
 * @brief: Defines common algorithms like Multi layer perceptron
 *
 * This file is part of the microgradpp project.
 *
 * Created by: Gautam Sharma
 * Date: Sep 29, 2024
 * License: MIT
 *
 * Copyright (c) 2024 Gautam Sharma. All rights reserved.
 * Unauthorized copying of this file, via any medium, is strictly prohibited.
 *
 */


#pragma once

// std libs
#include <iostream>
#include <algorithm>

// Microgradpp headers
#include "Layer.hpp"
#include "TypeDefs.hpp"

namespace microgradpp::algorithms{
    class MLP{
        private:
            std::vector<size_t> sizes;
            std::vector<Layer> layers;
            float learningRate;
        public:
            // Image works with sigmoid
            // Preferred way to instantiate a MLP
            MLP(size_t nin, std::vector<size_t> nouts, const float learningRate=0.0025):learningRate(learningRate){
                //sizes.reserve(4);
                sizes.push_back(nin);
                std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes) );
                for(size_t idx=0; idx < sizes.size() - 1 ; ++idx){
                    layers.emplace_back(sizes[idx], sizes[idx+1],ActivationType::TANH);
                }
            }

            MLP(size_t nin, size_t nout1, size_t  nout2, const float learningRate=0.0025):learningRate(learningRate){
                //sizes.reserve(4);
                sizes.push_back(nin);
                std::vector<size_t> nouts = {nout1, nout2};
                std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes) );
                for(size_t idx=0; idx < sizes.size() - 1 ; ++idx){
                    layers.emplace_back(sizes[idx], sizes[idx+1],ActivationType::TANH);
                }
            }


            MLP(size_t nin, size_t nout1, size_t  nout2, size_t nout3, const float learningRate=0.0025):learningRate(learningRate){
                //sizes.reserve(4);
                sizes.push_back(nin);
                std::vector<size_t> nouts = {nout1, nout2, nout3};
                std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes) );
                for(size_t idx=0; idx < sizes.size() - 1 ; ++idx){
                    layers.emplace_back(sizes[idx], sizes[idx+1],ActivationType::TANH);
                }
            }


            ~MLP(){
                std::cout << "microgradpp::algorithms::MLP destroyed" << std::endl;
            }


            void update() const{
                for (auto &p: this->parameters()) {
                    p->data += (float)((float)-this->learningRate * (float)p->grad);
                }
            }

            void zeroGrad(){
                for(auto& layer: this->layers){
                    layer.zeroGrad();
                }
            }

            Tensor1D forward(const Tensor1D& input){
                return this->operator()(input);
            }

            Tensor1D operator()(const Tensor1D& input){
                Tensor1D x = input;
                for( auto& layer : this->layers){
                    auto y  = layer(x);
                    x = y;
                }
                return x;
//                std::vector<ValuePtr> x = input;
//                std::for_each(this->layers.begin(), this->layers.end(), [&x](auto& layer){
//                    auto y  = layer(x);
//                    x = y;
//                });
//                return x;
            }

            void printParameters() const{
                const auto params = this->parameters();
                printf("Num parameters: %d\n", (int)params.size());
                for(const auto& p : params){
                    std::cout<< &p << " ";
                    printf("[data=%f,grad=%lf]\n", p->data, p->grad);
                }
                printf("\n");
            }

            __MICROGRADPP_NO_DISCARD__
            std::vector<Value*> parameters() const{
                std::vector<Value*> params;
                if(params.empty()) {
                    for (const auto &layer: this->layers) {
                        for (const auto &p: layer.parameters()) {
                            params.push_back(p);
                        }
                    }
                }
                return params;
            }

    };
}