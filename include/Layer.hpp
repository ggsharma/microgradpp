//
// Created by Gautam Sharma on 7/1/24.
//

#ifndef MICROGRADPP_LAYER_HPP
#define MICROGRADPP_LAYER_HPP

#include <vector>
#include <algorithm>

#include "Neuron.hpp"
#include "TypeDefs.hpp"



namespace microgradpp {

    class Layer{
    private:
        //mutable std::vector<Value*> params;
    public:
        std::vector<Neuron> neurons;
        Layer(size_t nin , size_t nout, const ActivationType& activation){
            for(size_t idx = 0; idx < nout; ++idx){
                this->neurons.emplace_back(nin, activation);
            }
        }

        Tensor1D operator()(const Tensor1D & x) {
            Tensor1D out;
            out.reserve(this->neurons.size());
            std::for_each(this->neurons.begin(), this->neurons.end(), [&out, x=x](   auto neuron)mutable{
                out.emplace_back(neuron(x));
            });
            return out;
        }

        void zeroGrad(){
            for(auto& neuron: this->neurons){
                neuron.zeroGrad();
            }
        }

        __MICROGRADPP_NO_DISCARD__
        std::vector<Value*> parameters() const{
            std::vector<Value*> params;
            if(params.empty()) {
                for(const auto& neuron : neurons){
                    for(const auto& p : neuron.parameters()){
                        params.push_back(p.get());
                    }
                }
            }
            return params;
        }

        void print() const{
            const auto params = this->parameters();
            printf("Num parameters: %d\n", (int)params.size());
            for(const auto& p : params){
                std::cout<< &p << " ";
                printf("[data=%f,grad=%lf]\n", p->data, p->grad);
            }
            printf("\n");
        }
    };
};

#endif //MICROGRADPP_LAYER_HPP
