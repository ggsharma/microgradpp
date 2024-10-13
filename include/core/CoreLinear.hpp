#pragma once


#include <iostream>

#include "MppCore.hpp"
#include "Neuron.hpp"
#include "Value.hpp"
#include "TypeDefs.hpp"

namespace microgradpp::core {
    class CoreLinear : public MppCore {
    private:
        size_t _nin, _nout;
        std::vector<Neuron> _neurons;

    public:
        CoreLinear(size_t nin, size_t nout): _nin(nin), _nout(nout){
            for(size_t idx = 0; idx < nout; ++idx){
                this->_neurons.emplace_back(nin);
            }
        }

        Tensor1D operator()(const Tensor1D& x) override {
            Tensor1D out;
            out.reserve(this->_neurons.size());
            std::for_each(this->_neurons.begin(), this->_neurons.end(), [&out, x=x](   auto neuron)mutable{
                out.emplace_back(neuron(x));
            });
            return out;
        }


        void print() const override final{
            std::cout << _nin << " X " << _nout << " Linear Layer"<< std::endl;
        };


        void zeroGrad() override final{
            for(auto& neuron: this->_neurons){
                neuron.zeroGrad();
            }
        }


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