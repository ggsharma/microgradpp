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
        Layer(size_t nin , size_t nout, const ActivationType& activation = ActivationType::SIGMOID){
            for(size_t idx = 0; idx < nout; ++idx){
                this->neurons.emplace_back(nin, activation);
            }
        }

        std::vector<ValuePtr> operator()(const std::vector<ValuePtr>& x) {
            std::vector<ValuePtr> out;
            out.reserve(this->neurons.size());
            std::for_each(this->neurons.begin(), this->neurons.end(), [&out, x = x](   auto neuron)mutable{
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


    class MLP{
    private:
        std::vector<size_t> sizes;
        std::vector<Layer> layers;
        float learningRate;
    public:
        // Preferred way to instantiate a MLP
        MLP(size_t nin, std::vector<size_t> nouts, const float learningRate=0.0025):learningRate(learningRate){
            sizes.reserve(4);
            sizes.push_back(nin);
            std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes) );
            for(size_t idx=0; idx < sizes.size() - 1 ; ++idx){
                layers.emplace_back(sizes[idx], sizes[idx+1],ActivationType::SIGMOID);
            }
        }

        MLP(size_t nin, size_t nout1, size_t  nout2, const float learningRate=0.0025):learningRate(learningRate){
            sizes.reserve(4);
            sizes.push_back(nin);
            std::vector<size_t> nouts = {nout1, nout2};
            std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes) );
            for(size_t idx=0; idx < sizes.size() - 1 ; ++idx){
                layers.emplace_back(sizes[idx], sizes[idx+1],ActivationType::SIGMOID);
            }
        }

        MLP(size_t nin, size_t nout1, size_t  nout2, size_t nout3, const float learningRate=0.0025):learningRate(learningRate){
            sizes.reserve(4);
            sizes.push_back(nin);
            std::vector<size_t> nouts = {nout1, nout2, nout3};
            std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes) );
            for(size_t idx=0; idx < sizes.size() - 1 ; ++idx){
                layers.emplace_back(sizes[idx], sizes[idx+1],ActivationType::SIGMOID);
            }
        }


        ~MLP(){
            std::cout << "mlp destroyed" << std::endl;
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


        std::vector<ValuePtr> operator()(const std::vector<ValuePtr>& input){
            std::vector<ValuePtr> x = input;
            for( auto& layer : this->layers){
                auto y  = layer(x);
                x = y;
            }
            return x;
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

};

#endif //MICROGRADPP_LAYER_HPP
