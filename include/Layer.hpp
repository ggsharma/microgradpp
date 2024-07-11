//
// Created by Gautam Sharma on 7/1/24.
//

#ifndef MICROGRADPP_LAYER_HPP
#define MICROGRADPP_LAYER_HPP

#include <vector>
#include <algorithm>

#include "Neuron.hpp"



namespace microgradpp {

class Layer{
private:
    std::vector<Neuron> neurons;
public:
    Layer(size_t nin , size_t nout, const ActivationType& activation = ActivationType::SIGMOID){
        for(size_t idx = 0; idx < nout; ++idx){
            this->neurons.emplace_back(nin, activation);
        }
    }

    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x) const{
        std::vector<std::shared_ptr<Value>> out;
        out.reserve(this->neurons.size());
        std::for_each(this->neurons.begin(), this->neurons.end(), [&out, &x]( const auto&neuron){
            out.emplace_back(neuron(x));
        });
        return out;
    }

    void zeroGrad(){
        for(auto& neuron: this->neurons){
            neuron.zeroGrad();
        }
    }

    std::vector<std::shared_ptr<Value>> parameters() const{
        std::vector<std::shared_ptr<Value>> out;
        for(const auto& neuron : neurons){
            for(const auto& p : neuron.parameters()){
                out.push_back(p);
            }
        }
        return out;
    }
};

class MLP{
private:
    std::vector<size_t> sizes;
    std::vector<Layer> layers;
    double learningRate;
public:
    MLP(size_t nin, std::vector<size_t> nouts, const double learningRate=0.0025):learningRate(learningRate){
        sizes.reserve(4);
        sizes.push_back(nin);
        std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes) );
        for(size_t idx=0; idx < sizes.size() - 1 ; ++idx){
            layers.emplace_back(sizes[idx], sizes[idx+1]);
        }
    }

    std::vector<std::shared_ptr<Value>> convertToValue(const std::vector<float>& input){
        std::vector<std::shared_ptr<Value>> out;
        for(size_t idx=0; idx<input.size(); ++idx){
            out.emplace_back(Value::create(input[idx]));
        }
        return out;
    }

    void update(){
        for (auto &p: this->parameters()) {
            p->data += (double)((double)-this->learningRate * (double)p->grad);
        }
    }

    std::vector<float> convertFromValue(const std::vector<std::shared_ptr<Value>>& input){
        std::vector<float> out;
        for(size_t idx=0; idx<input.size(); ++idx){
            out.emplace_back(input[idx]->data);
        }
        return out;
    }

    void zeroGrad(){
        for(auto& layer: this->layers){
            layer.zeroGrad();
        }
    }

    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& input){
        std::vector<std::shared_ptr<Value>> x = input;
        for(const auto& layer : this->layers){
             auto y  = layer(x);
             x = y;

        }
        return x;
    }

    void printParameters(){
        const auto params = this->parameters();
        printf("Num parameters: %d\n", (int)params.size());
        for(const auto& p : params){
            std::cout<< &p << " ";
            printf("[data=%f,grad=%lf]\n", p->data, p->grad);
        }
        printf("\n");
    }

    std::vector<std::shared_ptr<Value>> parameters() const{
        std::vector<std::shared_ptr<Value>> out;
        for(const auto& layer : this->layers){
            for(const auto& p : layer.parameters()){
                out.push_back(p);
            }
        }

        return out;
    }

};

}

#endif //MICROGRADPP_LAYER_HPP
