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
    //mutable std::vector<Value*> params;
public:
    std::vector<Neuron> neurons;
    Layer(size_t nin , size_t nout, const ActivationType& activation = ActivationType::SIGMOID){
        for(size_t idx = 0; idx < nout; ++idx){
            this->neurons.emplace_back(nin, activation);
        }
    }

    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x) {
        std::vector<std::shared_ptr<Value>> out;
        out.reserve(this->neurons.size());
        std::for_each(this->neurons.begin(), this->neurons.end(), [&out, &x](  const auto&neuron){
            out.emplace_back(neuron(x));
        });
        return out;
    }

    void zeroGrad(){
        for(auto& neuron: this->neurons){
            neuron.zeroGrad();
        }
    }

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
};

class MLP{
private:
    std::vector<size_t> sizes;
    std::vector<Layer> layers;
    //std::shared_ptr<Value> loss = Value::create(0.0);
    //mutable std::vector<Value*> params;
    double learningRate;
public:
    MLP(size_t nin, std::vector<size_t> nouts, const double learningRate=0.0025):learningRate(learningRate){
        sizes.reserve(4);
        sizes.push_back(nin);
        std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes) );
        for(size_t idx=0; idx < sizes.size() - 1 ; ++idx){
            layers.emplace_back(sizes[idx], sizes[idx+1],ActivationType::RELU);
        }
    }

    MLP(size_t nin, size_t nout1, size_t  nout2, const double learningRate=0.0025):learningRate(learningRate){
        sizes.reserve(4);
        sizes.push_back(nin);
        std::vector<size_t> nouts = {nout1, nout2};
        std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes) );
        for(size_t idx=0; idx < sizes.size() - 1 ; ++idx){
            layers.emplace_back(sizes[idx], sizes[idx+1],ActivationType::TANH);
        }
    }

    MLP(size_t nin, size_t nout1, size_t  nout2, size_t nout3, const double learningRate=0.0025):learningRate(learningRate){
        sizes.reserve(4);
        sizes.push_back(nin);
        std::vector<size_t> nouts = {nout1, nout2, nout3};
        std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes) );
        for(size_t idx=0; idx < sizes.size() - 1 ; ++idx){
            layers.emplace_back(sizes[idx], sizes[idx+1],ActivationType::TANH);
        }
    }

    ~MLP(){
        std::cout << "mlp destroyed" << std::endl;
    }

    void clear(){
        layers.clear();
    }

    std::vector<std::shared_ptr<Value>> convertToValue(const std::vector<float>& input){
        std::vector<std::shared_ptr<Value>> out;
        for(size_t idx=0; idx<input.size(); ++idx){
            out.emplace_back(Value::create(input[idx]));
        }
        return out;
    }

    void backProp(){
        std::vector<std::shared_ptr<Value>> ys  = {Value::create(-0.5),Value::create(0.8),Value::create(0.5),Value::create(1)};
        std::shared_ptr<Value> loss = Value::create(0.0);
        std::vector<std::vector<std::shared_ptr<Value>>> ypred;

        std::vector<std::vector<std::shared_ptr<Value>>> xs = {{Value::create(0.2), Value::create(0.3), Value::create(-1.0)},
                                                              {Value::create(0.4), Value::create(0.3), Value::create(0.1)},
                                                              {Value::create(0.5), Value::create(0.1), Value::create(-0.1)},
                                                              {Value::create(1.0), Value::create(1.0), Value::create(-1.0)}};

        for(const auto &input: xs) {
            ypred.emplace_back(this->test(input));
        }

        // Calculate loss
        for (size_t i = 0; i < ys.size(); ++i) {
            loss += (ys.at(i) - ypred[i][0]) ^ 2;
        }

        loss->backProp();

        this->update();

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

    std::vector<std::shared_ptr<Value>> test(const std::vector<std::shared_ptr<Value>>& input){
        std::vector<std::shared_ptr<Value>> x = input;
        for( auto& layer : this->layers){
            auto y  = layer(x);
            x = y;

        }
        return x;
    }

    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& input){
        std::vector<std::shared_ptr<Value>> x = input;
        for( auto& layer : this->layers){
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
