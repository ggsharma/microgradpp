//
// Created by Gautam Sharma on 6/30/24.
//

#ifndef MICROGRADPP_NEURON_HPP
#define MICROGRADPP_NEURON_HPP

#include "Value.hpp"
#include <memory>
#include <vector>
#include <random>
#include <stdio.h>
#include <numeric> // for std::inner_product

namespace microgradpp{
    // Function to generate a random float between -1 and 1
    double getRandomFloat() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(-1, 1);
        return dis(gen);
    }

    class Neuron{
    private:
        std::vector<std::shared_ptr<Value>> weights;
        std::shared_ptr<Value> bias = nullptr;
    public:
        Neuron(size_t nin){
            for(size_t idx = 0; idx < nin; ++idx){
                this->weights.emplace_back(Value::create(getRandomFloat()));
            }
            this->bias = Value::create(0);
        }

        // For testing
        Neuron(size_t nin, double val){
            for(size_t idx = 0; idx < nin; ++idx){
                this->weights.emplace_back(Value::create(val));
            }
            this->weights[0]->label = "w1";
            this->weights[1]->label = "w2";
            this->bias = Value::create(val);
            this->bias->label = "bias";
        }

        void zeroGrad(){
            auto params = this->parameters();
            for(auto&p : params){
                p->grad = 0;
            }
        }

        // Dot product of a Neurons weights with the input
        std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& x) const{
            // Ensure both vectors are of the same size
            if (x.size() != weights.size()) {
                throw std::invalid_argument("Vectors must be of the same length");
            }

            auto sum = Value::create(0.0);

            for(size_t idx = 0; idx<weights.size() ; ++idx){
                sum += x[idx] * weights[idx];
            }

            sum += this->bias;

            auto res = sum->relu();

            return res;
        }

        std::vector<std::shared_ptr<Value>> parameters() const{
            std::vector<std::shared_ptr<Value>> out;
            std::copy(this->weights.begin(), this->weights.end(), std::back_inserter(out));
            out.emplace_back(this->bias);
            return out;
        }

        void printParameters(){
            printf("Number of Parameters: %d \n", (int)weights.size() + 1);
            for(const auto&param : weights){
                printf("%f, %f", param->data,param->grad);
                printf("\n");
            }
            printf("%f, %f", this->bias->data,this->bias->grad);
            printf("\n");
            printf("\n");
        }

        size_t getParametersSize() const{
            return weights.size() + 1;
        }

    };
}



#endif //MICROGRADPP_NEURON_HPP
