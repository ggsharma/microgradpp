//
// Created by Gautam Sharma on 6/30/24.
//

#ifndef MICROGRADPP_NEURON_HPP
#define MICROGRADPP_NEURON_HPP

#include "Value.hpp"
#include "Activation.hpp"

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
        const ActivationType activation_t;
    public:
        Neuron(size_t nin, const ActivationType& activation_t): activation_t(activation_t){
            for(size_t idx = 0; idx < nin; ++idx){
                this->weights.emplace_back(Value::create(getRandomFloat()));
            }
            this->bias = Value::create(0);
        }

        // For testing
        Neuron(size_t nin, double val,const ActivationType& activation_t = ActivationType::SIGMOID):activation_t(activation_t){
            for(size_t idx = 0; idx < nin; ++idx){
                this->weights.emplace_back(Value::create(val));
            }
            this->bias = Value::create(val);
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

            // Dot product -> dot product is supported from C++20
            // TODO: need to make it more efficient
            for(size_t idx = 0; idx<weights.size() ; ++idx){
                sum += x[idx] * weights[idx];
            }

            // Add bias
            sum += this->bias;

            const auto& activationFcn = Activation::mActivationFcn[activation_t];
            auto result = activationFcn(sum);

            return result;
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
