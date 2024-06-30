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

namespace microgradpp{
    // Function to generate a random float between -1 and 1
    float getRandomFloat() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(-1.0, 1.0);
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

        void printParameters(){
            printf("Number of Parameters: %d \n", (int)weights.size() + 1);
            for(const auto&param : weights){
                printf("%f , ", param->data);
            }
            printf("\n");
        }

        size_t getParametersSize() const{
            return weights.size() + 1;
        }
    };
}



#endif //MICROGRADPP_NEURON_HPP
