#pragma once

#include <iostream>
#include <cassert>

#include "MppCore.hpp"
#include "Activation.hpp"


namespace microgradpp::core {
    class CoreReLU : public MppCore {
    public:

        void print() const override final{
            std::cout << "ReLU Layer"<< std::endl;
        };

        Tensor1D operator()(const Tensor1D& in) override{
            const auto& activationFcn = Activation::mActivationFcn.at(ActivationType::RELU);
            Tensor1D out;
            for(const auto& value : in){
                out.push_back(activationFcn(value));
            }
            return out;
        }
    };
}