#pragma once
#include "MppCore.hpp"
#include "Activation.hpp"
#include <iostream>
#include <cassert>
namespace microgradpp::core {
    class CoreReLU : public MppCore {
    public:
        void print() const override final{
            std::cout << "ReLU Layer"<< std::endl;
        };

        std::vector<ValuePtr> operator()(const std::vector<ValuePtr>& in) override{
            const auto& activationFcn = Activation::mActivationFcn.at(ActivationType::RELU);
            std::vector<ValuePtr> out;
            for(const auto& e : in){
                out.push_back(activationFcn(e));
            }
            return out;
        }
    };
}