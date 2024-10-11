#pragma once
#include <vector>

#include "CoreReLU.hpp"
#include "CoreLinear.hpp"

namespace microgradpp{
    class Tensor;
}

namespace microgradpp::core{

    class Sequential{
    private:
        std::vector<std::shared_ptr<MppCore>> _layerSequence;
    public:
        Sequential(std::vector<std::shared_ptr<MppCore>> layerSequence):_layerSequence(std::move(layerSequence)){};

        std::vector<ValuePtr> operator ()(const std::vector<ValuePtr>& input){
            auto result = input;
            for(auto& layer : _layerSequence){
                auto out = layer->operator()(result);
                result = out;
            }
            return result;
        }

        void printParameters(){
            for(const auto& layerSeq: _layerSequence){
                layerSeq->printParameters();
            }
        }

        void print(){
            for(const auto& layerSeq: _layerSequence){
                layerSeq->print();
            }
        }
    };
}
