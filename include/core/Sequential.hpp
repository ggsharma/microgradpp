#pragma once
#include <vector>

#include "CoreReLU.hpp"
#include "CoreLinear.hpp"
#include "TypeDefs.hpp"


namespace microgradpp::core{

    class Sequential{
    private:
        std::vector<std::shared_ptr<MppCore>> _layerSequence;
    public:
        Sequential(std::vector<std::shared_ptr<MppCore>> layerSequence):_layerSequence(std::move(layerSequence)){};

        Tensor1D operator ()(const Tensor1D& input){
            auto result = input;
            for(auto& layer : _layerSequence){
                auto out = layer->operator()(result);
                result = out;
            }
            return result;
        }


        __MICROGRADPP_NO_DISCARD__
        std::vector<Value*> parameters() const{
            std::vector<Value*> params;
            if(params.empty()) {
                for (const auto &layerSeq: _layerSequence) {
                    for (const auto &p: layerSeq->parameters()) {
                        params.push_back(p);
                    }
                }
            }
            return params;
        }

        void update(float learningRate){
            for (auto &p: this->parameters()) {
                p->data += (float)((float)-learningRate * (float)p->grad);
            }
        }

        void printParameters(){
            for(const auto& layerSeq: _layerSequence){
                layerSeq->printParameters();
            }
        }

        void zeroGrad(){
            for(const auto& layerSeq: _layerSequence){
                layerSeq->zeroGrad();
            }
        }

        void print(){
            for(const auto& layerSeq: _layerSequence){
                layerSeq->print();
            }
        }
    };
}
