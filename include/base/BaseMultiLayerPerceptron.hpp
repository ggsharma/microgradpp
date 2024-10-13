#pragma once

#include "core/Sequential.hpp"
#include "TypeDefs.hpp"

using microgradpp::core::Sequential;


namespace microgradpp::base{

    class BaseMultiLayerPerceptron{

        Sequential _baseSequential;

    public:
        Sequential& sequential = _baseSequential;

        BaseMultiLayerPerceptron(const Sequential& sequential) :_baseSequential(sequential) {};

        void print(){
            this->_baseSequential.print();
        }

        void printParameters(){
            this->_baseSequential.printParameters();
        }

        void zeroGrad(){
            this->_baseSequential.zeroGrad();
        }

        void update(){
            this->_baseSequential.update(this->learningRate);
        }

        Tensor1D operator()(const Tensor1D& input){
            return this->forward(input);
        }

        virtual Tensor1D forward(Tensor1D input) = 0;

    protected:
        float learningRate = 0.001;
    };
}
