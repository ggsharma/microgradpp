#pragma once

#include "core/Sequential.hpp"


using microgradpp::core::Sequential;

namespace microgradpp{
    class Tensor;
}
namespace microgradpp::base{

    class BaseMultiLayerPerceptron{
    private:
        Sequential _baseSequential;
    public:

        BaseMultiLayerPerceptron(const Sequential& sequential) :_baseSequential(sequential) {};

        void print(){
            this->_baseSequential.print();
        }

        void printParameters(){
            this->_baseSequential.printParameters();
        }

        std::vector<ValuePtr> operator()(std::vector<ValuePtr> input){
            return this->_baseSequential(input);
        }

        virtual Tensor forward(Tensor input) = 0;
    };
}



// mlp(input)