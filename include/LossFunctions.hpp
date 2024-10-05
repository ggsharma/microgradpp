#pragma once
// stdlibs
#include <cassert>

#include "AbstractLoss.hpp"
#include "Tensor.hpp"

namespace microgradpp::loss{

    class MeanSquaredError : public AbstractLoss<Tensor>{
    public:
        MeanSquaredError () = default;
        ValuePtr operator()(const Tensor& groundTruth, const Tensor& prediction) override{
            // Calculate loss
            auto loss = Value::create(0.0f);
            assert(groundTruth.size() == prediction.size());
            for (size_t i = 0; i < groundTruth.size(); ++i) {
                auto c = Value::subtract(groundTruth.at(i) , prediction.at(i));
                auto b = Value::multiply(c, c);
                loss = Value::add(loss, b);
            }
            return loss;
        }
    };


    class MeanSquaredErrorFor1DPixels : public AbstractLoss<Tensor>{
    public:
        MeanSquaredErrorFor1DPixels () = default;
        ValuePtr operator()(const Tensor& groundTruth, const Tensor& prediction) override{
            // Calculate loss
            auto loss = Value::create(0.0f);
            const size_t  maxSize = prediction[0].size();
            assert(groundTruth.size() == prediction.size());
            for (size_t i = 0; i < maxSize; ++i) {
                auto c = Value::subtract(groundTruth.at(0,i) , prediction.at(0,i));
                auto b = Value::multiply(c, c);
                loss = Value::add(loss, b);
            }
            return loss;
        }
    };
}