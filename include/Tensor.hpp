//
// Created by Gautam Sharma on 7/7/24.
//

#ifndef MICROGRADPP_TENSOR_HPP
#define MICROGRADPP_TENSOR_HPP

#include "Value.hpp"
#include <initializer_list>
#include <vector>
#include <memory>
#include <iostream>

namespace microgradpp {
    class Tensor {
    public:
        std::vector<std::vector<std::shared_ptr<Value>>> tensor;

        Tensor() = default;

        Tensor(const std::initializer_list<double>& input){
            for (const auto& value : input){
                   std::vector<std::shared_ptr<Value>> subTensor;
                   subTensor.emplace_back(std::make_shared<Value>(value));
                   tensor.emplace_back(subTensor);
            }
        }

        Tensor(const std::vector<double>& input){
            std::vector<std::shared_ptr<Value>> subTensor;
            for (const auto& value : input){
                subTensor.emplace_back(std::make_shared<Value>(value));
            }
            tensor.emplace_back(subTensor);
        }


        Tensor(const std::vector<float>& input){
            std::vector<std::shared_ptr<Value>> subTensor;
            for (const auto& value : input){
                subTensor.emplace_back(std::make_shared<Value>(static_cast<double>(value)));
            }
            tensor.emplace_back(subTensor);
        }

        // Constructor for a vector of initializer lists of doubles
        Tensor(const std::initializer_list<std::initializer_list<double>>& input) {
            for (const auto& list : input) {
                std::vector<std::shared_ptr<Value>> subTensor;
                for (auto& value : list) {
                    subTensor.emplace_back(std::make_shared<Value>(value));
                }
                tensor.emplace_back(subTensor);
            }
        }

        // Provide begin() and end() methods to allow range-based for loop
        auto begin() {
            return tensor.begin();
        }

        auto end() {
            return tensor.end();
        }

        auto begin() const {
            return tensor.begin();
        }

        auto end() const {
            return tensor.end();
        }

        // Overload output stream
        friend std::ostream & operator << (std::ostream &os, const Tensor &tensor) {
            for (const auto& row : tensor.tensor) {
                for (const auto& val : row) {
                    os << val;  // Assuming you want to print the data value
                }
                os << std::endl;
            }
            return os;
        }

        void zeroGrad(){
            for(const auto& subTensor: tensor){
                for(const auto& value: subTensor){
                    value->grad = 0.0;
                }
            }
        }


        /*
         * idx: row index
         */
         std::vector<std::shared_ptr<Value>> operator[](const size_t idx) const{
            if(tensor.size() <= idx){
                throw std::invalid_argument("Accessing a Tensor out of bounds");
            }
            return tensor[idx];
        }


        /*
         * idx: row index
         * jdx: col index
         */
         std::shared_ptr<Value> at(const size_t idx, const size_t jdx) const{
            if(tensor.size() <= idx || tensor[idx].size() <= jdx){
                throw std::invalid_argument("Accessing a Tensor out of bounds");
            }
            return tensor[idx][jdx];
        }

        void push_back(const std::vector<std::shared_ptr<Value>>& value){
            std::vector<std::shared_ptr<Value>> subTensor;
            std::copy(value.begin(), value.end(), std::back_inserter(subTensor));
            this->tensor.emplace_back(subTensor);
         }


        size_t size() const{
            return tensor.size();
        }

    };
}

#endif //MICROGRADPP_TENSOR_HPP
