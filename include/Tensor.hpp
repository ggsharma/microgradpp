/*
 * @file: Tensor.hpp
 * @brief: Defines Tensor1D and 2D class that is the main data structure to perform mathematical operations
 *
 * This file is part of the microgradpp project.
 *
 * Created by: Gautam Sharma
 * Created on: August, 2024
 * Last Modified: Oct 12, 2024
 * License: MIT
 *
 * Copyright (c) 2024 Gautam Sharma. All rights reserved.
 * Unauthorized distributing of this file for commercial use, via any medium, is strictly prohibited.
 *
 */

#ifndef MICROGRADPP_TENSOR_HPP
#define MICROGRADPP_TENSOR_HPP

#include "Value.hpp"
#include <initializer_list>
#include <vector>
#include <memory>
#include <iostream>


namespace microgradpp {



    template<class T>
    class BaseTensor{
    protected:
        T tensor;
    public:
        // Provide begin() and end() methods to allow range-based for loop
        __MICROGRADPP_NO_DISCARD__
         auto begin() {
            return this->tensor.begin();
        }

        __MICROGRADPP_NO_DISCARD__
         auto end() {
            return this->tensor.end();
        }

        __MICROGRADPP_NO_DISCARD__
         auto begin() const {
            return this->tensor.begin();
        }

        __MICROGRADPP_NO_DISCARD__
         auto end() const {
            return this->tensor.end();
        }

        __MICROGRADPP_NO_DISCARD__
        size_t size() const{
            return this->tensor.size();
        }

        void reset(){
            this->tensor.clear();
        }

        void reserve(size_t size){
            this->tensor.reserve(size);
        }

        using iterator = typename T::iterator;
        using const_iterator = typename T::const_iterator;

        void insert(iterator a, const_iterator b, const_iterator c) {
            this->tensor.insert(a, b, c);
        }

    };

    // Internal Use
    template<class T>
    class _Tensor1D : public BaseTensor<T>{
    public:

        _Tensor1D() = default;

        explicit _Tensor1D(const std::vector<float>& input){
            this->tensor.reserve(input.size());
            for (const auto& value : input){
                this->tensor.emplace_back(Value::create(value));
            }
        }

        // Overload output stream
        friend std::ostream & operator << (std::ostream &os, const _Tensor1D &tensor) {
            for (const auto& col : tensor.tensor) {
                os << col;
            }
            return os;
        }

        void zeroGrad(){
            for(auto& value: this->tensor){
                    value->grad = 0.0;
            }
        }

        /*
         * idx: col index
         */
        ValuePtr operator[](const size_t idx) const{
            return accessElement(idx);
        }

        /*
         * idx: col index
         */
        __MICROGRADPP_NO_DISCARD__
        ValuePtr at(const size_t idx) const{
            return accessElement(idx);
        }

        void push_back(const ValuePtr& value){
            this->tensor.emplace_back(value);
        }

        void emplace_back(const ValuePtr& value){
            this->tensor.emplace_back(value);
        }
    private:
        ValuePtr accessElement(size_t idx) const {
            if (idx >= this->tensor.size()) {
                throw std::out_of_range("Accessing Tensor1D out of bounds");
            }
            return this->tensor[idx];
        }

    };


    // Type trait for extracting std::vector<ValuePtr> from T
    template<typename T>
    struct ExtractValuePtrVector {
        using type = std::vector<ValuePtr>; // Default type if not matched
    };

    // Specialization for _Tensor1D types
    template<typename U>
    struct ExtractValuePtrVector<std::vector<_Tensor1D<std::vector<U>>>> {
    using type = _Tensor1D<std::vector<ValuePtr>>;
};


    // Internal Use
    template<class T>
    class _Tensor2D : public BaseTensor<T> {
    public:
        using Tensor1D_t = typename ExtractValuePtrVector<T>::type;;

        _Tensor2D() = default;

        // Constructor for a vector of initializer lists of doubles
        _Tensor2D(const std::initializer_list<std::initializer_list<float>>& input) {
            for (const auto& list : input) {
                Tensor1D_t subTensor;
                for (auto& value : list) {
                    subTensor.emplace_back(Value::create(value));
                }
                this->tensor.emplace_back(subTensor);
            }
        }

        // Constructor for a vector of initializer lists of doubles
        // make a flattened Tensor
        _Tensor2D(const std::vector<float>& input) {
            Tensor1D_t subTensor(input);
            this->tensor.emplace_back(subTensor);
        }


        // Overload output stream
        friend std::ostream & operator << (std::ostream &os, const _Tensor2D &tensor) {
            for (const auto& row : tensor.tensor) {
                for (const auto& val : row) {
                    os << val;  // Assuming you want to print the data value
                }
            }
            return os;
        }

        void zeroGrad(){
            for(auto& subTensor: this->tensor){
                for(auto& value: subTensor){
                    value->grad = 0.0;
                }
            }
        }


        /*
         * idx: row index
         */
        Tensor1D_t operator[](const size_t idx) const{
            if(this->tensor.size() <= idx){
                throw std::invalid_argument("Accessing a Tensor out of bounds");
            }
            return this->tensor[idx];
        }


        /*
         * idx: row index
         * jdx: col index
         */
        __MICROGRADPP_NO_DISCARD__
        ValuePtr at(const size_t idx, const size_t jdx = 0) const{
            if(this->tensor.size() <= idx || this->tensor[idx].size() <= jdx){
                throw std::invalid_argument("Accessing a Tensor out of bounds");
            }
            return this->tensor[idx][jdx];
        }

        void push_back(const Tensor1D_t& value){
            this->tensor.emplace_back(value);
        }

        __MICROGRADPP_NO_DISCARD__
        size_t size() const{
            return this->tensor.size();
        }

    };


    typedef _Tensor1D<std::vector<ValuePtr>> Tensor1D;
    typedef _Tensor2D<std::vector<Tensor1D>> Tensor2D; // std::vector<std::vector<ValuePtr>>


    class Tensor {
    public:
        std::vector<std::vector<ValuePtr>> tensor;

        Tensor() = default;

         Tensor(const std::initializer_list<float>& input){
            for (const auto& value : input){
                std::vector<ValuePtr> subTensor;
                subTensor.emplace_back(Value::create(value));
                tensor.emplace_back(subTensor);
            }
        }

        explicit Tensor(const std::vector<float>& input){
            std::vector<ValuePtr> subTensor;
            subTensor.reserve(input.size());
            for (const auto& value : input){
                subTensor.emplace_back(Value::create(value));
            }
            tensor.emplace_back(subTensor);
        }

        // Constructor for a vector of initializer lists of doubles
        Tensor(const std::initializer_list<std::initializer_list<float>>& input) {
            for (const auto& list : input) {
                std::vector<ValuePtr> subTensor;
                for (auto& value : list) {
                    subTensor.emplace_back(Value::create(value));
                }
                tensor.emplace_back(subTensor);
            }
        }

        // Provide begin() and end() methods to allow range-based for loop
        __MICROGRADPP_NO_DISCARD__
        auto begin() {
            return tensor.begin();
        }

        __MICROGRADPP_NO_DISCARD__
        auto end() {
            return tensor.end();
        }

        __MICROGRADPP_NO_DISCARD__
        auto begin() const {
            return tensor.begin();
        }

        __MICROGRADPP_NO_DISCARD__
        auto end() const {
            return tensor.end();
        }

        // Overload output stream
        friend std::ostream & operator << (std::ostream &os, const Tensor &tensor) {
            for (const auto& row : tensor.tensor) {
                for (const auto& val : row) {
                    os << val;  // Assuming you want to print the data value
                }
            }
            return os;
        }

        void zeroGrad(){
            for(auto& subTensor: tensor){
                for(auto& value: subTensor){
                    value->grad = 0.0;
                }
            }
        }

        void reset(){
            tensor.clear();
        }

        /*
         * idx: row index
         */
        std::vector<ValuePtr> operator[](const size_t idx) const{
            if(tensor.size() <= idx){
                throw std::invalid_argument("Accessing a Tensor out of bounds");
            }
            return tensor[idx];
        }


        /*
         * idx: row index
         * jdx: col index
         */
        __MICROGRADPP_NO_DISCARD__
        ValuePtr at(const size_t idx, const size_t jdx = 0) const{
            if(tensor.size() <= idx || tensor[idx].size() <= jdx){
                throw std::invalid_argument("Accessing a Tensor out of bounds");
            }
            return tensor[idx][jdx];
        }

        void push_back(const std::vector<ValuePtr>& value){
            std::vector<ValuePtr> subTensor;
            std::copy(value.begin(), value.end(), std::back_inserter(subTensor));
            this->tensor.emplace_back(subTensor);
        }

        __MICROGRADPP_NO_DISCARD__
        size_t size() const{
            return tensor.size();
        }



        // Web
//       __WEB__ float web_at(const size_t idx, const size_t jdx){
//            if(tensor.size() <= idx || tensor[idx].size() <= jdx){
//                throw std::invalid_argument("Accessing a Tensor out of bounds");
//            }
//            return tensor[idx][jdx]->data;
//         }

    };
}

#endif //MICROGRADPP_TENSOR_HPP
