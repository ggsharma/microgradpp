/**
 *  @file Tensor.hpp
 *  @brief Defines Tensor1D and 2D classes that are the main data structures for performing mathematical operations.
 *
 *  This file is part of the microgradpp project, a lightweight C++ library for neural
 *  network training and inference.
 *
 *  @section License
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the MIT License.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 *  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 *  PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 *  FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 *  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
 *
 *  @section Author
 *  Gautam Sharma
 *  Email: gautamsharma2813@gmail.com
 *  Date: October 12, 2024
 *
 *  @details
 *  This header file contains the definition of Tensor1D and Tensor2D classes used for
 *  representing and manipulating multidimensional data structures, along with methods for
 *  mathematical operations and gradient management.
 */

#ifndef MICROGRADPP_TENSOR_HPP
#define MICROGRADPP_TENSOR_HPP

// Standard headers
#include <initializer_list>
#include <vector>
#include <memory>
#include <iostream>

// mpp headers
#include "Value.hpp"


namespace microgradpp {
    /**
     * @class BaseTensor
     * @brief A base class for tensor types providing common functionalities.
     *
     * @tparam T The type of tensor (e.g., std::vector<ValuePtr>).
   */
    template<class T>
    class BaseTensor{
    protected:
        T tensor; ///< The underlying data structure for the tensor.
    public:
        // Provide begin() and end() methods to allow range-based for loop

        /**
         * @brief Returns an iterator to the beginning of the tensor.
         * @return An iterator to the first element.
         */
        __MICROGRADPP_NO_DISCARD__
        auto begin() {
            return this->tensor.begin();
        }

        /**
         * @brief Returns an iterator to the end of the tensor.
         * @return An iterator to the past-the-end element.
         */
        __MICROGRADPP_NO_DISCARD__
        auto end() {
            return this->tensor.end();
        }

        /**
         * @brief Returns a const iterator to the beginning of the tensor.
         * @return A const iterator to the first element.
         */
        __MICROGRADPP_NO_DISCARD__
        auto begin() const {
            return this->tensor.begin();
        }

        /**
         * @brief Returns a const iterator to the end of the tensor.
         * @return A const iterator to the past-the-end element.
         */
        __MICROGRADPP_NO_DISCARD__
        auto end() const {
            return this->tensor.end();
        }

        /**
         * @brief Returns the size of the tensor.
         * @return The number of elements in the tensor.
         */
        __MICROGRADPP_NO_DISCARD__
                size_t size() const {
            return this->tensor.size();
        }

        /**
         * @brief Clears the tensor.
         */
        void reset() {
            this->tensor.clear();
        }

        /**
         * @brief Reserves memory for the tensor.
         * @param size The number of elements to reserve space for.
         */
        void reserve(size_t size) {
            this->tensor.reserve(size);
        }

        using iterator = typename T::iterator;       ///< Type alias for the iterator.
        using const_iterator = typename T::const_iterator; ///< Type alias for the const iterator.

        /**
         * @brief Inserts elements into the tensor.
         * @param a The position to insert elements.
         * @param b The starting position of the elements to insert.
         * @param c The ending position of the elements to insert.
         */
        void insert(iterator a, const_iterator b, const_iterator c) {
            this->tensor.insert(a, b, c);
        }

    };

    // Internal Use
    /**
     * @class _Tensor1D
     * @brief A class representing a 1D tensor (vector) of ValuePtr objects.
     *
     * This class provides methods for tensor operations, including zeroing gradients,
     * accessing elements, and pushing back new values.
     *
     * @tparam T The type of tensor (e.g., std::vector<ValuePtr>).
     */
    template<class T>
    class _Tensor1D : public BaseTensor<T>{
    public:

        _Tensor1D() = default;

        /**
            * @brief Constructs a 1D tensor from a vector of floats.
            * @param input The vector of floats to initialize the tensor.
        */
        explicit _Tensor1D(const std::vector<float>& input){
            this->tensor.reserve(input.size());
            for (const auto& value : input){
                this->tensor.emplace_back(Value::create(value));
            }
        }

        /**
             * @brief Overloads the output stream operator for printing the tensor.
             * @param os The output stream.
             * @param tensor The tensor to print.
             * @return The output stream.
        */
        friend std::ostream & operator << (std::ostream &os, const _Tensor1D &tensor) {
            for (const auto& col : tensor.tensor) {
                os << col;
            }
            return os;
        }

        /**
           * @brief Zeros the gradients of all elements in the tensor.
        */
        void zeroGrad() {
            for (auto& value : this->tensor) {
                value->grad = 0.0;
            }
        }

        /**
             * @brief Accesses an element in the tensor using the index operator.
             * @param idx The index of the element to access.
             * @return A pointer to the accessed Value.
         */
        ValuePtr operator[](const size_t idx) const {
            return accessElement(idx);
        }

        /**
         * @brief Accesses an element in the tensor using the at method.
         * @param idx The index of the element to access.
         * @return A pointer to the accessed Value.
         */
        __MICROGRADPP_NO_DISCARD__
                ValuePtr at(const size_t idx) const {
            return accessElement(idx);
        }

        /**
         * @brief Pushes a new ValuePtr onto the tensor.
         * @param value The ValuePtr to push back.
         */
        void push_back(const ValuePtr& value) {
            this->tensor.emplace_back(value);
        }

        /**
         * @brief Emplaces a new ValuePtr onto the tensor.
         * @param value The ValuePtr to emplace back.
         */
        void emplace_back(const ValuePtr& value) {
            this->tensor.emplace_back(value);
        }
    private:
        /**
       * @brief Accesses an element by index with bounds checking.
       * @param idx The index of the element to access.
       * @return A pointer to the accessed Value.
       * @throws std::out_of_range if the index is out of bounds.
       */
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
    /**
     * @class _Tensor2D
     * @brief A class representing a 2D tensor (matrix) of ValuePtr objects.
     *
     * This class provides methods for tensor operations, including zeroing gradients,
     * accessing elements, and pushing back new rows of tensors.
     *
     * @tparam T The type of tensor (e.g., std::vector<ValuePtr>).
     */
    template<class T>
    class _Tensor2D : public BaseTensor<T> {
    public:
        using Tensor1D_t = typename ExtractValuePtrVector<T>::type; ///< Type alias for 1D tensor.

        _Tensor2D() = default;

        /**
       * @brief Constructs a 2D tensor from an initializer list of initializer lists.
       * @param input The initializer list of initializer lists to initialize the tensor.
       */
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
        /**
         * @brief Constructs a 2D tensor from a vector of floats, making a flattened tensor.
         * @param input The vector of floats to initialize the tensor.
         */
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
