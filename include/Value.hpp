#pragma once
#include <cstdio>
#include <vector>
#include <string>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <cmath>
#include <memory>
#include <functional> // std::function
#include <iomanip>

// m++ headers
#include "Autograd.hpp"

namespace microgradpp {
    class Value;
    /// Global instance of Autograd to manage the computation tape
    Autograd Autograd::global_tape;

    /**
    * @brief Custom hash function for std::shared_ptr<Value> to allow its use in unordered containers.
    */
    struct Hash {
        /**
        * @brief Generates a hash value for a shared pointer to Value.
        * @param value A shared pointer to a Value object.
        * @return A size_t representing the hash value.
        */
        size_t operator()(const std::shared_ptr<Value>& value) const;
    };


    using ValuePtr = std::shared_ptr<Value>;

    /**
     * @brief A class representing a value in the computational graph with automatic differentiation support.
     */
    class Value : public std::enable_shared_from_this<Value> {
    private:
        /**
         * @brief Constructor for Value.
         * @param data The numerical value stored.
         * @param op The operation used to create this value (optional).
         * @param id The unique identifier for this value (optional).
         */
        explicit Value(float data, std::string op, size_t id)
                : data(data), grad(0.0), op(std::move(op)), id(id) {}
    public:
        inline static size_t currentID = 0; ///< Global counter for generating unique IDs.
        float data = 0; ///< The actual numerical value.
        float grad = 0; ///< The gradient of the value (used in backpropagation).
        std::string op; ///< The operation used to create the value (e.g., +, *, etc.).
        size_t id = 0LU; ///< A unique identifier for the value.
        std::vector<ValuePtr> prev; ///< Pointers to the previous values that were inputs to this value.
        std::function<void()> backward = nullptr; ///< Function to compute the gradient during backpropagation.

        /**
        * @brief Generates a new unique ID for each value.
        * @return A size_t representing the unique ID.
        */
        static size_t generateID() {
            return ++currentID;
        }

        /**
         * @brief Factory method to create a new Value instance.
         * @param data The numerical value stored.
         * @param op The operation used to create this value (optional).
         * @param id The unique identifier for this value (optional).
         * @return A shared pointer to the created Value.
         */
        static ValuePtr create(float data, std::string op = ""){
            return std::shared_ptr<Value>(new Value(data,  std::move(op), generateID()));
        }

        /**
           * @brief Destructor for Value.
        */
        ~Value(){
            --currentID;
        }

        std::string label; ///< A label for the value (optional).
        inline static int labelIdx = 0; ///< Global counter for labels.

        const float GRADIENT_CLIP_VALUE = 1e4; ///< Gradient clipping threshold.
        const float EPSILON = 1e-7; ///< Small value to avoid numerical instability.

        //Value(const Value& v) = default;

        /**
             * @brief Clips the gradients to avoid exploding gradients.
        */
        void clip_gradients() {
            if (grad > GRADIENT_CLIP_VALUE) grad = GRADIENT_CLIP_VALUE;
            if (grad < -GRADIENT_CLIP_VALUE) grad = -GRADIENT_CLIP_VALUE;
        }


        /**
         * @brief Resets the value and its gradients.
         */
        void reset(){
            this->grad = 0.0;
            this->data = 0.0;
            this->prev.clear();
        }

        /**
         * @brief Resets the gradients to zero.
         */
        void resetGradients() {
            grad = 0;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Addition
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
          * @brief Adds two values and creates a new value.
          * @param lhs The left-hand side ValuePtr.
          * @param rhs The right-hand side ValuePtr.
          * @return A shared pointer to the new Value representing the sum.
          */
        static ValuePtr add(const ValuePtr& lhs, const ValuePtr& rhs) {
            auto out = create((float)(lhs->data + rhs->data), "+");
            out->prev = {lhs, rhs};

            Autograd::global_tape.add_entry(out, [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)]() {
                //if (auto lhs = lhs_weak.lock()) {
                lhs_weak.lock()->grad += out_weak.lock()->grad;
                //}
                //if (auto rhs = rhs_weak.lock()) {
                rhs_weak.lock()->grad += out_weak.lock()->grad;
                //}
            });

            return out;
        }

        /**
         * @brief Adds a value and a float.
         * @param lhs The left-hand side ValuePtr.
         * @param f The float to add.
         * @return A shared pointer to the new Value representing the sum.
         */
        static ValuePtr add(const ValuePtr& lhs, float f) {
            auto rhs = Value::create((float)f);
            auto out = create((float)(lhs->data + f), "+");
            out->prev = {lhs, rhs};

            Autograd::global_tape.add_entry(out, [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)]() {
                //if (auto lhs = lhs_weak.lock()) {
                lhs_weak.lock()->grad += out_weak.lock()->grad;
                //}
                //if (auto rhs = rhs_weak.lock()) {
                rhs_weak.lock()->grad += out_weak.lock()->grad;
                //}
            });

            return out;
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Multiplication
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /**
          * @brief Multiplies two values and creates a new value.
          * @param lhs The left-hand side ValuePtr.
          * @param rhs The right-hand side ValuePtr.
          * @return A shared pointer to the new Value representing the product.
          */
        static ValuePtr multiply(const ValuePtr& lhs, const ValuePtr& rhs) {
            auto out = create((float)(lhs->data * rhs->data), "*");
            out->prev = {lhs, rhs};

            Autograd::global_tape.add_entry(out,[lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)]() {
                //if (auto lhs = lhs_weak.lock()) {
                lhs_weak.lock()->grad += rhs_weak.lock()->data * out_weak.lock()->grad;
                //}
                //if (auto rhs = rhs_weak.lock()) {
                rhs_weak.lock()->grad += lhs_weak.lock()->data * out_weak.lock()->grad;
                //}
            });
            return out;
        }

        /**
           * @brief Multiplies a value and a float.
           * @param lhs The left-hand side ValuePtr.
           * @param f The float to multiply.
           * @return A shared pointer to the new Value representing the product.
           */
        static ValuePtr multiply(const ValuePtr& lhs, float f) {
            auto rhs = create(f);
            auto out = create((float)(lhs->data * f), "*");
            out->prev = {lhs, rhs};

            Autograd::global_tape.add_entry(out,[lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)]() {
                //if (auto lhs = lhs_weak.lock()) {
                lhs_weak.lock()->grad += rhs_weak.lock()->data * out_weak.lock()->grad;
                //}
                //if (auto rhs = rhs_weak.lock()) {
                rhs_weak.lock()->grad += lhs_weak.lock()->data * out_weak.lock()->grad;
                //}
            });
            return out;
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Power
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief Raises a value to the power of an exponent.
         * @param base The base ValuePtr.
         * @param exponent The float exponent.
         * @return A shared pointer to the new Value representing the result.
         */
        static ValuePtr pow(const ValuePtr& base, float exponent) {
            float newValue = std::pow(base->data, exponent);
            auto out = create(newValue, "^");
            out->prev = {base};

            Autograd::global_tape.add_entry(out,[base_weak = std::weak_ptr<Value>(base), out_weak = std::weak_ptr<Value>(out), exponent]() {
                if (auto base = base_weak.lock()) {
                    base->grad += exponent * std::pow(base->data, exponent - 1) * out_weak.lock()->grad;
                }
            });

            return out;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Division
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief Divides a value by a float.
         * @param lhs The left-hand side ValuePtr.
         * @param otherValue The float divisor.
         * @return A shared pointer to the new Value representing the quotient.
         */
        static ValuePtr divide ( const std::shared_ptr<Value>& lhs, float otherValue) {
            return multiply(lhs , std::pow(otherValue, -1.0f));
        }

        /**
        * @brief Divides a value by a float.
        * @param lhs The left-hand side ValuePtr.
        * @param rhs The right-hand side ValuePtr.
        * @return A shared pointer to the new Value representing the quotient.
        */
        static ValuePtr divide(const ValuePtr& lhs, const ValuePtr& rhs) {
            auto reciprocal = pow(rhs, -1);
            return multiply(lhs, reciprocal);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Subtraction
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
       * @brief Subtracts two values and creates a new value.
       * @param lhs The left-hand side ValuePtr.
       * @param rhs The right-hand side ValuePtr.
       * @return A shared pointer to the new Value representing the difference.
       */
        static ValuePtr subtract(const ValuePtr& lhs, const ValuePtr& rhs) {
            auto out = create((float)(lhs->data - rhs->data), "-");
            out->prev = {lhs, rhs};

            Autograd::global_tape.add_entry(out,[lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)]() {
                //if (auto lhs = lhs_weak.lock()) {
                lhs_weak.lock()->grad += out_weak.lock()->grad;
                //}
                //if (auto rhs = rhs_weak.lock()) {
                rhs_weak.lock()->grad -= out_weak.lock()->grad;
                //}
            });

            return out;
        }

        /**
       * @brief Subtracts two values and creates a new value.
       * @param lhs The left-hand side ValuePtr.
       * @param f float
       * @return A shared pointer to the new Value representing the difference.
       */
        static ValuePtr subtract(const ValuePtr& lhs, float f) {
            auto rhs = create(f);
            auto out = create((float)(lhs->data - rhs->data), "-");
            out->prev = {lhs, rhs};

            Autograd::global_tape.add_entry(out,[lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)]() {
                //if (auto lhs = lhs_weak.lock()) {
                lhs_weak.lock()->grad += out_weak.lock()->grad;
                //}
                //if (auto rhs = rhs_weak.lock()) {
                rhs_weak.lock()->grad -= out_weak.lock()->grad;
                //}
            });

            return out;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Activation functions
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
       * @brief Applies the tanh activation function.
       * @param v The input ValuePtr.
       * @return A shared pointer to the new Value representing the tanh output.
       */
        static ValuePtr tanh(const ValuePtr& v) {
            float x = v->data;
            float t = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
            auto out = create(t, "tanh");
            out->prev = {v};

            Autograd::global_tape.add_entry(out,[v_weak = std::weak_ptr<Value>(v), t, out_weak = std::weak_ptr<Value>(out)]() {
                v_weak.lock()->grad += (1 - t * t) * out_weak.lock()->grad;
            });

            return out;
        }

        /**
         * @brief Applies the ReLU activation function.
         * @param v The input ValuePtr.
         * @return A shared pointer to the new Value representing the ReLU output.
         */
        static ValuePtr relu(const ValuePtr& v) {
            float val = std::max(0.0f, v->data);
            auto out = create(val, "ReLU");
            out->prev = {v};

            Autograd::global_tape.add_entry(out,[v_weak = std::weak_ptr<Value>(v), out_weak = std::weak_ptr<Value>(out)]() {
                if (v_weak.lock()) {
                    v_weak.lock()->grad += static_cast<float>((out_weak.lock()->data > 0)) * out_weak.lock()->grad;
                };
            });

            return out;
        }

        /**
           * @brief Applies the sigmoid activation function.
           * @param v The input ValuePtr.
           * @return A shared pointer to the new Value representing the sigmoid output.
        */
        static ValuePtr sigmoid(const ValuePtr& v) {
            float x = v->data;
            float t = std::exp(x) / (1 + std::exp(x));
            auto out = create(t, "Sigmoid");
            out->prev = {v};

            Autograd::global_tape.add_entry(out,[v_weak = std::weak_ptr<Value>(v), t, out_weak = std::weak_ptr<Value>(out)]() {
                v_weak.lock()->grad += t * (1 - t) * out_weak.lock()->grad;
                //v_weak.lock()->clip_gradients();
            });

            return out;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Topological sort
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//        void buildTopo(std::shared_ptr<Value> v, std::unordered_set<std::shared_ptr<Value>, Hash>& visited, std::vector<std::shared_ptr<Value>>& topo) {
//            if (visited.find(v) == visited.end()) {
//                visited.insert(v);
//                for (const auto& child : v->prev) {
//                    buildTopo(child, visited, topo);
//                }
//                topo.push_back(v);
//            }
//        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Backpropagation algorithm
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Backpropagation
        void backProp() {
            this->_backward();
        }
        /**
            * @brief Initiates the backward pass by setting the gradient of the output value to 1.
        */
        void _backward() {
            grad = 1.0f;
            Autograd::global_tape.backward();
        }

//        /**
//         * @brief Builds the topological order of the computational graph.
//         * @param v The input ValuePtr.
//         * @param visited A set to track visited nodes.
//         * @param topo The topologically sorted output values.
//         */
//        void buildTopo(const ValuePtr& v, std::unordered_set<ValuePtr>& visited, std::vector<ValuePtr>& topo) {
//            if (visited.find(v) == visited.end()) {
//                visited.insert(v);
//                for (const auto& child : v->prev) {
//                    buildTopo(child, visited, topo);
//                }
//                topo.push_back(v);
//            }
//        }

        bool operator==(const Value& other) const;

        friend std::ostream & operator << (std::ostream &os, const std::shared_ptr<Value> &v){
            os << "[data: " << std::setw(3) << v->data << ", grad: " << std::setw(3) << v->grad << "] ";
            return os;
        }
    };



    size_t Hash::operator()(const std::shared_ptr<Value>& value) const {
        if (!value) {
            return 0;
        }
        return std::hash<const void*>()(value.get());
    }

    bool Value::operator==(const Value& other) const {
        return data == other.data && op == other.op && prev == other.prev;
    }
}


