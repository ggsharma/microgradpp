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

    Autograd Autograd::global_tape;

    struct Hash {
        size_t operator()(const std::shared_ptr<Value>& value) const;
    };


    using ValuePtr = std::shared_ptr<Value>;
    class Value : public std::enable_shared_from_this<Value> {

    private:
        Value(float data, const std::string& op = "", size_t id = 0)
                : data(data), grad(0.0), op(op), id(id) {}
    public:
        inline static size_t currentID = 0;
        float data;
        float grad;
        std::string op;
        size_t id = 0LU;
        std::vector<ValuePtr> prev;
        std::function<void()> backward;

        static size_t generateID() {
            return currentID++;
        }

        /**
         * @param data : float
         * @param op : string
         * @param id : string
         * @brief Something brief about this
         */
        static ValuePtr create(float data, const std::string& op = "", size_t id = 0){
            if(id == 0){
                id = generateID();
            }
            return std::shared_ptr<Value>(new Value(data,  op, id));
        }

        ~Value(){
            --currentID;
//            if(currentID == 0){
//                std::cout << "Clearing out\n";
//                Autograd::global_tape.clear();
//            }
        }

        std::string label;
        inline static int labelIdx = 0;

        Value(const Value& v) = default;

        const float GRADIENT_CLIP_VALUE = 1e4; // Gradient clipping value
        const float EPSILON = 1e-7; // Small value for numerical stability

        void clip_gradients() {
//            if (grad > GRADIENT_CLIP_VALUE) grad = GRADIENT_CLIP_VALUE;
//            if (grad < -GRADIENT_CLIP_VALUE) grad = -GRADIENT_CLIP_VALUE;
        }

        void reset(){
            this->grad = 0.0;
            this->data = 0.0;
            this->prev.clear();
        }

        void resetGradients() {
            grad = 0;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Addition
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Static methods to create operations
        static ValuePtr add(const ValuePtr& lhs, const ValuePtr& rhs) {
            auto out = create(lhs->data + rhs->data, "+", generateID());
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


        static ValuePtr add(const ValuePtr& lhs, float f) {
            auto rhs = Value::create((float)f);
            auto out = create(lhs->data + f, "+", generateID());
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
        static ValuePtr multiply(const ValuePtr& lhs, const ValuePtr& rhs) {
            auto out = create(lhs->data * rhs->data, "*", generateID());
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


        static ValuePtr multiply(const ValuePtr& lhs, float f) {
            auto rhs = create(f);
            auto out = create(lhs->data * f, "*", generateID());
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
        static ValuePtr pow(const ValuePtr& base, float exponent) {
            float newValue = std::pow(base->data, exponent);
            auto out = create(newValue, "^", generateID());
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
        static ValuePtr divide ( const std::shared_ptr<Value>& lhs, float otherValue) {
            return multiply(lhs , std::pow(otherValue, -1));
        }

        static ValuePtr divide(const ValuePtr& lhs, const ValuePtr& rhs) {
            auto reciprocal = pow(rhs, -1);
            return multiply(lhs, reciprocal);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Subtraction
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        static ValuePtr subtract(const ValuePtr& lhs, const ValuePtr& rhs) {
            auto out = create(lhs->data - rhs->data, "-", generateID());
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


        static ValuePtr subtract(const ValuePtr& lhs, float f) {
            auto rhs = create(f);
            auto out = create(lhs->data - rhs->data, "-", generateID());
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
        static ValuePtr tanh(const ValuePtr& v) {
            float x = v->data;
            float t = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
            auto out = create(t, "tanh", generateID());
            out->prev = {v};
            Autograd::global_tape.add_entry(out,[v_weak = std::weak_ptr<Value>(v), t, out_weak = std::weak_ptr<Value>(out)]() {
                v_weak.lock()->grad += (1 - t * t) * out_weak.lock()->grad;
            });
            return out;
        }

        static ValuePtr relu(const ValuePtr& v) {
            float val = std::max(0.0f, v->data);
            auto out = create(val, "ReLU", generateID());
            out->prev = {v};
            Autograd::global_tape.add_entry(out,[v, out]() {
                if (v) v->grad += (out->data > 0) * out->grad;
            });
            return out;
        }

        static ValuePtr sigmoid(const ValuePtr& v) {
            float x = v->data;
            float t = std::exp(x) / (1 + std::exp(x));
            auto out = create(t, "Sigmoid", generateID());
            out->prev = {v};
            Autograd::global_tape.add_entry(out,[v_weak = std::weak_ptr<Value>(v), t, out_weak = std::weak_ptr<Value>(out)]() {
                v_weak.lock()->grad += t * (1 - t) * out_weak.lock()->grad;
            });
            return out;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Topological sort
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void buildTopo(std::shared_ptr<Value> v, std::unordered_set<std::shared_ptr<Value>, Hash>& visited, std::vector<std::shared_ptr<Value>>& topo) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (const auto& child : v->prev) {
                    buildTopo(child, visited, topo);
                }
                topo.push_back(v);
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Backpropagation algorithm
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Backpropagation
        void backProp() {
            this->_backward();
        }

        // A backward pass function that assigns a gradient of 1 to the output value
        void _backward() {
            grad = 1.0f;
            Autograd::global_tape.backward();            // Start the backward pass on the tape

        }

        void buildTopo(const ValuePtr& v, std::unordered_map<size_t, bool>& visited, std::vector<ValuePtr>& topo) {
            if (visited.find(v->id) == visited.end()) {
                visited[v->id] = true;
                for (const auto& child : v->prev) {
                    buildTopo(child, visited, topo);
                }
                topo.push_back(v);
            }
        }

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


