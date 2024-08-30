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

namespace microgradpp {
    class Value;

    struct Hash {
        size_t operator()(const std::shared_ptr<Value>& value) const;
    };

    class Value : public std::enable_shared_from_this<Value> {
    private:
        // Private constructor
        Value(float data, const std::vector<std::shared_ptr<Value>>& children = {}, const std::string& op = "", const std::string& label = "")
                : data(data), prev(children.begin(), children.end()), op(op), label(label) {}

    public:
        // Factory method for creating Value instances
        static std::shared_ptr<Value> create(float data, const std::vector<std::shared_ptr<Value>>& children = {}, const std::string& op = "", const std::string& label = "") {
            std::string l;
            if (label.empty()) {
                l = std::to_string(labelIdx); // Convert int to string
                ++labelIdx;
            } else {
                l = label;
            }
            return std::shared_ptr<Value>(new Value(data, children, op, l));
        }

        ~Value(){
            --labelIdx;
        }

        float data = 0;
        float grad = 0;
        std::function<void()> backward = nullptr;
        std::unordered_set<std::shared_ptr<Value>, Hash> prev;
        std::string op;
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
        friend std::shared_ptr<Value> operator +(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
            auto out = Value::create((float)(lhs->data + rhs->data), {lhs, rhs}, "+");

            auto backward = [l = std::weak_ptr<Value>(lhs), r = std::weak_ptr<Value>(rhs), o=std::weak_ptr<Value>(out)]() mutable {
                if (auto lhs = l.lock()) {
                    if (auto out = o.lock()) {
                        lhs->grad += (float)out->grad;
                        lhs->clip_gradients();
                    }
                }
                if (auto rhs = r.lock()) {
                    if (auto out = o.lock()) {
                        rhs->grad += (float)out->grad;
                        rhs->clip_gradients();
                    }
                }
            };

            out->backward = backward;
            return out;
        }

        friend std::shared_ptr<Value> operator +(const std::shared_ptr<Value>& lhs, float f) {
            auto other = Value::create((float)f);
            auto out = Value::create((float)(lhs->data + other->data), {lhs, other}, "+");

            auto backward = [l = std::weak_ptr<Value>(lhs), r = std::weak_ptr<Value>(other), o=std::weak_ptr<Value>(out)]() mutable {
                if (auto lhs = l.lock()) {
                    if (auto out = o.lock()) {
                        lhs->grad += (float)out->grad;
                        lhs->clip_gradients();
                    }
                }
                if (auto rhs = r.lock()) {
                    if (auto out = o.lock()) {
                        rhs->grad += (float)out->grad;
                        rhs->clip_gradients();
                    }
                }
            };

            out->backward = backward;
            return out;
        }

        friend std::shared_ptr<Value>& operator +=(std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
            lhs = lhs + rhs;
            return lhs;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Multiplication
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        friend std::shared_ptr<Value> operator *(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
            auto out = Value::create((float)(lhs->data * rhs->data), {lhs, rhs}, "*");

            auto backward = [l = std::weak_ptr<Value>(lhs), r = std::weak_ptr<Value>(rhs), o=std::weak_ptr<Value>(out)]() mutable {
                l.lock()->grad += (float)(r.lock()->data * o.lock()->grad);
                r.lock()->grad += (float)(l.lock()->data * o.lock()->grad);
            };

            out->backward = backward;
            return out;
        }

        friend std::shared_ptr<Value> operator *(const std::shared_ptr<Value>& lhs, float f) {
            auto other = Value::create((float)f);
            auto out = Value::create((float)(lhs->data * other->data), {lhs, other}, "*");

            auto backward = [lhs = std::weak_ptr<Value>(lhs), other=std::weak_ptr<Value>(other), out=std::weak_ptr<Value>(out)]() mutable {
                lhs.lock()->grad += (float)(other.lock()->data * out.lock()->grad);
                other.lock()->grad += (float)(lhs.lock()->data * out.lock()->grad);
            };
            out->backward = backward;
            return out;
        }

        friend std::shared_ptr<Value>& operator *=(std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
            lhs = lhs * rhs;
            return lhs;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Power
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        friend std::shared_ptr<Value> operator ^(const std::shared_ptr<Value>& lhs, float otherValue) {
            auto newValue = (float)std::pow(lhs->data, otherValue);
            auto out = Value::create(newValue, {lhs}, "^");

            auto backward = [lhs=std::weak_ptr<Value>(lhs), out=std::weak_ptr<Value>(out), otherValue]() mutable {
                lhs.lock()->grad += (float)(otherValue * (float)std::pow(lhs.lock()->data, otherValue - 1) * out.lock()->grad);
                //lhs->clip_gradients();
            };
            out->backward = backward;
            return out;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Division
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        friend std::shared_ptr<Value> operator /( const std::shared_ptr<Value>& lhs, float otherValue) {
            return lhs * std::pow(otherValue, -1);
        }

        friend std::shared_ptr<Value> operator /( const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
            return lhs *  (rhs ^ -1);
            //return lhs;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Subtraction
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        friend std::shared_ptr<Value> operator -(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
            auto out = Value::create((float)(lhs->data - rhs->data), {lhs, rhs}, "-");

            auto backward = [lhs=std::weak_ptr<Value>(lhs), rhs=std::weak_ptr<Value>(rhs), out=std::weak_ptr<Value>(out)]() mutable {
                lhs.lock()->grad += (float)out.lock()->grad;
                rhs.lock()->grad -= (float)out.lock()->grad;
            };
            out->backward = backward;
            return out;
        }

        friend std::shared_ptr<Value> operator -(const std::shared_ptr<Value>& lhs, float f) {
            auto other = Value::create((float)f);
            auto out = Value::create((float)(lhs->data - other->data), {lhs, other}, "-");

            auto backward = [lhs=std::weak_ptr<Value>(lhs), other=std::weak_ptr<Value>(other), out=std::weak_ptr<Value>(out)]() mutable {
                lhs.lock()->grad += (float)out.lock()->grad;
                other.lock()->grad -= (float)out.lock()->grad;
            };
            out->backward = backward;
            return out;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Activation functions
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        std::shared_ptr<Value> tanh() {
            float x = this->data;
            float t = (float )(std::exp(2 * x) - 1) / (float )(std::exp(2 * x) + 1);
            auto out = Value::create(t, {shared_from_this()}, "tanh");

            auto backward = [this, t, out=std::weak_ptr<Value>(out)]() mutable {
                this->grad += (float )(1 - t * t) * out.lock()->grad;
            };
            out->backward = backward;
            return out;
        }

        std::shared_ptr<Value> relu() {
            float val = this->data < 0 ? 0 : this->data;
            auto out = Value::create(val, {shared_from_this()}, "ReLU");

            auto backward = [this, out = out]() mutable {
                //std::cout << "Before relu : lhs.grad=" << this->grad << std::endl;
                this->grad += (float)((out->data > 0) * out->grad);
                //std::cout << "After relu : lhs.grad=" << this->grad << std::endl;
            };
            out->backward = backward;
            return out;
        }

        std::shared_ptr<Value> sigmoid() {
            float x = this->data;
            float t = (float )(std::exp(x)) / (float )(1 + std::exp(x));
            auto out = Value::create(t, {shared_from_this()}, "Sigmoid");

            auto backward = [this, out = out, t= t]() mutable {
                this->grad += (float)( float(t* (1 - t)) * out->grad);
                this->clip_gradients();
            };
            out->backward = backward;
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
        void backProp() {
            std::vector<std::shared_ptr<Value>> topo;
            std::unordered_set<std::shared_ptr<Value>, Hash> visited;
            auto shared = shared_from_this();
            buildTopo(shared, visited, topo);
            shared->grad = (float)1.0;
            for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                if ((*it)->backward) {
                    (*it)->backward();
                }
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


