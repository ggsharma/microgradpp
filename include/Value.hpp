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

namespace microgradpp {
    class Value;

    struct Hash {
        size_t operator()(const std::shared_ptr<Value> value) const;
    };

    class Value : public std::enable_shared_from_this<Value> {
    public:
        Value(double data, const std::vector<std::shared_ptr<Value>>& children = {}, const std::string& op = "", const std::string& label = "")
                : data(data), prev(children.begin(), children.end()), op(op), label(label) {}

        static std::shared_ptr<Value> create(double data, const std::vector<std::shared_ptr<Value>>& children = {}, const std::string& op = "", const std::string& label = "") {
            std::string l;
            if (label.empty()) {
                l = std::to_string(labelIdx); // Convert int to string
                ++labelIdx;
            } else {
                l = label;
            }
            return std::make_shared<Value>(data, children, op, l);
        }

        double data = 0;
        double grad = 0;
        std::function<void()> backward = nullptr;
        std::unordered_set<std::shared_ptr<Value>, Hash> prev;
        std::string op;
        std::string label;
        inline static int labelIdx = 0;

        Value(const Value& v) = default;

        const double GRADIENT_CLIP_VALUE = 1e4; // Gradient clipping value
        const double EPSILON = 1e-7; // Small value for numerical stability

        void clip_gradients() {
//            if (grad > GRADIENT_CLIP_VALUE) grad = GRADIENT_CLIP_VALUE;
//            if (grad < -GRADIENT_CLIP_VALUE) grad = -GRADIENT_CLIP_VALUE;
        }

        void resetGradients() {
            grad = 0;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Addition
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        friend std::shared_ptr<Value> operator +(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
            auto out = Value::create((double)(lhs->data + rhs->data), {lhs, rhs}, "+");

            auto backward = [lhs = lhs, rhs = rhs, out = out]() mutable {
                //std::cout << "Before + : lhs.grad=" << lhs->grad << ", rhs.grad=" << rhs->grad << std::endl;
                lhs->grad += (double)out->grad;
                rhs->grad += (double)out->grad;
                lhs->clip_gradients();
                rhs->clip_gradients();
                //std::cout << "After + : lhs.grad=" << lhs->grad << ", rhs.grad=" << rhs->grad << std::endl;
            };

            out->backward = backward;
            return out;
        }

        friend std::shared_ptr<Value> operator +(const std::shared_ptr<Value>& lhs, double f) {
            auto other = Value::create((double)f);
            auto out = Value::create((double)(lhs->data + other->data), {lhs, other}, "+");

            auto backward = [lhs = lhs, other = other, out = out]() mutable {
                //std::cout << "Before + : lhs.grad=" << lhs->grad << ", rhs.grad=" << other->grad << std::endl;
                lhs->grad += (double)(out->grad);
                other->grad += (double)(out->grad);
                lhs->clip_gradients();
                other->clip_gradients();
                //std::cout << "After + : lhs.grad=" << lhs->grad << ", rhs.grad=" << other->grad << std::endl;
            };
            out->backward = backward;
            return out;
        }

        friend std::shared_ptr<Value> operator +=(std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
            lhs = lhs + rhs;
            return lhs;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Multiplication
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        friend std::shared_ptr<Value> operator *(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
            auto out = Value::create((double)(lhs->data * rhs->data), {lhs, rhs}, "*");

            auto backward = [lhs, rhs, out]() mutable {
                //std::cout << "Before * : lhs.grad=" << lhs->grad << ", rhs.grad=" << rhs->grad << std::endl;
                lhs->grad += (double)(rhs->data * out->grad);
                rhs->grad += (double)(lhs->data * out->grad);
                lhs->clip_gradients();
                rhs->clip_gradients();
                //std::cout << "After * : lhs.grad=" << lhs->grad << ", rhs.grad=" << rhs->grad << std::endl;
            };

            out->backward = backward;
            return out;
        }

        friend std::shared_ptr<Value> operator *(const std::shared_ptr<Value>& lhs, double f) {
            auto other = Value::create((double)f);
            auto out = Value::create((double)(lhs->data * other->data), {lhs, other}, "*");

            auto backward = [lhs, other, out]() mutable {
                //std::cout << "Before * : lhs.grad=" << lhs->grad << ", rhs.grad=" << other->grad << std::endl;
                lhs->grad += (double)(other->data * out->grad);
                other->grad += (double)(lhs->data * out->grad);
                lhs->clip_gradients();
                other->clip_gradients();
                //std::cout << "Before * : lhs.grad=" << lhs->grad << ", rhs.grad=" << other->grad << std::endl;
            };

            out->backward = backward;
            return out;
        }

        friend std::shared_ptr<Value> operator *=(std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
            lhs = lhs * rhs;
            return lhs;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Power
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        friend std::shared_ptr<Value> operator ^(const std::shared_ptr<Value>& lhs, double otherValue) {
            auto newValue = (double)std::pow(lhs->data, otherValue);
            auto out = Value::create(newValue, {lhs}, "^");

            auto backward = [lhs, out, otherValue]() mutable {
                lhs->grad += (double)(otherValue * (double)std::pow(lhs->data, otherValue - 1) * out->grad);
                lhs->clip_gradients();
            };

            out->backward = backward;
            return out;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Division
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        friend std::shared_ptr<Value> operator /(const std::shared_ptr<Value>& lhs, double otherValue) {
            return lhs * std::pow(otherValue, -1);
        }

        friend std::shared_ptr<Value> operator /(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
            return lhs * (rhs ^ -1);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Subtraction
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        friend std::shared_ptr<Value> operator -(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
            auto out = Value::create((double)(lhs->data - rhs->data), {lhs, rhs}, "-");

            auto backward = [lhs, rhs, out]() mutable {
                //std::cout << "Before - : lhs.grad=" << lhs->grad << ", rhs.grad=" << rhs->grad << std::endl;
                lhs->grad += (double)out->grad;
                rhs->grad -= (double)out->grad;
                lhs->clip_gradients();
                rhs->clip_gradients();
                //std::cout << "After - : lhs.grad=" << lhs->grad << ", rhs.grad=" << rhs->grad << std::endl;
            };

            out->backward = backward;
            return out;
        }

        friend std::shared_ptr<Value> operator -(const std::shared_ptr<Value>& lhs, double f) {
            auto other = Value::create((double)f);
            auto out = Value::create((double)(lhs->data - other->data), {lhs, other}, "-");

            auto backward = [lhs, other, out]() mutable {
                //std::cout << "Before - : lhs.grad=" << lhs->grad << ", rhs.grad=" << other->grad << std::endl;
                lhs->grad += (double)out->grad;
                other->grad -= (double)out->grad;
                lhs->clip_gradients();
                other->clip_gradients();
                //std::cout << "After - : lhs.grad=" << lhs->grad << ", rhs.grad=" << other->grad << std::endl;
            };

            out->backward = backward;
            return out;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Activation functions
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        std::shared_ptr<Value> tanh() {
            double x = this->data;
            double t = (double )(std::exp(2 * x) - 1) / (double )(std::exp(2 * x) + 1);
            auto out = Value::create(t, {shared_from_this()}, "tanh");

            auto backward = [this, t, out = out]() mutable {
                this->grad += (double )(1 - t * t) * out->grad;
                this->clip_gradients();
            };

            out->backward = backward;
            return out;
        }

        std::shared_ptr<Value> relu() {
            double val = this->data < 0 ? 0 : this->data;
            auto out = Value::create(val, {shared_from_this()}, "ReLU");

            auto backward = [this, out = out]() mutable {
                //std::cout << "Before relu : lhs.grad=" << this->grad << std::endl;
                this->grad += (double)((out->data > 0) * out->grad);
                this->clip_gradients();
                //std::cout << "After relu : lhs.grad=" << this->grad << std::endl;
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
            shared->grad = (double)1.0;
            std::cout<< "Num Parameters: " << topo.size() << std::endl;
            for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                if ((*it)->backward) {
                    (*it)->backward();
                    //(*it)->clip_gradients(); // Clip gradients here
                }
            }
        }

        bool operator==(const Value& other) const;

        friend std::ostream & operator << (std::ostream &os, const std::shared_ptr<Value> &v){
            os << "[data: " << v->data << ", grad: "<<  v->grad << "]" << std::endl;
            return os;
        }
    };

    size_t Hash::operator()(const std::shared_ptr<Value> value) const {
        return std::hash<std::string>()(value.get()->op) ^ std::hash<double>()(value.get()->data);
    }

    bool Value::operator==(const Value& other) const {
        return data == other.data && op == other.op && prev == other.prev;
    }
}
