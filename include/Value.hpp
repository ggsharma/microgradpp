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
        Value(double data, const std::vector<std::shared_ptr<Value>>& children = {}, const std::string& op = "", const std::string& label = "")
                : data(data), prev(children.begin(), children.end()), op(op), label(label) {}

    public:
        // Factory method for creating Value instances
        static std::shared_ptr<Value> create(double data, const std::vector<std::shared_ptr<Value>>& children = {}, const std::string& op = "", const std::string& label = "") {
            std::string l;
            if (label.empty()) {
                l = std::to_string(labelIdx); // Convert int to string
                ++labelIdx;
                //std::cout << labelIdx << std::endl;
            } else {
                l = label;
                //std::cout << labelIdx << std::endl;
            }
            return std::shared_ptr<Value>(new Value(data, children, op, l));
        }




        ~Value(){
            --labelIdx;
            //std::cout << "Destructor being called!\n";
        }

        double data = 0;
        double grad = 0;
        std::function<void()> backward = nullptr;
        std::unordered_set<std::shared_ptr<Value>, Hash> prev;
        //inline static std::unordered_set<std::shared_ptr<Value>, Hash> memory = {};
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
            auto out = Value::create((double)(lhs->data + rhs->data), {lhs, rhs}, "+");
//            auto l = std::weak_ptr<Value>(lhs);
//            auto r = std::weak_ptr<Value>(rhs);
//            auto o = std::weak_ptr<Value>(out);

//            auto backward = [lhs = std::weak_ptr<Value>(lhs), rhs = std::weak_ptr<Value>(rhs), out = std::weak_ptr<Value>(
//                    out)]() mutable {
//                lhs.lock()->grad += out.lock()->grad;
//                rhs.lock()->grad += out.lock()->grad;
//            };


            //return out;

            auto backward = [l = std::weak_ptr<Value>(lhs), r = std::weak_ptr<Value>(rhs), o=std::weak_ptr<Value>(out)]() mutable {
                if (auto lhs = l.lock()) {
                    if (auto out = o.lock()) {
                        lhs->grad += (double)out->grad;
                        lhs->clip_gradients();
                    }
                }
                if (auto rhs = r.lock()) {
                    if (auto out = o.lock()) {
                        rhs->grad += (double)out->grad;
                        rhs->clip_gradients();
                    }
                }
            };

            out->backward = backward;
//            auto backward = [lhs = lhs, rhs = rhs, out = out]() mutable {
//                //std::cout << "Before + : lhs.grad=" << lhs->grad << ", rhs.grad=" << rhs->grad << std::endl;
//                lhs->grad += (double)out->grad;
//                rhs->grad += (double)out->grad;
//                lhs->clip_gradients();
//                rhs->clip_gradients();
//                //std::cout << "After + : lhs.grad=" << lhs->grad << ", rhs.grad=" << rhs->grad << std::endl;
//            };

            return out;
        }

        friend std::shared_ptr<Value> operator +(const std::shared_ptr<Value>& lhs, double f) {
            auto other = Value::create((double)f);
            auto out = Value::create((double)(lhs->data + other->data), {lhs, other}, "+");

//            auto backward = [lhs = lhs, other = other, out = out]() mutable {
//                //std::cout << "Before + : lhs.grad=" << lhs->grad << ", rhs.grad=" << other->grad << std::endl;
//                lhs->grad += (double)(out->grad);
//                other->grad += (double)(out->grad);
//                lhs->clip_gradients();
//                other->clip_gradients();
//                //std::cout << "After + : lhs.grad=" << lhs->grad << ", rhs.grad=" << other->grad << std::endl;
//            };


            auto backward = [l = std::weak_ptr<Value>(lhs), r = std::weak_ptr<Value>(other), o=std::weak_ptr<Value>(out)]() mutable {
                if (auto lhs = l.lock()) {
                    if (auto out = o.lock()) {
                        lhs->grad += (double)out->grad;
                        lhs->clip_gradients();
                    }
                }
                if (auto rhs = r.lock()) {
                    if (auto out = o.lock()) {
                        rhs->grad += (double)out->grad;
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
            auto out = Value::create((double)(lhs->data * rhs->data), {lhs, rhs}, "*");

            auto backward = [l = std::weak_ptr<Value>(lhs), r = std::weak_ptr<Value>(rhs), o=std::weak_ptr<Value>(out)]() mutable {
                l.lock()->grad += (double)(r.lock()->data * o.lock()->grad);
                r.lock()->grad += (double)(l.lock()->data * o.lock()->grad);
            };
//            auto backward = [lhs, rhs, out]() mutable {
//                //std::cout << "Before * : lhs.grad=" << lhs->grad << ", rhs.grad=" << rhs->grad << std::endl;
//                lhs->grad += (double)(rhs->data * out->grad);
//                rhs->grad += (double)(lhs->data * out->grad);
//                lhs->clip_gradients();
//                rhs->clip_gradients();
//                //std::cout << "After * : lhs.grad=" << lhs->grad << ", rhs.grad=" << rhs->grad << std::endl;
//            };

            out->backward = backward;
            return out;
        }

        friend std::shared_ptr<Value> operator *(const std::shared_ptr<Value>& lhs, double f) {
            auto other = Value::create((double)f);
            auto out = Value::create((double)(lhs->data * other->data), {lhs, other}, "*");

            auto backward = [lhs = std::weak_ptr<Value>(lhs), other=std::weak_ptr<Value>(other), out=std::weak_ptr<Value>(out)]() mutable {
                //std::cout << "Before * : lhs.grad=" << lhs->grad << ", rhs.grad=" << other->grad << std::endl;
                lhs.lock()->grad += (double)(other.lock()->data * out.lock()->grad);
                other.lock()->grad += (double)(lhs.lock()->data * out.lock()->grad);
//                lhs->clip_gradients();
//                other->clip_gradients();
                //std::cout << "Before * : lhs.grad=" << lhs->grad << ", rhs.grad=" << other->grad << std::endl;
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
        friend std::shared_ptr<Value> operator ^(const std::shared_ptr<Value>& lhs, double otherValue) {
            auto newValue = (double)std::pow(lhs->data, otherValue);
            auto out = Value::create(newValue, {lhs}, "^");

            auto backward = [lhs=std::weak_ptr<Value>(lhs), out=std::weak_ptr<Value>(out), otherValue]() mutable {
                lhs.lock()->grad += (double)(otherValue * (double)std::pow(lhs.lock()->data, otherValue - 1) * out.lock()->grad);
                //lhs->clip_gradients();
            };
            out->backward = backward;
            return out;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Division
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        friend std::shared_ptr<Value> operator /( const std::shared_ptr<Value>& lhs, double otherValue) {
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
            auto out = Value::create((double)(lhs->data - rhs->data), {lhs, rhs}, "-");

            auto backward = [lhs=std::weak_ptr<Value>(lhs), rhs=std::weak_ptr<Value>(rhs), out=std::weak_ptr<Value>(out)]() mutable {
                //std::cout << "Before - : lhs.grad=" << lhs->grad << ", rhs.grad=" << rhs->grad << std::endl;
                lhs.lock()->grad += (double)out.lock()->grad;
                rhs.lock()->grad -= (double)out.lock()->grad;
//                lhs->clip_gradients();
//                rhs->clip_gradients();
                //std::cout << "After - : lhs.grad=" << lhs->grad << ", rhs.grad=" << rhs->grad << std::endl;
            };
            out->backward = backward;
            return out;
        }

        friend std::shared_ptr<Value> operator -(const std::shared_ptr<Value>& lhs, double f) {
            auto other = Value::create((double)f);
            auto out = Value::create((double)(lhs->data - other->data), {lhs, other}, "-");

            auto backward = [lhs=std::weak_ptr<Value>(lhs), other=std::weak_ptr<Value>(other), out=std::weak_ptr<Value>(out)]() mutable {
                //std::cout << "Before - : lhs.grad=" << lhs->grad << ", rhs.grad=" << other->grad << std::endl;
                lhs.lock()->grad += (double)out.lock()->grad;
                other.lock()->grad -= (double)out.lock()->grad;
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

            auto backward = [this, t, out=std::weak_ptr<Value>(out)]() mutable {
                this->grad += (double )(1 - t * t) * out.lock()->grad;
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
                //std::cout << "After relu : lhs.grad=" << this->grad << std::endl;
            };
            out->backward = backward;
            return out;
        }

        std::shared_ptr<Value> sigmoid() {
            double x = this->data;
            double t = (double )(std::exp(x)) / (double )(1 + std::exp(x));
            auto out = Value::create(t, {shared_from_this()}, "Sigmoid");

            auto backward = [this, out = out, t= t]() mutable {
                //std::cout << "Before relu : lhs.grad=" << this->grad << std::endl;
                this->grad += (double)( double(t* (1 - t)) * out->grad);
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
            //std::cout<< "Num Parameters: " << topo.size() << std::endl;
            for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                if ((*it)->backward) {
                    //std::cout <<(*it).get() << ": " <<  (*it).use_count() << std::endl;
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

//    size_t Hash::operator()(const std::weak_ptr<Value>& value) const {
//
//        return std::hash<const void*>()(value.lock().get());
//    }

    bool Value::operator==(const Value& other) const {
        return data == other.data && op == other.op && prev == other.prev;
    }
}


