////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Created by Gautam Sharma on 6/28/24.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <vector>
#include <string>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <math.h>
#include <memory>

namespace microgradpp{
    class Value;

    struct Hash {
        size_t operator()(const std::shared_ptr<Value> value) const;
    };

    class Value : public std::enable_shared_from_this<Value>{
    private:
        Value(float data, const std::vector<std::shared_ptr<Value>>& children={}, const std::string& op= "", const std::string& label="")
                : data(data), prev(children.begin(), children.end()), op(op), label(label) {}
    public:
        static std::shared_ptr<Value> create(float data, const std::vector<std::shared_ptr<Value>>& children = {}, const std::string& op = "", const std::string& label = "") {
            return std::shared_ptr<Value>(new Value(data, children, op, label));
        }

        float data;
        float grad = 0;
        std::function<void()> backward = nullptr;
        std::unordered_set<std::shared_ptr<Value>, Hash> prev;
        std::string op;
        std::string label;

        Value(const Value&v) = default;
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Addition
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Add  lhs + rhs
        friend std::shared_ptr<Value> operator +(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
            auto out = Value::create(lhs->data + rhs->data, {lhs, rhs}, "+");

            auto backward = [lhs, rhs, out]() mutable {
                lhs->grad += out->grad;
                rhs->grad += out->grad;
            };

            out->backward = backward;
            return out;
        }

        // Add lhs + constant
        friend std::shared_ptr<Value> operator +(const std::shared_ptr<Value>& lhs, float f) {
            auto other = Value::create(f);
            auto out = Value::create(lhs->data + other->data, {lhs, other}, "+");

            auto backward = [lhs, other, out]() mutable {
                lhs->grad += out->grad;
                other->grad += out->grad;
            };
            out->backward = backward;
            return out;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Multiplication
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Multiply lhs + rhs
        friend std::shared_ptr<Value> operator *(const std::shared_ptr<Value>& lhs,  const std::shared_ptr<Value>& rhs){
            auto out = Value::create(lhs->data * rhs->data, {lhs, rhs}, "*");

            auto backward = [lhs, rhs, out]() mutable {
                lhs->grad += rhs->data * out->grad;
                rhs->grad += lhs->data * out->grad;
            };

            out->backward = backward;
            return out;
        }

        // Multiply lhs + constant
        friend std::shared_ptr<Value> operator *(const std::shared_ptr<Value>& lhs,  float f){
            auto other = Value::create(f);
            auto out = Value::create(lhs->data * other->data, {lhs, other}, "*");

            auto backward = [lhs, other, out]() mutable {
                lhs->grad += other->data * out->grad;
                other->grad += lhs->data * out->grad;
            };

            out->backward = backward;
            return out;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // power
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // exp lhs ^ constant
        friend std::shared_ptr<Value> operator ^(const std::shared_ptr<Value>& lhs,float otherValue){
            auto newValue = std::pow(lhs->data, otherValue);
            auto out = Value::create(newValue, {lhs}, "*");

            auto backward = [lhs, out, otherValue]() mutable {
                lhs->grad += otherValue * std::pow(lhs->data, otherValue - 1) * out->grad;
            };

            out->backward = backward;
            return out;
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Divide
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        friend std::shared_ptr<Value> operator /(const std::shared_ptr<Value>& lhs,float otherValue){
            return lhs * std::pow(otherValue, -1);
        }

        friend std::shared_ptr<Value> operator /(const std::shared_ptr<Value>& lhs,const std::shared_ptr<Value>& rhs){
            return lhs * (rhs ^ -1);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Subtract
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Subtract Value
        friend std::shared_ptr<Value> operator -(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs){
            auto out = Value::create(lhs->data - rhs->data, {lhs, rhs}, "*");

            auto backward = [lhs, rhs, out]() mutable{
                lhs->grad +=  out->grad;
                rhs->grad -=  out->grad;
            };

            out->backward = backward;
            return out;
        }

        // Subtract constant
        friend std::shared_ptr<Value> operator -(const std::shared_ptr<Value>& lhs, float f){
            auto other = Value::create(f);
            auto out = Value::create(lhs->data - other->data, {lhs, other}, "*");

            auto backward = [lhs, other, out]() mutable {
                lhs->grad +=  out->grad;
                other->grad -=  out->grad;
            };

            out->backward = backward;
            return out;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Activation functions
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        std::shared_ptr<Value> tanh(){
            auto x = this->data;
            double t = (double)(std::exp(2*x) - 1.0f) / (double)(std::exp(2*x) + 1.0f);
            auto out = Value::create(t, {shared_from_this()}, "tanh");
            auto backward = [this,t,&out]() mutable{
                this->grad +=  (double)(1 - t*t) * (double)out->grad;
            };
            out->backward = backward;
            return out;
        }

        std::shared_ptr<Value> relu(){
            auto val = this->data < 0 ? 0 :  this->data;
            auto out = Value::create(val, {shared_from_this()}, "ReLU");
            auto backward = [this, &out](){
                this->grad +=  (out->data > 0) * out->grad;
            };
            out->backward = backward;
            return out;
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Topological sort
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // buildTopo
        void buildTopo(std::shared_ptr<Value> v, std::unordered_set<std::shared_ptr<Value>, Hash>& visited, std::vector<std::shared_ptr<Value>>& topo){
            if(visited.find(v) == visited.end()){
                visited.insert(v);
                for(const auto& child : v->prev){
                    buildTopo(child, visited, topo);
                }
                topo.push_back(v);
            }
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Backprop algo
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // backProp
        void backProp(){
            std::vector<std::shared_ptr<Value>> topo;
            std::unordered_set<std::shared_ptr<Value>, Hash> visited;
            buildTopo(shared_from_this(), visited, topo);
            this->grad = 1;
            for( auto it = topo.rbegin(); it != topo.rend(); ++it ){
                if((*it)->backward)  {
                    (*it)->backward();
                }
                printf("data : %f, grad :  %f %s\n", (*it)->data, (*it)->grad, (*it)->label.c_str());

            }
        }

        bool operator==(const Value& other) const;
    };


    size_t Hash::operator()(const std::shared_ptr<Value> value) const {
        return std::hash<std::string>()(value.get()->op) ^ std::hash<float>()(value.get()->data);
    }

    bool Value::operator==(const Value& other) const {
        return data == other.data && op == other.op && prev == other.prev;
    }
}
