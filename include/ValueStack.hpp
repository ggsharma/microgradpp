////
//// Created by Gautam Sharma on 7/2/24.
////
//
//#ifndef MICROGRADPP_VALUESTACK_HPP
//#define MICROGRADPP_VALUESTACK_HPP
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Created by Gautam Sharma on 6/28/24.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//#pragma  once
//#include <cstdio>
//#include <vector>
//#include <string>
//#include <unordered_set>
//#include <set>
//#include <algorithm>
//#include <math.h>
//#include <memory>
//#include <functional> // std::function
//
//namespace microgradpp{
//    class Value;
//
//    struct Hash {
//        size_t operator()(const Value value) const;
//    };
//
//    class Value {
//
//
//    public:
//        Value(float data, const std::vector<Value>& children={}, const std::string& op= "", const std::string& label="")
//                : data(data), prev(children.begin(), children.end()), op(op), label(label) {}
//
//        float data;
//        float grad = 0;
//        std::function<void()> backward = nullptr;
//        std::unordered_set<Value, Hash> prev;
//        std::string op;
//        std::string label;
//        inline static int labelIdx = 0;
//
//        Value(const Value&v) = default;
//        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        // Addition
//        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        // Add  lhs + rhs
//        friend Value operator +(Value& lhs, Value& rhs) {
//            auto out = Value(lhs.data + rhs.data, {lhs, rhs}, "+");
//
//            auto backward = [lhs= lhs, rhs=rhs, out=out]() mutable {
//                lhs.grad = std::min(lhs.grad  + out.grad, std::numeric_limits<float>::max());
//                rhs.grad = std::min(rhs.grad  + out.grad, std::numeric_limits<float>::max());
//            };
//
//            out.backward = backward;
//            return out;
//        }
//
//        // Add lhs + constant
//        friend Value operator +(Value& lhs, float f) {
//            auto other = Value(f);
//            auto out = Value(lhs.data + other.data, {lhs, other}, "+");
//
//            auto backward = [lhs = lhs, other = other, out = out]() mutable {
//                lhs.grad = std::min(lhs.grad  + out.grad, std::numeric_limits<float>::max());
//                other.grad = std::min(other.grad  + out.grad, std::numeric_limits<float>::max());
//            };
//            out.backward = backward;
//            return out;
//        }
//
//        // Add lhs +=
//        friend Value operator +=(Value& lhs, Value& rhs) {
//            lhs =  lhs + rhs;
//            return lhs;
//        }
//
//        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        // Multiplication
//        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        // Multiply lhs + rhs
//        friend Value operator *(Value & lhs,  Value& rhs){
//            auto out = Value(lhs.data * rhs.data, {lhs, rhs}, "*");
//
//            auto backward = [lhs, rhs, out]() mutable {
//                lhs.grad = std::min(lhs.grad + (rhs.data * out.grad), std::numeric_limits<float>::max());
//                rhs.grad = std::min(rhs.grad + (lhs.data * out.grad), std::numeric_limits<float>::max());
//            };
//
//            out.backward = backward;
//            return out;
//        }
//
//        // Multiply lhs + constant
//        friend Value operator *(Value& lhs,  float f){
//            auto other = Value(f);
//            auto out = Value(lhs.data * other.data, {lhs, other}, "*");
//
//            auto backward = [lhs, other, out]() mutable {
//                lhs.grad = std::min(lhs.grad + (other.data * out.grad), std::numeric_limits<float>::max());
//                other.grad = std::min(other.grad + (lhs.data * out.grad), std::numeric_limits<float>::max());
//            };
//
//            out.backward = backward;
//            return out;
//        }
//
//        // Add lhs *=
//        friend Value operator *=(Value& lhs, Value& rhs) {
//            lhs =  lhs * rhs;
//            return lhs;
//        }
//
//        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        // power
//        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        // exp lhs ^ constant
////        friend std::shared_ptr<Value> operator ^(const std::shared_ptr<Value>& lhs,float otherValue){
////            auto newValue = std::pow(lhs->data, otherValue);
////            auto out = Value::create(newValue, {lhs}, "*");
////
////            auto backward = [lhs, out, otherValue]() mutable {
////                lhs->grad += otherValue * std::pow(lhs->data, otherValue - 1) * out->grad;
////            };
////
////            out->backward = backward;
////            return out;
////        }
////        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////        // Divide
////        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////        friend std::shared_ptr<Value> operator /(const std::shared_ptr<Value>& lhs,float otherValue){
////            return lhs * std::pow(otherValue, -1);
////        }
////
////        friend std::shared_ptr<Value> operator /(const std::shared_ptr<Value>& lhs,const std::shared_ptr<Value>& rhs){
////            return lhs * (rhs ^ -1);
////        }
//
//        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        // Subtract
//        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        // Subtract Value
//        friend Value operator -(Value& lhs, Value& rhs){
//            auto out = Value(lhs.data - rhs.data, {lhs, rhs}, "-");
//
//            auto backward = [lhs, rhs, out]() mutable{
//                lhs.grad +=  out.grad;
//                rhs.grad -=  out.grad;
//            };
//            out.backward = backward;
//            return out;
//        }
//
//        // Subtract constant
//        friend Value operator -(Value& lhs, float f){
//            auto other = Value(f);
//            auto out = Value(lhs.data - other.data, {lhs, other}, "-");
//
//            auto backward = [lhs, other, out]() mutable {
//                lhs.grad +=  out.grad;
//                other.grad -=  out.grad;
//            };
//            out.backward = backward;
//            return out;
//        }
//
//        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        // Activation functions
//        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        Value tanh(){
//            auto x = this->data;
//            double t = (double)(std::exp(2*x) - 1.0f) / (double)(std::exp(2*x) + 1.0f);
//            auto out = Value(t, {}, "tanh");
//            auto backward = [this,t,out = out]() mutable{
//                this->grad +=  (double)(1 - t*t) * (double)out.grad;
//            };
//            out.backward = backward;
//            return out;
//        }
//
//        Value relu(){
//            auto val = this->data < 0 ? 0 :  this->data;
//            auto out = Value(val, {*this}, "ReLU");
//            auto backward = [this, out=out](){
//                this->grad +=  (out.data > 0) * out.grad;
//            };
//            out.backward = backward;
//            return out;
//        }
//        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        // Topological sort
//        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        // buildTopo
//        void buildTopo(Value v, std::unordered_set<Value, Hash>& visited, std::vector<Value>& topo){
//            if(visited.find(v) == visited.end()){
//                visited.insert(v);
//                for(const auto& child : v.prev){
//                    buildTopo(child, visited, topo);
//                }
//                topo.push_back(v);
//            }
//        }
//        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        // Backprop algo
//        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        // backProp
//        void backProp(){
//            std::vector<Value> topo;
//            std::unordered_set<Value, Hash> visited;
//            auto &shared = *this;
//            buildTopo(shared, visited, topo);
//            printf("Building topo");
//            printf("Total number of nodes in the graph:  %d\n", (int)topo.size());
//            shared.grad = 1;
//            for( auto it = topo.rbegin(); it != topo.rend(); ++it ){
//
//                if((*it).backward)  {
//                    (*it).backward();
//                }
//                //printf("[%s data:%f, grad:%f]\n", (*it)->label.c_str(),(*it)->data, (*it)->grad);
//
//            }
//            printf("\n");
//        }
//
//        bool operator==(const Value& other) const;
//    };
//
//
//    size_t Hash::operator()(const Value value) const {
//        return std::hash<std::string>()(value.op) ^ std::hash<float>()(value.data);
//    }
//
//    bool Value::operator==(const Value& other) const {
//        return data == other.data && op == other.op && prev == other.prev;
//    }
//}
//
//#endif //MICROGRADPP_VALUESTACK_HPP
