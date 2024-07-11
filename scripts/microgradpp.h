// Content from ../include/Value.hpp
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
                    (*it)->backward();
                    //(*it)->clip_gradients(); // Clip gradients here
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


// Content from ../include/Activation.hpp
//
// Created by Gautam Sharma on 7/9/24.
//

#ifndef MICROGRADPP_ACTIVATION_HPP
#define MICROGRADPP_ACTIVATION_HPP

// std libs
#include <unordered_map>
#include <functional>

// micrograd libs


namespace microgradpp{
    enum ActivationType{
        RELU,
        TANH,
        SIGMOID
    };
    class Activation{

        static std::shared_ptr<Value> Relu(const std::shared_ptr<Value>& val){
            return val->relu();
        }
        static std::shared_ptr<Value> TanH(const std::shared_ptr<Value>& val){
            return val->tanh();
        }
        static std::shared_ptr<Value> Sigmoid(const std::shared_ptr<Value>& val){
            return val->sigmoid();
        }

    public:
        static inline std::unordered_map<ActivationType, std::function<std::shared_ptr<Value>(const std::shared_ptr<Value>&)>> mActivationFcn = {
                {ActivationType::RELU, Relu},
                {ActivationType::TANH, TanH},
                {ActivationType::SIGMOID, Sigmoid}
        };

    };
}

#endif //MICROGRADPP_ACTIVATION_HPP


// Content from ../include/Tensor.hpp
//
// Created by Gautam Sharma on 7/7/24.
//

#ifndef MICROGRADPP_TENSOR_HPP
#define MICROGRADPP_TENSOR_HPP

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


// Content from ../include/Neuron.hpp
//
// Created by Gautam Sharma on 6/30/24.
//

#ifndef MICROGRADPP_NEURON_HPP
#define MICROGRADPP_NEURON_HPP


#include <memory>
#include <vector>
#include <random>
#include <stdio.h>
#include <numeric> // for std::inner_product



namespace microgradpp{


    // Function to generate a random float between -1 and 1
    double getRandomFloat() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(-1, 1);
        return dis(gen);
    }

    class Neuron{
    private:
        std::vector<std::shared_ptr<Value>> weights;
        std::shared_ptr<Value> bias = nullptr;
        const ActivationType activation_t;
    public:
        Neuron(size_t nin, const ActivationType& activation_t): activation_t(activation_t){
            for(size_t idx = 0; idx < nin; ++idx){
                this->weights.emplace_back(Value::create(getRandomFloat()));
            }
            this->bias = Value::create(0);
        }

        // For testing
        Neuron(size_t nin, double val,const ActivationType& activation_t = ActivationType::SIGMOID):activation_t(activation_t){
            for(size_t idx = 0; idx < nin; ++idx){
                this->weights.emplace_back(Value::create(val));
            }
            this->bias = Value::create(val);
        }

        void zeroGrad(){
            auto params = this->parameters();
            for(auto&p : params){
                p->grad = 0;
            }
        }

        // Dot product of a Neurons weights with the input
        std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& x) const{
            // Ensure both vectors are of the same size
            if (x.size() != weights.size()) {
                throw std::invalid_argument("Vectors must be of the same length");
            }
            auto sum = Value::create(0.0);

            // Dot product -> dot product is supported from C++20
            // TODO: need to make it more efficient
            for(size_t idx = 0; idx<weights.size() ; ++idx){
                sum += x[idx] * weights[idx];
            }

            // Add bias
            sum += this->bias;

            const auto& activationFcn = Activation::mActivationFcn[activation_t];
            auto result = activationFcn(sum);

            return result;
        }

        std::vector<std::shared_ptr<Value>> parameters() const{
            std::vector<std::shared_ptr<Value>> out;
            std::copy(this->weights.begin(), this->weights.end(), std::back_inserter(out));
            out.emplace_back(this->bias);
            return out;
        }

        void printParameters(){
            printf("Number of Parameters: %d \n", (int)weights.size() + 1);
            for(const auto&param : weights){
                printf("%f, %f", param->data,param->grad);
                printf("\n");
            }
            printf("%f, %f", this->bias->data,this->bias->grad);
            printf("\n");
            printf("\n");
        }

        size_t getParametersSize() const{
            return weights.size() + 1;
        }

    };
}

#endif //MICROGRADPP_NEURON_HPP


// Content from ../include/Layer.hpp
//
// Created by Gautam Sharma on 7/1/24.
//

#ifndef MICROGRADPP_LAYER_HPP
#define MICROGRADPP_LAYER_HPP

#include <vector>
#include <algorithm>




namespace microgradpp {

class Layer{
private:
    std::vector<Neuron> neurons;
public:
    Layer(size_t nin , size_t nout, const ActivationType& activation = ActivationType::SIGMOID){
        for(size_t idx = 0; idx < nout; ++idx){
            this->neurons.emplace_back(nin, activation);
        }
    }

    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x) const{
        std::vector<std::shared_ptr<Value>> out;
        out.reserve(this->neurons.size());
        std::for_each(this->neurons.begin(), this->neurons.end(), [&out, &x]( const auto&neuron){
            out.emplace_back(neuron(x));
        });
        return out;
    }

    void zeroGrad(){
        for(auto& neuron: this->neurons){
            neuron.zeroGrad();
        }
    }

    std::vector<std::shared_ptr<Value>> parameters() const{
        std::vector<std::shared_ptr<Value>> out;
        for(const auto& neuron : neurons){
            for(const auto& p : neuron.parameters()){
                out.push_back(p);
            }
        }
        return out;
    }
};

class MLP{
private:
    std::vector<size_t> sizes;
    std::vector<Layer> layers;
    double learningRate;
public:
    MLP(size_t nin, std::vector<size_t> nouts, const double learningRate=0.0025):learningRate(learningRate){
        sizes.reserve(4);
        sizes.push_back(nin);
        std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes) );
        for(size_t idx=0; idx < sizes.size() - 1 ; ++idx){
            layers.emplace_back(sizes[idx], sizes[idx+1]);
        }
    }

    std::vector<std::shared_ptr<Value>> convertToValue(const std::vector<float>& input){
        std::vector<std::shared_ptr<Value>> out;
        for(size_t idx=0; idx<input.size(); ++idx){
            out.emplace_back(Value::create(input[idx]));
        }
        return out;
    }

    void update(){
        for (auto &p: this->parameters()) {
            p->data += (double)((double)-this->learningRate * (double)p->grad);
        }
    }

    std::vector<float> convertFromValue(const std::vector<std::shared_ptr<Value>>& input){
        std::vector<float> out;
        for(size_t idx=0; idx<input.size(); ++idx){
            out.emplace_back(input[idx]->data);
        }
        return out;
    }

    void zeroGrad(){
        for(auto& layer: this->layers){
            layer.zeroGrad();
        }
    }

    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& input){
        std::vector<std::shared_ptr<Value>> x = input;
        for(const auto& layer : this->layers){
             auto y  = layer(x);
             x = y;

        }
        return x;
    }

    void printParameters(){
        const auto params = this->parameters();
        printf("Num parameters: %d\n", (int)params.size());
        for(const auto& p : params){
            std::cout<< &p << " ";
            printf("[data=%f,grad=%lf]\n", p->data, p->grad);
        }
        printf("\n");
    }

    std::vector<std::shared_ptr<Value>> parameters() const{
        std::vector<std::shared_ptr<Value>> out;
        for(const auto& layer : this->layers){
            for(const auto& p : layer.parameters()){
                out.push_back(p);
            }
        }

        return out;
    }

};

}

#endif //MICROGRADPP_LAYER_HPP


