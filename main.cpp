#ifdef PLOT
#include <matplot/matplot.h>
#endif


#include <cstdio>
#include <vector>
#include <string>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <math.h>
#include <memory>

using namespace std;
class Value;

struct Hash {
    size_t operator()(const std::shared_ptr<Value> value) const;
};

class Value : public std::enable_shared_from_this<Value>{
private:
    Value(float data, const vector<std::shared_ptr<Value>>& children={}, const string& op= "", const string& label="")
            : data(data), prev(children.begin(), children.end()), op(op), label(label) {}
public:
    static std::shared_ptr<Value> create(float data, const vector<std::shared_ptr<Value>>& children = {}, const string& op = "", const string& label = "") {
        return std::shared_ptr<Value>(new Value(data, children, op, label));
    }

    float data;
    float grad = 0;
    std::function<void()> backward = nullptr;
    unordered_set<std::shared_ptr<Value>, Hash> prev;
    string op;
    string label;

    Value(const Value&v) = default;


    // Add Ref
    // Add Ref
    friend std::shared_ptr<Value> operator +(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
        auto out = Value::create(lhs->data + rhs->data, {lhs, rhs}, "+");

        auto backward = [lhs, rhs, out]() mutable {
            lhs->grad += out->grad;
            rhs->grad += out->grad;
        };

        out->backward = backward;
        return out;
    }

//    Value operator+(float otherData){
//        auto other = Value(otherData, {});
//        auto out = Value(this->data + other.data, {this, &other}, "+");
//
//    }

    // Multiply
    friend std::shared_ptr<Value> operator *(const std::shared_ptr<Value>& lhs,  const std::shared_ptr<Value>& rhs){
        auto out = Value::create(lhs->data * rhs->data, {lhs, rhs}, "*");

        auto backward = [&](){
            lhs->grad += rhs->data * out->grad;
            rhs->grad += lhs->data * out->grad;
        };

        out->backward = backward;
        return out;
    }

    // Subtract
    friend std::shared_ptr<Value> operator -(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs){
        auto out = Value::create(lhs->data - rhs->data, {lhs, rhs}, "*");

        auto backward = [&](){
            lhs->grad +=  out->grad;
            rhs->grad -=  out->grad;
        };

        out->backward = backward;
        return out;
    }

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
        auto backward = [&](){
            this->grad +=  (out->data > 0) * out->grad;
        };
        out->backward = backward;
        return out;
    }

    // buildTopo
    void buildTopo(std::shared_ptr<Value> v, unordered_set<std::shared_ptr<Value>, Hash>& visited, vector<std::shared_ptr<Value>>& topo){
        if(visited.find(v) == visited.end()){
            visited.insert(v);
            for(const auto& child : v->prev){
                buildTopo(child, visited, topo);
            }
            topo.push_back(v);
        }
    }

    // backProp
    void backProp(){
        vector<std::shared_ptr<Value>> topo;
        unordered_set<std::shared_ptr<Value>, Hash> visited;
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
    return hash<string>()(value.get()->op) ^ hash<float>()(value.get()->data);
}

bool Value::operator==(const Value& other) const {
    return data == other.data && op == other.op && prev == other.prev;
}

int main(){

printf("Hello from micrograd++\n");
////auto a = Value(3.0);
//auto a = Value(2.0);
//a.label = "a";
//auto b = Value(-1.0);
//b.label = "b";
//auto c = Value(1.0);
//c.label = "c";
//auto d = Value(1.0);
//d.label = "d";
//auto e = a*a;
//e.label = "a^2";
//auto f = b*b;
//f.label = "b^2";
//auto g = c*c;
//g.label = "c^2";
//auto h = d*d;
//h.label = "d^2";
//auto i = e + f;
//i.label = e.label + " + " +  f.label;
//auto j = g + h;
//j.label = g.label + " + " +  h.label;
//auto k = i+ j;
//k.label = i.label + " + " +  j.label;
//printf("%f", k.data);
//k.backProp();

//auto a = Value(1.0);
//a.label = "a";
//
//auto b = Value(2.0);
//b.label = "b";
//
//auto c = a - b;
//    c.label = "c";
//
//auto d = c.relu();
//d.label = "d";
//
//d.backProp();
//printf("%f", d.grad);

//auto a = Value(4.0);
//a.label = "a";
//auto b = a.tanh();
//b.backProp();
    auto a = Value::create(4.0);
    a->label = "a";
    auto b = a->tanh();
    b->label = "b";
    auto c = b + a;
    auto d = Value::create(16.56);
    auto e = d - c;
    e->label = "e";
    e->backProp();


}
//"command" : ["a = Value(4.0)", "b = a.tanh()", "c = b + a", "d = Value(16.56)", "e = d - c","e.backward()"],