#ifdef PLOT
#include <matplot/matplot.h>
#endif

#include <iostream>

#include "Value.hpp"
#include "Layer.hpp"
#include "Neuron.hpp"

using namespace std;


int main() {
    using microgradpp::Value;
    using microgradpp::Neuron;
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

//    auto a = Value::create(4.0);
//    a->label = "a";
//    auto b = a->tanh();
//    b->label = "b";
//    auto c = b + a;
//    c->label = "c";
//    auto d = Value::create(16.56);
//    d->label = "d";
//    auto e = d - c;
//    e->label = "e";
//    auto f = e * 67 + 78;
//    f->label = "f";
//    f->backProp();

//auto a = Value::create(64);
//a->label = "a";
//auto b = a ^ 8;
//b->label = "b";
//b->backProp();

//auto a = Value::create(64);
//a->label = "a";
//auto b = a / 8;
//b->label = "b";
//b->backProp();


//auto a = Value::create(64);
//a->label = "a";
//auto b = Value::create(8);
//b->label = "b";
//auto c = a / b;
//c->label = "c";
//c->backProp();

//std::vector<std::vector<float>> xs = {
//        {2.0,3.0,-1.0},
//        {3.0, -1.0, 0.5},
//        {0.5, 1.0, 1.0},
//        {1.0, 1.0, -1.0}
//};

    std::vector<std::vector<std::shared_ptr<Value>>> xs = {
            {Value::create(2.0), Value::create(3.0),  Value::create(-1.0)},
            {Value::create(3.0), Value::create(-1.0), Value::create(0.5)},
            {Value::create(0.5), Value::create(1.0),  Value::create(1.0)},
            {Value::create(1.0), Value::create(1.0),  Value::create(-1.0)},
    };

    std::vector<std::shared_ptr<Value>> sampleInput = {Value::create(1.0), Value::create(2.0),  Value::create(3.0)};

    std::vector<double> ys = {1, -1, -1, 1};
    std::vector<double> x = {2, 3, -1};

    auto mlp = microgradpp::MLP(3, {4,1});

    std::vector<std::shared_ptr<Value>> ysValue;
    for (const auto &yys: ys) {
        ysValue.push_back(Value::create(yys));
    }
//auto k = mlp(ysValue);

    for (auto idx = 0; idx < 5000; ++idx) {
        auto loss = Value::create(0.0);
        //std::cout << "////////////////////////////////////////////////////////////////////////\n";
        std::vector<std::shared_ptr<Value>> ypred;
        //ypred.push_back(mlp(sampleInput)[0]);
        for (auto &xx: xs) {
            for(auto&xxx: xx){
                xxx->grad = 0;
            }
            ypred.push_back(mlp(xx)[0]);

        }
        //std::cout << "Ypred:\n"<< std::endl;
//        for(const auto& p : ypred){
//            std::cout << p << std::endl;
//        }

        for (size_t i = 0; i < ysValue.size(); ++i) {
            loss +=  (ysValue[i] - ypred[i])^2;// std::pow(ys[i] - ypred[i], 2);
        }

//        std::cout << "Loss:" << loss << std::endl;

        mlp.zeroGrad();
//        std::cout << "Before\n";
//        mlp.printParameters();
        loss->backProp();
//        std::cout << "After\n";
//        mlp.printParameters();

        for (auto &p: mlp.parameters()) {
            p->data += (double)((double)-0.0025 * (double)p->grad);
        }
        //std::cout << idx << " " << loss->data << std::endl;

        //mlp.printParameters();

//        auto validation =   {Value::create(24.0), Value::create(3.0),  Value::create(-1.0)};
//        std::cout<< "Prediction " << mlp(validation)[0]->data << std::endl;
    }
}
//    constexpr size_t expectedSize = 3;
//    auto a = Neuron(expectedSize, 12);
//    const std::vector<std::shared_ptr<Value>> x = {Value::create(-1),Value::create(2),Value::create(3)};
//
//    // Forward pass
//    std::shared_ptr<Value> result = a(x);
//    result->backProp();
//    a.printParameters();
//
//    result = a(x);
//    result->backProp();
//    a.printParameters();
//
//    result = a(x);
//    result->backProp();
//    a.printParameters();

//}
