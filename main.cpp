#ifdef PLOT
#include <matplot/matplot.h>
#endif

#include "Value.hpp"


using namespace std;


int main(){
using microgradpp::Value;
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


auto a = Value::create(64);
a->label = "a";
auto b = Value::create(8);
b->label = "b";
auto c = a / b;
c->label = "c";
c->backProp();
}
//"command" : ["a = Value(4.0)", "b = a.tanh()", "c = b + a", "d = Value(16.56)", "e = d - c","e.backward()"],