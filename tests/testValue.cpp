//
// Created by Gautam Sharma on 6/30/24.
//

#include "utils.hpp"
#include "GradTester.hpp"
#include "Value.hpp"
#include <chrono>

using microgradpp::Value;

 int main(){
     // Get the starting timestamp
     auto start = std::chrono::high_resolution_clock::now();

    //testAddValue
     {
         auto variables = microgradpp::utils::readVariablesFromJson("test_add_value_output.json");

         auto a = Value::create(64);
         a->label = "a";
         auto b = Value::create(8);
         b->label = "b";
         auto c = Value::add(a,b);
         c->label = "c";
         c->backProp();
         microgradpp::GradTester::equals<float>(a->data, variables["a"].data, "testAddValue a data");
         microgradpp::GradTester::equals<float>(a->grad, variables["a"].grad, "testAddValue a grad");
         microgradpp::GradTester::equals<float>(b->data, variables["b"].data, "testAddValue b data");
         microgradpp::GradTester::equals<float>(b->grad, variables["b"].grad, "testAddValue b grad");
         microgradpp::GradTester::equals<float>(c->data, variables["c"].data, "testAddValue c data");
         microgradpp::GradTester::equals<float>(c->grad, variables["c"].grad, "testAddValue c grad");
     }
     //testAddConstant
     {
         auto variables = microgradpp::utils::readVariablesFromJson("test_add_constant_output.json");

         auto a = Value::create(64);
         a->label = "a";
         auto b = Value::add(a ,8.90);
         b->label = "b";
         b->backProp();

         microgradpp::GradTester::equals<float>(a->data, variables["a"].data, "testAddConstant a data");
         microgradpp::GradTester::equals<float>(a->grad, variables["a"].grad, "testAddConstant a grad");
         microgradpp::GradTester::equals<float>(b->data, variables["b"].data, "testAddConstant b data");
         microgradpp::GradTester::equals<float>(b->grad, variables["b"].grad, "testAddConstant b grad");
     }
     //testSubtractValue
     {
         auto variables = microgradpp::utils::readVariablesFromJson("test_subtract_value_output.json");

         auto a = Value::create(64);
         a->label = "a";
         auto b = Value::create(8);
         b->label = "b";
         auto c = Value::subtract(a,b);
         c->label = "c";
         c->backProp();


         microgradpp::GradTester::equals<float>(a->data, variables["a"].data, "testSubtractValue a data");
         microgradpp::GradTester::equals<float>(a->grad, variables["a"].grad, "testSubtractValue a grad");
         microgradpp::GradTester::equals<float>(b->data, variables["b"].data, "testSubtractValue b data");
         microgradpp::GradTester::equals<float>(b->grad, variables["b"].grad, "testSubtractValue b grad");
         microgradpp::GradTester::equals<float>(c->data, variables["c"].data, "testSubtractValue c data");
         microgradpp::GradTester::equals<float>(c->grad, variables["c"].grad, "testSubtractValue c grad");
     }

     //testSubtractConstant
     {
         auto variables = microgradpp::utils::readVariablesFromJson("test_subtract_constant_output.json");

         auto a = Value::create(64);
         a->label = "a";
         auto b = Value::subtract(a,8.90);
         b->label = "b";
         b->backProp();

         microgradpp::GradTester::equals<float>(a->data, variables["a"].data, "testSubtractConstant a data");
         microgradpp::GradTester::equals<float>(a->grad, variables["a"].grad, "testSubtractConstant a grad");
         microgradpp::GradTester::equals<float>(b->data, variables["b"].data, "testSubtractConstant b data");
         microgradpp::GradTester::equals<float>(b->grad, variables["b"].grad, "testSubtractConstant b grad");
     }

     //testMultiplyValue
     {
         auto variables = microgradpp::utils::readVariablesFromJson("test_multiply_value_output.json");

         auto a = Value::create(64);
         a->label = "a";
         auto b = Value::create(8);
         b->label = "b";
         auto c = Value::multiply(a,b);
         c->label = "c";
         c->backProp();


         microgradpp::GradTester::equals<float>(a->data, variables["a"].data, "testMultiplyValue a data");
         microgradpp::GradTester::equals<float>(a->grad, variables["a"].grad, "testMultiplyValue a grad");
         microgradpp::GradTester::equals<float>(b->data, variables["b"].data, "testMultiplyValue b data");
         microgradpp::GradTester::equals<float>(b->grad, variables["b"].grad, "testMultiplyValue b grad");
         microgradpp::GradTester::equals<float>(c->data, variables["c"].data, "testMultiplyValue c data");
         microgradpp::GradTester::equals<float>(c->grad, variables["c"].grad, "testMultiplyValue c grad");
     }

     //testMultiplyConstant
     {
         auto variables = microgradpp::utils::readVariablesFromJson("test_multiply_constant_output.json");

         auto a = Value::create(64);
         a->label = "a";
         auto b = Value::multiply(a,8.90);
         b->label = "b";
         b->backProp();

         microgradpp::GradTester::equals<float>(a->data, variables["a"].data, "testMultiplyConstant a data");
         microgradpp::GradTester::equals<float>(a->grad, variables["a"].grad, "testMultiplyConstant a grad");
         microgradpp::GradTester::equals<float>(b->data, variables["b"].data, "testMultiplyConstant b data");
         microgradpp::GradTester::equals<float>(b->grad, variables["b"].grad, "testMultiplyConstant b grad");
     }

     // testDivideValue
     {
         auto variables = microgradpp::utils::readVariablesFromJson("test_divide_value_output.json");

         auto a = Value::create(64);
         a->label = "a";
         auto b = Value::create(8);
         b->label = "b";
         auto c =  Value::divide(a,b);
         c->label = "c";
         c->backProp();


         microgradpp::GradTester::equals<float>(a->data, variables["a"].data, "testDivideValue a data");
         microgradpp::GradTester::equals<float>(a->grad, variables["a"].grad, "testDivideValue a grad");
         microgradpp::GradTester::equals<float>(b->data, variables["b"].data, "testDivideValue b data");
         microgradpp::GradTester::equals<float>(b->grad, variables["b"].grad, "testDivideValue b grad");
         microgradpp::GradTester::equals<float>(c->data, variables["c"].data, "testDivideValue c data");
         microgradpp::GradTester::equals<float>(c->grad, variables["c"].grad, "testDivideValue c grad");
     }

    // testDivideConstant
     {
         auto variables = microgradpp::utils::readVariablesFromJson("test_divide_constant_output.json");

         auto a = Value::create(64);
         a->label = "a";
         auto b =  Value::divide(a,8);
         b->label = "b";
         b->backProp();

         microgradpp::GradTester::equals<float>(a->data, variables["a"].data, "testDivideConstant a data");
         microgradpp::GradTester::equals<float>(a->grad, variables["a"].grad, "testDivideConstant a grad");
         microgradpp::GradTester::equals<float>(b->data, variables["b"].data, "testDivideConstant b data");
         microgradpp::GradTester::equals<float>(b->grad, variables["b"].grad, "testDivideConstant b grad");
     }
     // testTanh
     {
         auto variables = microgradpp::utils::readVariablesFromJson("test_tanh_output.json");

         auto a = Value::create(7.89);
         a->label = "a";
         auto b = Value::tanh(a);
         b->label = "b";
         b->backProp();
         microgradpp::GradTester::equals<float>(a->data, variables["a"].data, "testTanh a data");
         microgradpp::GradTester::equals<float>(a->grad, variables["a"].grad, "testTanh a grad");
         microgradpp::GradTester::equals<float>(b->data, variables["b"].data, "testTanh b data");
         microgradpp::GradTester::equals<float>(b->grad, variables["b"].grad, "testTanh b grad");
     }
     // testRelu
     {
         auto variables = microgradpp::utils::readVariablesFromJson("test_relu_output.json");

         auto a = Value::create(7.89);
         a->label = "a";
         auto b = Value::relu(a);
         b->label = "b";
         b->backProp();
         microgradpp::GradTester::equals<float>(a->data, variables["a"].data, "testRelu a data");
         microgradpp::GradTester::equals<float>(a->grad, variables["a"].grad, "testRelu a grad");
         microgradpp::GradTester::equals<float>(b->data, variables["b"].data, "testRelu b data");
         microgradpp::GradTester::equals<float>(b->grad, variables["b"].grad, "testRelu b grad");
     }

     // testReluLong
     {
         auto variables = microgradpp::utils::readVariablesFromJson("test_value_relu_long_output.json");

         auto a = Value::create(0.4);
         a->label = "a";
         auto b = Value::create(0.03);
         b->label = "b";

         auto c = Value::multiply(a,b);
         c->label = "c";
         auto d = Value::relu(c);
         d->label = "d";
         d->_backward();
         microgradpp::GradTester::equals<float>(a->data, variables["a"].data, "testRelu a data");
         microgradpp::GradTester::equals<float>(a->grad, variables["a"].grad, "testRelu a grad");
         microgradpp::GradTester::equals<float>(b->data, variables["b"].data, "testRelu b data");
         microgradpp::GradTester::equals<float>(b->grad, variables["b"].grad, "testRelu b grad");
         microgradpp::GradTester::equals<float>(c->data, variables["c"].data, "testRelu c data");
         microgradpp::GradTester::equals<float>(c->grad, variables["c"].grad, "testRelu c grad");
         microgradpp::GradTester::equals<float>(d->data, variables["d"].data, "testRelu d data");
         microgradpp::GradTester::equals<float>(d->grad, variables["d"].grad, "testRelu d grad");
     }
     // testPlusEquals
     // TODO
//     {
//         auto variables = microgradpp::utils::readVariablesFromJson("test_plus_equals_output.json");
//
//         auto a = Value::create(64);
//         a->label = "a";
//         auto b = Value::create(4.89);
//         b->label = "b";
//         b += a;
//         b->backProp();
//         microgradpp::GradTester::equals<float>(a->data, variables["a"].data, "testPlusEquals a data");
//         microgradpp::GradTester::equals<float>(a->grad, variables["a"].grad, "testPlusEquals a grad");
//         microgradpp::GradTester::equals<float>(b->data, variables["b"].data, "testPlusEquals b data");
//         microgradpp::GradTester::equals<float>(b->grad, variables["b"].grad, "testPlusEquals b grad");
//     }
//
//
//     // testMultiplyEquals
//     {
//         auto variables = microgradpp::utils::readVariablesFromJson("test_multiply_equals_output.json");
//
//         auto a = Value::create(64);
//         a->label = "a";
//         auto b = Value::create(4.89);
//         b->label = "b";
//         b *= a;
//         b->backProp();
//         microgradpp::GradTester::equals<float>(a->data, variables["a"].data, "testMultiplyEquals a data");
//         microgradpp::GradTester::equals<float>(a->grad, variables["a"].grad, "testMultiplyEquals a grad");
//         microgradpp::GradTester::equals<float>(b->data, variables["b"].data, "testMultiplyEquals b data");
//         microgradpp::GradTester::equals<float>(b->grad, variables["b"].grad, "testMultiplyEquals b grad");
//     }

     // Get the ending timestamp
     auto end = std::chrono::high_resolution_clock::now();

     // Calculate the duration
     std::chrono::duration<double> duration = end - start;

     // Output the duration in seconds
     std::cout << "Time taken by testValue: " << duration.count() << " seconds" << std::endl;

}