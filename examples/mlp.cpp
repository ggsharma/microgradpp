#include <iostream>
#include "Value.hpp"
#include "Layer.hpp"
#include "Neuron.hpp"
#include "Tensor.hpp"


using namespace std;


int main() {
    //using namespace matplot;

    using microgradpp::Value;
    using microgradpp::Neuron;
    using microgradpp::Tensor;

    printf("Hello from micrograd++\n");

    // Input data
    Tensor xs = {{0.2,0.3,-1.0}, {0.4,0.3,0.1},{0.5,0.1,-0.1}, {1.0,1.0,-1.0}};

    // Expected output:
    // Sum of each row in the input should be equal to each entry in ys
    // Example: 0.2+0.3+-1 = -0.5
    Tensor ys = {-0.5, 0.8, 0.5, 1};

    // For plotting
    std::vector<double> lossValues;
    std::vector<double> iterations;

    /*
     * Initialize micrograd
     * @input : 3 params
     * @layer 1 = 4 neurons
     * @layer 2 = 1 neuron -> output
     */
    constexpr double learningRate = 0.0025;

    auto mlp = microgradpp::MLP(3, {4,1}, learningRate);

    std::shared_ptr<Value> loss;



    // Start learning loop
    for (auto idx = 0; idx < 50; ++idx) {
        // Initialize loss
        loss = Value::create(0.0);

        std::cout << "////////////////////////////////////////////////////////////////////////\n";

        Tensor ypred;

        // Ensure the gradients of inputs is always zero
        xs.zeroGrad();

        // Predict values
        for (const auto &input: xs) {
            ypred.push_back(mlp(input));
        }

        // Calculate loss
        for (size_t i = 0; i < ys.size(); ++i) {
            loss +=  (ys.at(0,i) - ypred.at(0,i))^2;
        }

        // Plot loss
        iterations.push_back(idx);
        lossValues.push_back(loss->data);

        // Ensure all gradients are zero
        mlp.zeroGrad();

        // Perform backprop
        loss->backProp();

        // Update parameters
        mlp.update();

        std::cout << idx << " " << loss->data << std::endl;
    }


}