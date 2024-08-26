#include <iostream>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds

#include <sys/resource.h>
#include <sys/time.h>
#include <cassert>
#include "Value.hpp"
#include "Layer.hpp"
#include "Neuron.hpp"
#include "Tensor.hpp"
#include <sys/sysctl.h>

#include <unistd.h> // for getpid
using namespace std;

#include <mach/mach.h>
#include <unistd.h> // for getpid

long getMemoryUsage() {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;

    kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count);
    if (kr != KERN_SUCCESS) {
        std::cerr << "task_info call failed" << std::endl;
        return -1; // Error
    }

    // Return the resident memory size in kilobytes
    return info.resident_size / 1024; // Convert bytes to KB
}

int main() {
    using microgradpp::Value;
    using microgradpp::Neuron;
    using microgradpp::Tensor;



    {

        //long initial_memory_usage = getMemoryUsage();

        // Input data
        Tensor xs = {{1},{2},{3},{4}};

        // Example: 0.2+0.3+-1 = -0.5
        Tensor ys = {1,3,5,7};

        Tensor validation = {{5}};
        constexpr double learningRate = 0.0001;

        //auto mlp = std::make_unique<microgradpp::MLP>(3, 80,80, 1, learningRate);

        auto mlp = microgradpp::MLP(1, {400, 400,1}, learningRate);;


        Tensor ypred;


        // Start learning loop
        auto start = std::chrono::high_resolution_clock::now();
        for (auto idx = 0; idx < 250; ++idx) {
            {
                //std::cout << "////////////////////////////////////////////////////////////////////////\n";
                // Initialize loss
                //initial_memory_usage = getMemoryUsage();
                //std::cout << "Initial memory usage: " << initial_memory_usage << " KB\n";

                std::shared_ptr<Value> loss = Value::create(0.0);;

                //Tensor ypred;

                // Ensure the gradients of inputs is always zero
                xs.zeroGrad();

                // Predict values
                for (const auto &input: xs) {
                    ypred.push_back(mlp(input));
                }

                // Calculate loss
                for (size_t i = 0; i < ys.size(); ++i) {
                    loss +=  (ys.at(i) - ypred.at(i))^2;
                }

                printf("Loss: %f\n", loss->data);

                // Ensure all gradients are zero
                mlp.zeroGrad();

                // Perform backprop
                loss->backProp();

                //auto a = mlp.parameters();
                // Update parameters
                mlp.update();

                ypred.reset();
            }
        }
        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate elapsed time
        auto elapsed  = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Time taken for loop: " << elapsed<< " milliseconds" << std::endl;

        auto start1 = std::chrono::high_resolution_clock::now();
        auto d = mlp(validation[0]);
        std::cout << d[0]->data <<  std::endl;
        auto end1 = std::chrono::high_resolution_clock::now();
        auto elapsed1 =  std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();

        std::cout << "Model prediction time " << elapsed1 << " milliseconds" << std::endl;
    }
}


//int main() {
//    using microgradpp::Value;
//    using microgradpp::Neuron;
//    using microgradpp::Tensor;
////    std::shared_ptr<Value> v  = Value::create(9.0);
////    {
////        auto c = v;
////    }
//
//
//
//    {
//        // Track initial memory usage
//        long initial_memory_usage = getMemoryUsage();
//        //std::cout << "Initial memory usage: " << initial_memory_usage << " KB\n";
//
//
//
//        //printf("Hello from micrograd++\n");
//
//        // Input data
//        Tensor xs = {{0.2, 0.3, -1.0},
//                     {0.4, 0.3, 0.1},
//                     {0.5, 0.1, -0.1},
//                     {1.0, 1.0, -1.0}};
//
////        std::vector<std::vector<std::shared_ptr<Value>>> xs = {{Value::create(0.2), Value::create(0.3), Value::create(-1.0)},
////                                                               {Value::create(0.4), Value::create(0.3), Value::create(0.1)},
////                                                               {Value::create(0.5), Value::create(0.1), Value::create(-0.1)},
////                                                               {Value::create(1.0), Value::create(1.0), Value::create(-1.0)}};
//
//        // Expected output:
//        // Sum of each row in the input should be equal to each entry in ys
//        // Example: 0.2+0.3+-1 = -0.5
//        Tensor ys = {-0.5, 0.8, 0.5, 1};
//
//        // For plotting
////        std::vector<double> lossValues;
////        std::vector<double> iterations;
//
//        /*
//         * Initialize micrograd
//         * @input : 3 params
//         * @layer 1 = 4 neurons
//         * @layer 2 = 1 neuron -> output
//         */
//        constexpr double learningRate = 0.00025;
//
//        auto mlp = std::make_unique<microgradpp::MLP>(3, 80,80, 1, learningRate);
//
////        {
////            auto mlp = microgradpp::MLP(3, {4, 1}, learningRate);;
////            mlp.update();
////        }
//
//        std::shared_ptr<Value> loss = Value::create(0.0);;
//        Tensor ypred;
//
//
//        //std::vector<std::shared_ptr<Value>> ys  = {Value::create(-0.5),Value::create(0.8),Value::create(0.5),Value::create(1)};
//        //std::vector<std::vector<std::shared_ptr<Value>>> ypred;
//        // Start learning loop
//        auto start = std::chrono::high_resolution_clock::now();
//        for (auto idx = 0; idx < 1000; ++idx) {
//            {
//                //std::cout << "////////////////////////////////////////////////////////////////////////\n";
//                // Initialize loss
//                initial_memory_usage = getMemoryUsage();
//                //std::cout << "Initial memory usage: " << initial_memory_usage << " KB\n";
//
//                loss->reset();
//
//                //Tensor ypred;
//
//                // Ensure the gradients of inputs is always zero
//                xs.zeroGrad();
//
//                // Predict values
//                for (const auto &input: xs) {
//                    ypred.push_back((*mlp)(input));
//                }
//
//                // Calculate loss
//                for (size_t i = 0; i < ys.size(); ++i) {
//                    loss +=  (ys.at(i) - ypred.at(i))^2;
//                }
//
//
//                // Ensure all gradients are zero
//                mlp->zeroGrad();
//
//                // Perform backprop
//                loss->backProp();
//
//                // Update parameters
//                mlp->update();
//
//                //std::cout << "Extra Memory usage: " << getMemoryUsage() - initial_memory_usage << " KB\n";
//
//                //std::this_thread::sleep_for(std::chrono::milliseconds(100));
//
//                std::cout << "Iteration : " << idx << " " << "Loss: " << loss->data << "Extra Memory usage: " << getMemoryUsage() - initial_memory_usage << " KB\n";
//                ypred.reset();
//            }
//        }
//        // Record end time
//        auto end = std::chrono::high_resolution_clock::now();
//
//        // Calculate elapsed time
//        std::chrono::duration<double> elapsed = end - start;
//
//        std::cout << "Time taken for loop: " << elapsed.count() << " seconds" << std::endl;
//
//        // Track final memory usage
//        long final_memory_usage = getMemoryUsage();
//        std::cout << "Final memory usage: " << final_memory_usage << " KB\n";
//        std::cout << "Memory increase: " << (final_memory_usage - initial_memory_usage) << " KB\n";
//
//    }
//
//    assert(Value::labelIdx == 0);
//
//    //int c = 2;
//
//}




///

//
//
//#include <iostream>
//#include <thread>         // std::this_thread::sleep_for
//#include <chrono>         // std::chrono::seconds
//
//#include <sys/resource.h>
//#include <sys/time.h>
//
//#include "Value.hpp"
//#include "Layer.hpp"
//#include "Neuron.hpp"
//#include "Tensor.hpp"
//#include <sys/sysctl.h>
//#include <unistd.h> // for getpid
//using namespace std;
//
//#include <mach/mach.h>
//#include <unistd.h> // for getpid
//
//long getMemoryUsage() {
//    mach_task_basic_info_data_t info;
//    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
//
//    kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count);
//    if (kr != KERN_SUCCESS) {
//        std::cerr << "task_info call failed" << std::endl;
//        return -1; // Error
//    }
//
//    // Return the resident memory size in kilobytes
//    return info.resident_size / 1024; // Convert bytes to KB
//}
//
//
//int main() {
//    using microgradpp::Value;
//    using microgradpp::Neuron;
//    using microgradpp::Tensor;
//    std::shared_ptr<Value> v  = Value::create(9.0);
//    {
//        auto c = v;
//    }
//
//
//
//    {
//        // Track initial memory usage
//        long initial_memory_usage = getMemoryUsage();
//        std::cout << "Initial memory usage: " << initial_memory_usage << " KB\n";
//
//
//
//        printf("Hello from micrograd++\n");
//
//        // Input data
////        Tensor xs = {0.2};
//
//        // Expected output:
//        // Sum of each row in the input should be equal to each entry in ys
//        // Example: 0.2+0.3+-1 = -0.5
//        Tensor ys = {-0.5,1};
//
//        // For plotting
//        std::vector<double> lossValues;
//        std::vector<double> iterations;
//
//        /*
//         * Initialize micrograd
//         * @input : 3 params
//         * @layer 1 = 4 neurons
//         * @layer 2 = 1 neuron -> output
//         */
//        constexpr double learningRate = 0.0025;
//
//        auto mlp = microgradpp::MLP(1, {1}, learningRate);
//
//        std::shared_ptr<Value> loss = Value::create(0.0);
//
//        Tensor ypred;
//        Tensor xs = {0.2,0.3};
//        // Start learning loop
//        for (auto idx = 0; idx < 200; ++idx) {
//            {
//                //auto mlp = microgradpp::MLP(3, {4, 1}, learningRate);
//                std::cout << "////////////////////////////////////////////////////////////////////////\n";
//
//                // Initialize loss
//                initial_memory_usage = getMemoryUsage();
//                std::cout << "Initial memory usage: " << initial_memory_usage << " KB\n";
//                //loss->reset();
//
////                while (loss.use_count()) {
////                    loss.reset();
////                }
//
//
//                //loss = Value::create(0.0);;
//                //        std::shared_ptr<Value> loss = Value::create(0.0);;
////                loss->data = 0.0;
////                loss->grad = 0.0;
////                loss->prev.clear();
//                loss = Value::create(0.0);
//
//
//                // Ensure the gradients of inputs is always zero
//                Tensor xs = {0.2,0.3};
//
//
//                // Predict values
//                for (const auto &input: xs) {
//                    ypred.push_back(mlp(input));
//                }
//
//
//                // Calculate loss
//                for (size_t i = 0; i < ys.size(); ++i) {
//                    loss += (ys.at(i) - ypred.at(i)) ^ 2;
//                }
//
//                // Ensure all gradients are zero
//                mlp.zeroGrad();
//
//                //mlp.printUseCount();
//                // Perform backprop
//                loss->backProp();
//                //mlp.printUseCount();
//                // Update parameters
//                mlp.update();
//                mlp.printParameters();
//                //mlp.printUseCount();
//
//                //mlp.clear();
//                //mlp.printUseCount();
//                //loss.reset();
//                std::cout << "Extra Memory usage: " << getMemoryUsage() - initial_memory_usage << " KB\n";
//                //mlp.printUseCount();
//                //std::cout << sizeof(mlp) << std::endl;
//
//                std::this_thread::sleep_for(std::chrono::milliseconds(100));
//
//                //std::cout << "Iteration : " << idx << " " << "Loss: " << loss->data << std::endl;
//                ypred.reset();
//                //mlp.clear();
//            }
//        }
//
//        // Track final memory usage
//        long final_memory_usage = getMemoryUsage();
//        std::cout << "Final memory usage: " << final_memory_usage << " KB\n";
//        std::cout << "Memory increase: " << (final_memory_usage - initial_memory_usage) << " KB\n";
//    }
//
//    //int c = 2;
//
//}