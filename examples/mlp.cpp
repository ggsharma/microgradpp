#include <iostream>
#include <chrono>         // std::chrono::seconds
#include <cassert>


//Microgradpp headers
#include "Value.hpp"
#include "Layer.hpp"
#include "Neuron.hpp"
#include "Tensor.hpp"
#include "Autograd.hpp"
#include "Algorithms.hpp"
#include "LossFunctions.hpp"

using namespace std;

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#include <sys/resource.h>
#include <sys/types.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif

long getMemoryUsage() {
#ifdef __APPLE__
    // macOS implementation
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;

    kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count);
    if (kr != KERN_SUCCESS) {
        std::cerr << "task_info call failed" << std::endl;
        return -1; // Error
    }

    // Return the resident memory size in kilobytes
    return info.resident_size / 1024; // Convert bytes to KB

#elif defined(__linux__)
    // Linux implementation
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        std::cerr << "getrusage call failed" << std::endl;
        return -1; // Error
    }

    // Return resident memory size in kilobytes
    return usage.ru_maxrss; // ru_maxrss is already in kilobytes

    #elif defined(_WIN32)
    // Windows implementation
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        // Return the working set size (resident set size) in kilobytes
        return pmc.WorkingSetSize / 1024; // Convert bytes to KB
    } else {
        std::cerr << "GetProcessMemoryInfo call failed" << std::endl;
        return -1; // Error
    }

    #else
    #error "Unsupported platform"
#endif
}



int main() {
    using microgradpp::Value;
    using microgradpp::Neuron;
    using microgradpp::Tensor;
    using microgradpp::algorithms::MLP;
    using microgradpp::Autograd;
    using microgradpp::loss::MeanSquaredError;

    {
        // Track initial memory usage
        long initial_memory_usage = getMemoryUsage();

        printf("Hello from micrograd++\n");

        // Input data
        Tensor xs = {{-0.6766,  0.8353, -0.9439,  0.4799,  0.6168,  0.8016,  0.6596,  0.6993,
                             0.8828,  0.5242},
                     {-0.2141,  0.1933, -0.7998,  0.0819,  0.6718,  0.3808,  0.5816,  0.2885,
                             -0.4778,  0.0306},
                     {-0.6099,  0.5842, -0.3745, -0.8219,  0.6124, -0.1630, -0.1224,  0.1551,
                             -0.5373,  0.1043},
                     {-0.3497, -0.3523, -0.1984, -0.4061, -0.6675, -0.1741,  0.4931, -0.0871,
                             0.0267,  0.2483},
                     { 0.1297,  0.9122, -0.2129,  0.7722,  0.0825, -0.0995, -0.4025, -0.4215,
                             -0.1012,  0.4956},
                     {-0.0448, -0.1437,  0.3143, -0.9829,  0.1245, -0.8198,  0.2691, -0.9047,
                             -0.7453, -0.7846},
                     {0.0870, -0.2442,  0.1773,  0.2571,  0.2505,  0.2470,  0.3236,  0.9934,
                             0.6233, -0.1927},
                     {-0.2964,  0.4280, -0.3200, -0.1161,  0.5686,  0.6662, -0.1916,  0.0113,
                             0.2825,  0.0549},
                     { 0.3629,  0.2835,  0.7314, -0.4154, -0.6252, -0.5470, -0.9598, -0.2905,
                             -0.5402,  0.7025},
                     {-0.5084,  0.5187,  0.9158,  0.0918, -0.7394,  0.8970, -0.9747,  0.2243,
                             0.2184, -0.0035},
                     {-0.6667, -0.1672, -0.3710, -0.0728, -0.0996,  0.1241,  0.9159, -0.6808,
                             0.3783, -0.3842},
                     {-0.2431, -0.7049,  0.3577,  0.8191,  0.5694,  0.6286, -0.8143, -0.5630,
                             -0.0369, -0.6643},
                     {-0.0254, -0.1791,  0.2230,  0.6701, -0.1998, -0.7423, -0.0505, -0.6672,
                             0.0114,  0.8584},
                     { 0.6508,  0.9048, -0.3075, -0.1011, -0.1199,  0.6666, -0.0782,  0.3139,
                             -0.3874, -0.7753},
                     {-0.2426,  0.7815,  0.4511,  0.4927, -0.4100, -0.4285, -0.5225, -0.3485,
                             0.9928,  0.3645},
                     {-0.2752,  0.0956,  0.2617, -0.1669, -0.8581,  0.7859,  0.3837, -0.8880,
                             -0.0896,  0.3921},
                     { 0.2129, -0.7676, -0.9401,  0.0475,  0.4118, -0.7637, -0.6371,  0.0691,
                             -0.1854,  0.9865},
                     {-0.2638, -0.1132,  0.1891,  0.1182,  0.9774, -0.7161,  0.2602, -0.4918,
                             0.8282, -0.8364},
                     { 0.9084,  0.2898, -0.8604, -0.2312, -0.9680,  0.9569,  0.4857,  0.1534,
                             0.7059,  0.1348},
                     { 0.9870, -0.0352, -0.0840,  0.5445,  0.8870,  0.1733,  0.7722,  0.4949,

                             0.7470, -0.3344}};
//         Expected output:
//         Sum of each row in the input should be equal to each entry in ys
//         Example: 0.2+0.3+-1 = -0.5
        Tensor ys = { 0.3879,  0.0737, -0.1173, -0.1467,  0.1155, -0.3718,  0.2522,  0.1087,
                      -0.1298,  0.0640, -0.1024, -0.0652, -0.0101,  0.0767,  0.1131, -0.0359,
                      -0.1566, -0.0048,  0.1575,  0.4152};

        constexpr float learningRate = 0.001;
        constexpr size_t numIterations = 100000;

        // Define loss function
        MeanSquaredError lossFcn;

        /**
         * Initialize micrograd
         * @input : 10 params
         * @layer 1 = 100 neurons
         * @layer 2 = 100 neurons
         * @layer = output
         */
        auto mlp = std::make_unique<MLP>(10, 10,10,1, learningRate);

        // Initialize prediction Tensor
        Tensor ypred;

        // Start learning loop
        auto start = std::chrono::high_resolution_clock::now();

        auto init =  getMemoryUsage();;

        for (auto idx = 0; idx < numIterations; ++idx) {

                //std::cout << "////////////////////////////////////////////////////////////////////////\n";

                __MICROGRADPP_CLEAR__

                initial_memory_usage = getMemoryUsage();

                // Ensure the gradients of inputs is always zero
                xs.zeroGrad();

                // Predict values
                for (const auto &input: xs) {
                    ypred.push_back((*mlp)(input));  //
                }


                auto loss = lossFcn(ys, ypred);

                // Ensure all gradients are zero
                mlp->zeroGrad();

                // Perform backprop
                loss->backProp();

                // Update parameters
                mlp->update();

                std::cout << "Extra Memory usage: " << getMemoryUsage() - initial_memory_usage << " KB\n";

                std::cout << "Iteration : " << idx << " " << "Loss: " << loss->data << "Extra Memory usage: " << getMemoryUsage() - initial_memory_usage << " KB\n";

                // Reset prediction
                ypred.reset();

        }

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate elapsed time
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Time taken for loop: " << elapsed.count() << " seconds" << std::endl;

        // Track final memory usage
        long final_memory_usage = getMemoryUsage();
        std::cout << "Final memory usage: " << final_memory_usage << " KB\n";
        std::cout << "Memory increase: " << (final_memory_usage - init) << " KB\n";
    }

    assert(Value::labelIdx == 0);

}




