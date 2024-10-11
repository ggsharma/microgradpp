#ifdef PLOT
#include <matplot/matplot.h>
#endif

#include <iostream>

#include "Value.hpp"
#include "Layer.hpp"
#include "Neuron.hpp"
#include "Tensor.hpp"
#include "Algorithms.hpp"

using namespace std;

#include "base/BaseMultiLayerPerceptron.hpp"
#include "Tensor.hpp"
#include "nn/NeuralNet.hpp"
#include "core/Sequential.hpp"
#include "core/MppCore.hpp"
namespace microgradpp{

    using microgradpp::base::BaseMultiLayerPerceptron;
    using microgradpp::core::Sequential;
    using microgradpp::core::MppCore;
    using namespace microgradpp::nn;
    class Example_MLP : public microgradpp::base::BaseMultiLayerPerceptron{
    public:

        Example_MLP():
                BaseMultiLayerPerceptron(Sequential({nn::Linear(10,8),
                                                                            nn::ReLU(),
                                                                            nn::Linear(8,1)


                                                                            }))
        {

        }



         Tensor forward(Tensor input) override{
            Tensor t;
            return t;
        };


//        Tensor forward(Tensor in){
//
//        }
    };

}


int main() {
    using microgradpp::Value;
    using microgradpp::Neuron;
    using microgradpp::Tensor;
    using microgradpp::algorithms::MLP;
    printf("Hello from micrograd++\n");
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
    microgradpp::Example_MLP m ;
    m.print();
    m.printParameters();
    m(xs[0]);

}
