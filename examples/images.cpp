//
// Created by Gautam Sharma on 7/7/24.
//


#include <iostream>
#include "Value.hpp"
#include "Layer.hpp"
#include "Tensor.hpp"
#include "Algorithms.hpp"
#include "TypeDefs.hpp"
#include "Autograd.hpp"
#include "LossFunctions.hpp"
// This example needs openCV to compile

// For visualization
#include <opencv2/opencv.hpp>

cv::Mat vectorToMat(const std::vector<float>& vec, int rows, int cols, int channels) {
    if (vec.size() != rows * cols * channels) {
        std::cerr << "Size of vector does not match given dimensions" << std::endl;
        return {};
    }

    // Create a Mat with the appropriate type and dimensions
    cv::Mat mat(rows, cols, CV_32FC3);

    // Copy data from the vector to the Mat
    std::memcpy(mat.data, vec.data(), vec.size() * sizeof(float));

    return mat;
}
// Function to resize an image
cv::Mat resizeImage(const cv::Mat& image, int newWidth, int newHeight) {
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(newWidth, newHeight));
    return resizedImage;
}

std::vector<float> normalizeVector(const std::vector<float>& vec) {
    auto minmax = std::minmax_element(vec.begin(), vec.end());
    float minVal = *minmax.first;
    float maxVal = *minmax.second;
    float range = maxVal - minVal;

    std::vector<float> normalizedVec;
    normalizedVec.reserve(vec.size());

    for (const float& val : vec) {
        float normalizedVal = val/255;// 2 * ((val - minVal) / range) - 1;
        normalizedVec.push_back(normalizedVal);
    }

    return normalizedVec;
}

std::vector<float> denormalizeVector(const std::vector<float>& vec) {
    std::vector<float> denormalizedVec;
    denormalizedVec.reserve(vec.size());

    for (const float& val : vec) {
        auto denormalizedVal = (float)(val * 255);
        denormalizedVec.push_back(denormalizedVal);
    }

    return denormalizedVec;
}

int main() {
    using namespace cv;
    using microgradpp::Tensor;
    using microgradpp::Value;
    using microgradpp::algorithms::MLP;

    Mat img = imread("./german_shephard.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Could not read the image" << std::endl;
        return 1;
    }

    // Resize the image to new dimensions
    int newWidth = 100;
    int newHeight = 100;
    cv::Mat resizedImage = resizeImage(img, newWidth, newHeight);
    resizedImage.convertTo(resizedImage, CV_8UC3);

//    cv::imshow("True Pixels", resizedImage);
//    waitKey(100);
    // Flatten the matrix into a single row
    cv::Mat flatMat = resizedImage.reshape(1, 1);
    std::vector<float> vec(flatMat.begin<uint8_t>(), flatMat.end<uint8_t>());


    auto normVec = normalizeVector(vec);
    Tensor input(normVec);
    Tensor output(normVec);

    //std::vector<float> baseline(vec.begin()+8080, vec.begin()+8090);

    // Initialize MLP
    constexpr float learningRate =  0.1;
    auto mlp = std::make_unique<MLP>(newWidth * newHeight ,std::vector<size_t> {4,static_cast<size_t>(newWidth * newHeight )}, learningRate);

//    std::shared_ptr<Value> loss;
    std::vector<float> pixels;

    microgradpp::loss::MeanSquaredErrorForPixels lossFcn;
    Tensor ypred;


    for (auto idx = 0; idx < 500; ++idx) {
        __MICROGRADPP_CLEAR__
        //std::cout << "++++++++++++" << std::endl;

        input.zeroGrad();
        //pixels.clear();
        // Predict values
        //std::cout << "Input data is: " << input << "\n";
        for (const auto& inp : input) {
            ypred.push_back(mlp->forward(inp));
        }

        // std::cout << ypred << std::endl;

//        for (const auto& y : ypred) {
//            for (const auto& val : y) {
//                pixels.push_back(static_cast<float>(val->data));
//            }
//        }

//        auto p = denormalizeVector(pixels);
//
//        cv::Mat predictedMatDuringIter(p, false);  //vectorToMat(p, newHeight, newWidth,1);
//        predictedMatDuringIter.convertTo(predictedMatDuringIter, CV_8UC3);
//        predictedMatDuringIter = predictedMatDuringIter.reshape(1,newWidth);
//        std::string label = "Iteration: " + std::to_string(idx);
//        cv::putText(predictedMatDuringIter, label, Point(50, 190), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
//        cv::imshow("Predicted True Pixels", predictedMatDuringIter);
//        waitKey(50);

        // Backpropagation
        // Ensure all gradients are zero
        //mlp->zeroGrad();
//        //auto groundTruth =
        //auto loss = lossFcn(output, ypred);
//        //auto groundTruth =
        auto loss = Value::create(0.0);
        const size_t  maxSize = ypred[0].size();
        for (size_t i = 0; i < maxSize; ++i) {
            auto v = Value::multiply(Value::subtract(output.at(0,i) , ypred.at(0,i)),Value::subtract(output.at(0,i) , ypred.at(0,i)));
            loss = Value::add(loss, Value::divide(v,1000));
        }
//        // Ensure all gradients are zero
        mlp->zeroGrad();

        // Perform backprop
        loss->_backward();

        // Update parameters
        mlp->update();

        ypred.reset();

        //mlp->printParameters();

        std::cout << idx << " " << loss->data << std::endl;
    }

    return 0;
}
