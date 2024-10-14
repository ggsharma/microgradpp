//
// Created by Gautam Sharma on 7/7/24.
//

/**
 * @file Main.cpp
 * @brief An example demonstrating how to use the Micrograd++ library for an image processing
 *        application. The code utilizes an MLP model for pixel-wise predictions, visualized with OpenCV.
 *
 * @details This program loads an image, resizes it, and uses a multi-layer perceptron (MLP) model
 *          to predict pixel intensities for the grayscale image. It displays the input image
 *          and visualizes the model's output in real-time.
 *          **Note:** OpenCV is required for compilation and visualization.
 */

#include <iostream>
#include "Value.hpp"
#include "Layer.hpp"
#include "Tensor.hpp"
#include "Algorithms.hpp"
#include "TypeDefs.hpp"
#include "Autograd.hpp"
#include "LossFunctions.hpp"


#include "base/BaseMultiLayerPerceptron.hpp"
#include "nn/NeuralNet.hpp"
#include "core/Sequential.hpp"
#include "core/MppCore.hpp"
#include "LossFunctions.hpp"

// Visualization library
#include <opencv2/opencv.hpp>

/**
 * @brief Converts a 1D vector into an OpenCV Mat.
 * @param vec The 1D vector containing pixel values.
 * @param rows Number of rows in the target matrix.
 * @param cols Number of columns in the target matrix.
 * @param channels Number of color channels.
 * @return An OpenCV Mat representation of the vector if sizes match; an empty Mat otherwise.
 */
cv::Mat vectorToMat(const std::vector<float>& vec, int rows, int cols, int channels) {
    if (vec.size() != rows * cols * channels) {
        std::cerr << "Size of vector does not match given dimensions" << std::endl;
        return {};
    }

    cv::Mat mat(rows, cols, CV_32FC3);
    std::memcpy(mat.data, vec.data(), vec.size() * sizeof(float));
    return mat;
}

/**
 * @brief Resizes an input image to specified dimensions.
 * @param image The input image.
 * @param newWidth The desired width.
 * @param newHeight The desired height.
 * @return The resized image as an OpenCV Mat.
 */
cv::Mat resizeImage(const cv::Mat& image, int newWidth, int newHeight) {
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(newWidth, newHeight));
    return resizedImage;
}

/**
 * @brief Normalizes a vector's values to a 0-1 range for neural network input.
 * @param vec The input vector with original pixel intensities (0-255).
 * @return A normalized vector with values between 0 and 1.
 */
std::vector<float> normalizeVector(const std::vector<float>& vec) {
    auto minmax = std::minmax_element(vec.begin(), vec.end());
    float minVal = *minmax.first;
    float maxVal = *minmax.second;
    float range = maxVal - minVal;

    std::vector<float> normalizedVec;
    normalizedVec.reserve(vec.size());

    for (const float& val : vec) {
        float normalizedVal = val / 255;  // Normalize to 0-1
        normalizedVec.push_back(normalizedVal);
    }
    return normalizedVec;
}

/**
 * @brief Converts a normalized vector back to original pixel intensity range (0-255).
 * @param vec The normalized vector (0-1).
 * @return A denormalized vector with values scaled back to 0-255.
 */
std::vector<float> denormalizeVector(const std::vector<float>& vec) {
    std::vector<float> denormalizedVec;
    denormalizedVec.reserve(vec.size());

    for (const float& val : vec) {
        auto denormalizedVal = static_cast<float>(val * 255);
        denormalizedVec.push_back(denormalizedVal);
    }
    return denormalizedVec;
}

namespace microgradpp{

    using microgradpp::base::BaseMultiLayerPerceptron;
    using microgradpp::core::Sequential;
    using microgradpp::core::MppCore;
    using namespace microgradpp::nn;

    class Example_Images : public BaseMultiLayerPerceptron{
    public:
        size_t width, height;
        Example_Images(size_t width, size_t height):width(width), height(height),
                BaseMultiLayerPerceptron(Sequential(
                        {
                                nn::Linear(width*height,4),
                                nn::TanH(),
                                nn::Linear(4,width*height)
                        }))
        {

            this->learningRate = 0.00001;
        }


        Tensor1D forward(Tensor1D input) override{
            return this->sequential(input);
        };

    };

}


/**
 * @brief Main function for loading and preprocessing an image, feeding it to an MLP, and visualizing the output.
 *
 * @details This function reads an image, resizes it, normalizes the pixel values, and initializes an MLP model to predict pixel values.
 *          For each iteration, it displays the predicted image with updated pixel values.
 *          The loss between the model prediction and actual pixel values is calculated and backpropagated through the MLP for optimization.
 *
 * @return 0 if successful, 1 if an error occurs in reading the image.
 */
int main() {
    using namespace cv;
    using microgradpp::Tensor1D;
    using microgradpp::Value;
    using microgradpp::Example_Images;
    using microgradpp::algorithms::MLP;
    using microgradpp::Tensor2D;
    using microgradpp::loss::MeanSquaredErrorFor1DPixels;

    // Load and preprocess image
    Mat img = imread("./german_shephard.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Could not read the image" << std::endl;
        return 1;
    }

    int newWidth = 50, newHeight = 50;
    cv::Mat resizedImage = resizeImage(img, newWidth, newHeight);
    resizedImage.convertTo(resizedImage, CV_8UC3);

    // Display input image
    cv::imshow("True Pixels", resizedImage);
    waitKey(100);

    cv::Mat flatMat = resizedImage.reshape(1, 1);  // Flatten image
    std::vector<float> vec(flatMat.begin<uint8_t>(), flatMat.end<uint8_t>());

    auto normVec = normalizeVector(vec);
    Tensor2D input = normVec;
    Tensor2D output = normVec;

    // MLP initialization
    auto mlp = std::make_unique<Example_Images>(newWidth , newHeight);

    std::vector<float> predictionPixels;
    microgradpp::loss::MeanSquaredErrorFor1DPixels lossFcn;
    Tensor2D ypred;

    for (auto idx = 0; idx < 50000; ++idx) {
        __MICROGRADPP_CLEAR__

        input.zeroGrad();
        predictionPixels.clear();

        for (const auto& inp : input) {
            ypred.push_back( mlp->operator()(inp) );  // Predict pixel values with MLP
        }

        for (const auto& y : ypred) {
            for (const auto& val : y) {
                predictionPixels.push_back(static_cast<float>(val->data));
            }
        }

        auto denormalizedPixels = denormalizeVector(predictionPixels);
        cv::Mat predictedMatDuringIter(denormalizedPixels, false);
        predictedMatDuringIter.convertTo(predictedMatDuringIter, CV_8UC3);
        predictedMatDuringIter = predictedMatDuringIter.reshape(1, newWidth);

        std::string label = "Iteration: " + std::to_string(idx);
        //cv::putText(predictedMatDuringIter, label, Point(0, 50), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 2.0);
        cv::imshow("Predicted True Pixels", predictedMatDuringIter);
        waitKey(50);

        // Loss calculation
        auto loss = lossFcn(output, ypred);

        mlp->zeroGrad();      // Zero gradients before backprop
        loss->backProp();     // Backpropagate loss
        mlp->update();       // Update MLP weights

        ypred.reset();
        std::cout << idx << " " << loss->data << std::endl;
    }

    return 0;
}
