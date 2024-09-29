////
//// Created by Gautam Sharma on 7/7/24.
////
//
//
//#include <iostream>
//#include "Value.hpp"
//#include "Layer.hpp"
//#include "Tensor.hpp"
//
//// This example needs openCV to compile
//
//// For visualization
//#include <opencv2/opencv.hpp>
//
//cv::Mat vectorToMat(const std::vector<float>& vec, int rows, int cols, int channels) {
//    if (vec.size() != rows * cols * channels) {
//        std::cerr << "Size of vector does not match given dimensions" << std::endl;
//        return cv::Mat();
//    }
//
//    // Create a Mat with the appropriate type and dimensions
//    cv::Mat mat(rows, cols, CV_32FC3);
//
//    // Copy data from the vector to the Mat
//    std::memcpy(mat.data, vec.data(), vec.size() * sizeof(float));
//
//    return mat;
//}
//// Function to resize an image
//cv::Mat resizeImage(const cv::Mat& image, int newWidth, int newHeight) {
//    cv::Mat resizedImage;
//    cv::resize(image, resizedImage, cv::Size(newWidth, newHeight));
//    return resizedImage;
//}
//
//std::vector<float> normalizeVector(const std::vector<float>& vec) {
//    auto minmax = std::minmax_element(vec.begin(), vec.end());
//    float minVal = *minmax.first;
//    float maxVal = *minmax.second;
//    float range = maxVal - minVal;
//
//    std::vector<float> normalizedVec;
//    normalizedVec.reserve(vec.size());
//
//    for (const float& val : vec) {
//        float normalizedVal = val/255;// 2 * ((val - minVal) / range) - 1;
//        normalizedVec.push_back(normalizedVal);
//    }
//
//    return normalizedVec;
//}
//
//std::vector<float> denormalizeVector(const std::vector<float>& vec) {
//    std::vector<float> denormalizedVec;
//    denormalizedVec.reserve(vec.size());
//
//    for (const float& val : vec) {
//        float denormalizedVal = (int)(val * 255);
//        denormalizedVec.push_back(denormalizedVal);
//    }
//
//    return denormalizedVec;
//}
//
//int main() {
//    using namespace cv;
//    using microgradpp::Tensor;
//    using microgradpp::Value;
//
//    Mat img = imread("../public/german_shephard.jpg", IMREAD_GRAYSCALE);
//    if (img.empty()) {
//        std::cerr << "Could not read the image" << std::endl;
//        return 1;
//    }
//
//    // Resize the image to new dimensions
//    int newWidth = 200;
//    int newHeight = 200;
//    cv::Mat resizedImage = resizeImage(img, newWidth, newHeight);
//    resizedImage.convertTo(resizedImage, CV_8UC3);
//
//    cv::imshow("True Pixels", resizedImage);
//    waitKey(100);
//    // Flatten the matrix into a single row
//    cv::Mat flatMat = resizedImage.reshape(1, 1);
//    std::vector<float> vec(flatMat.begin<uint8_t>(), flatMat.end<uint8_t>());
//
//
//    auto normVec = normalizeVector(vec);
//    Tensor input(normVec);
//    Tensor output(normVec);
//
//    std::vector<float> baseline(vec.begin()+8080, vec.begin()+8090);
//
//    // Initialize MLP
//    constexpr float learningRate = 0.00001;
//    auto mlp = microgradpp::MLP(newWidth * newHeight ,{4,4,static_cast<size_t>(newWidth * newHeight )}, learningRate);
//
//    std::shared_ptr<Value> loss;
//    std::vector<float> pixels;
//    for (auto idx = 0; idx < 1000000; ++idx) {
//        Tensor ypred;
//        input.zeroGrad();
//        pixels.clear();
//        // Predict values
//        for (const auto& inp : input) {
//            ypred.push_back(mlp(inp));
//        }
//
//        for (const auto& y : ypred) {
//            for (const auto& val : y) {
//                pixels.push_back(static_cast<float>(val->data));
//            }
//        }
//
//        auto p = denormalizeVector(pixels);
//        std::vector<float> p1(p.begin()+8080, p.begin()+8090);
////        std::cout << "Input: "<< input ;
//        std::cout << "Input: ";
//        for(auto&g : baseline){
//            std::cout << g << " ";
//        }
//
//        std::cout << std::endl;
//        std::cout << "Pred: ";
//        for(auto&g : p1){
//            std::cout << g << " ";
//        }
//
//        cv::Mat predictedMatDuringIter(p, false);  //vectorToMat(p, newHeight, newWidth,1);
//        predictedMatDuringIter.convertTo(predictedMatDuringIter, CV_8UC3);
//        predictedMatDuringIter = predictedMatDuringIter.reshape(1,newWidth);
//        std::string label = "Iteration: " + std::to_string(idx);
//        cv::putText(predictedMatDuringIter, label, Point(50, 190), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
//        cv::imshow("Predicted True Pixels", predictedMatDuringIter);
//        waitKey(50);
//        // Backpropagation
//        mlp.zeroGrad();
//
//        // Calculate loss
//        loss = Value::create(0.0);
//        //auto groundTruth =
//        const size_t  maxSize = ypred[0].size();
//        for (size_t i = 0; i < maxSize; ++i) {
//            //auto v = ypred.at(i)*255;
//            loss += (output.at(0,i)*255 - ypred.at(0,i)*255)^2;
//        }
//
//        loss->backProp();
//        mlp.update();
//
//        std::cout << idx << " " << loss->data << std::endl;
//    }
//
//    return 0;
//}
int main(){
    return 0;
}