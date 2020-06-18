#include "infer.h"
#include "infer_creator.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <torch/script.h> // One-stop header.
#include <cuda_runtime.h>



using namespace std;

void TestClassifier(int argc, char **argv){
    string model_path = "/home/chengli12/AlgoPro/ultron_cls_train/outputresnet18_model_best.pt";
    //string model_path = "/home/chengli12/git/mmdetection/work_dirs/maskrcnn/mask_rcnn.pt";
    string img_path = "/home/chengli12/AlgoPro/ultron_cls_train/data/lily/lily001.jpg";
    cv::Mat im = cv::imread(img_path);
    unique_ptr<Classifier> classifier = ClassifierCreator::create_classifier(Engine::Pytorch, 0);
    classifier->Init(model_path);
    vector<vector<float>> probs;
    vector<cv::Mat> frames;
    frames.push_back(im);
    int step = 10;
    for(int ii =0; ii < step; ii++) {
        classifier->Classify(frames, probs);
        for(auto i=0;i<probs[0].size();i++){
            cout<<probs[0][i]<<" ";
        }
        cout<<endl;
    }
}

void TestDemo(){

    torch::Device device(torch::kCUDA, 1);
    torch::jit::script::Module module = torch::jit::load("/home/chengli12/AlgoPro/ultron_cls_train/outputresnet18_model_best.pt");
    module.to(device, true);
    std::cout<<"load model done"<<std::endl;
    string img_path = "/home/chengli12/AlgoPro/ultron_cls_train/data/lily/lily001.jpg";
    //输入图像
    auto image = cv::imread(img_path, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat image_transfomed;
    cv::resize(image, image_transfomed, cv::Size(224, 224));
    cv::cvtColor(image_transfomed, image_transfomed, cv::COLOR_BGR2RGB);

    // 图像转换为Tensor
    torch::Tensor tensor_image = torch::from_blob(image_transfomed.data, {image_transfomed.rows, image_transfomed.cols,3},torch::kByte);
    tensor_image = tensor_image.permute({2,0,1});
    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);
    tensor_image = tensor_image.unsqueeze(0);
    tensor_image = tensor_image.to(device);
    // 网络前向计算
    // Execute the model and turn its output into a tensor.
    std::cout<<"predict process"<<std::endl;
    at::Tensor output = module.forward({tensor_image}).toTensor();
    std::cout<<"predict done"<<std::endl;
    auto max_result = output.max(1,true);
    auto max_index = std::get<1>(max_result).item<float>();    
}

int test_read() 
{
    //Open image file to read from
    char imgPath[] = "/home/chengli12/data/mini_car.jpg";
    ifstream fileImg(imgPath, ios::binary);
    fileImg.seekg(0, std::ios::end);
    int bufferLength = fileImg.tellg();
    fileImg.seekg(0, std::ios::beg);

    if (fileImg.fail())
    {
        cout << "Failed to read image" << endl;
        cin.get();
        return -1;
    }

    //Read image data into char array
    char *buffer = new char[bufferLength];
    fileImg.read(buffer, bufferLength);

    //Decode data into Mat 
    cv::Mat matImg;
    matImg = cv::imdecode(cv::Mat(1, bufferLength, CV_8UC1, buffer), CV_LOAD_IMAGE_UNCHANGED);

    //Create Window and display it
    cv::imwrite("./test.jpg", matImg);

    delete[] buffer;

    return 0;
}
int main(int argc, char **argv)
{
    //test_read();
    TestClassifier(argc, argv);
    std::cout<<"done"<<std::endl;
    return 0;
}