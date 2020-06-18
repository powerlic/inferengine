#include "infer_creator.hpp"
#include "ssd.h"
#include "classifier.h"
#include "fasterrcnn.h"
#include "maskrcnn.h"
#include "fasterrcnnfpn.h"
#include <ATen/cuda/CUDAContext.h>


std::unique_ptr<ObjectDetector> DetectorCreator::create_detector(DetectorType detector_type, int device_id)
{
    cudaSetDevice(device_id);
    cv::cuda::setDevice(device_id);
    cout<<"using device id "<<device_id<<endl;
    switch (detector_type)
    {
        case DetectorType::SSD:
            return std::unique_ptr<ObjectDetector>(new SSD(device_id));
        case DetectorType::FasterRcnn:
            return std::unique_ptr<ObjectDetector>(new FasterRcnn(device_id));
        case DetectorType::MaskRcnn:
            return std::unique_ptr<ObjectDetector>(new MaskRcnn(device_id));
        case DetectorType::FasterRcnnFpn:
            cout<<"create FasterRcnnFpn"<<endl;
            return std::unique_ptr<ObjectDetector>(new FasterRcnnFpn(device_id));
        default:
            std::cout<<"Only SSD FasterRcnn MaskRcnn is supported until now"<<endl;
            break;
    }
    return nullptr;
}

std::unique_ptr<Classifier> ClassifierCreator::create_classifier(Engine backend_type, int device_id)
{
    switch(backend_type)
    {
        case Engine::Pytorch:
            return std::unique_ptr<Classifier>(new TorchClassifier(device_id));
        default:
            cout<<"Only caffe classifier is supported until now"<<endl;
            break;
    }
    return nullptr;
}