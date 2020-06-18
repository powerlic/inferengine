#ifndef _INFER_CREATOR_HPP_
#define _INFER_CREATOR_HPP_
#include<memory>


using namespace std;



enum class DetectorType : int
{
    SSD = 0,
    FasterRcnn = 1,
    MaskRcnn = 2,
    FasterRcnnFpn = 3
};

enum class Engine : int
{
    Caffe,
    Tensorflow,
    Pytorch,
    Mxnet,
};


class Classifier;

class ClassifierCreator final
{
public:
    ClassifierCreator() = default;
    ~ClassifierCreator()= default;
    static unique_ptr<Classifier> create_classifier(Engine backend_type, int device_id);
};


class ObjectDetector;

class DetectorCreator final
{
public:
    DetectorCreator() = default;
    ~DetectorCreator()= default;
    static unique_ptr<ObjectDetector> create_detector(DetectorType detector_type, int device_id);
};




#endif