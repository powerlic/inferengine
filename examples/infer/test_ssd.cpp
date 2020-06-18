#include "infer.h"
#include "infer_creator.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace std;


std::vector<cv::Scalar> randomColors;

void get_random_colors()
{
    randomColors.resize(256 * 256 * 256);
    for (int i = 0; i < randomColors.size(); ++i)
    {
        randomColors[i] = cv::Scalar(i / 256 / 256, (i / 256) % 256, i % 256);
    }
    for (int i = randomColors.size(); i > 0; --i)
    {
        int p = std::rand() % i;
        auto tmp = randomColors[i];
        randomColors[i] = randomColors[p];
        randomColors[p] = tmp;
    }
}

void TestSSD(int argc, char **argv){
    string model_path = "/home/chengli12/git/mmdetection/work_dirs/ssd512_coco/ssd512_coco.pt";
    //string img_path = "/data/ultron/dataset/coco/val2017/000000397133.jpg";
    string img_path = "/home/chengli12/data/anqing.png";
    cv::Mat im = cv::imread(img_path);
    cout<<"load frame done"<<endl;
    cv::Mat frame = im.clone();
    int device_id = 1;
    int img_height = frame.rows;
    int img_width = frame.cols;
    unique_ptr<ObjectDetector> ssd = DetectorCreator::create_detector(DetectorType::SSD, device_id);
    ssd->Init(model_path);

    auto font_face = CV_FONT_HERSHEY_SIMPLEX;
    auto font_scale = 1.2;
    auto thickness = 2;
    auto frame_count = 0;
    auto det_time = 0.0;

    int step = 100;
    vector<DetectedObject> detected_objects;
    for(int ii =0; ii < step; ii++) {
        detected_objects.clear();
        ssd->Detect(im, detected_objects);
        cout<<"Detect "<<ii<<endl;
    }
    int i = 0;
    for (const auto object : detected_objects) {
        auto x1 = static_cast<int>(std::round(object.x1() * img_width));
        auto y1 = static_cast<int>(std::round(object.y1() * img_height));
        auto w = static_cast<int>(std::round(object.w * img_width));
        auto h = static_cast<int>(std::round(object.h * img_height));
        cv::Rect bbox(x1, y1, w, h);
        rectangle(frame, bbox, randomColors[object.id], thickness);
        std::stringstream ss;
        ss << object.id << ": " << std::round(object.score * 1000) / 10.0 << "%";
        putText(frame, ss.str(), cv::Point(bbox.x, bbox.y - 10), font_face, font_scale,
                randomColors[object.id], thickness, 8);
    }
    cv::imwrite("ssd.jpg", frame);
    cout<<"detect done"<<endl;
}

int main(int argc, char **argv)
{
    get_random_colors();
    TestSSD(argc, argv);
}


