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

void add_color_mask(const cv::Mat &src, const cv::Mat &cls_mask,  cv::Mat &dst, cv::Scalar color)
{
  cv::Mat src_copy = src.clone();
  for(int w = 0; w < cls_mask.cols; w++)
  for(int h = 0; h < cls_mask.rows; h++)
  {   
      int cls = cls_mask.at<float>(cv::Point(w, h));
      if(cls>0)
      {
        src_copy.at<cv::Vec3b>(cv::Point(w, h)) = cv::Vec3b(color[0], color[1], color[2]);
      }
  }
  //cv::imwrite("color_mask.jpg", src_copy);
  //cv::imwrite("src.jpg",src);
  cv::addWeighted(src, 0.3, src_copy, 0.7, 0, dst);
}

void TestMaskRcnn(int argc, char **argv){
    string model_path = "/home/chengli12/git/mmdetection/work_dirs/maskrcnn";
    //string img_path = "/data/ultron/dataset/coco/val2017/000000397133.jpg";
    string img_path = "/home/chengli12/data/anqing.png";
    cv::Mat im = cv::imread(img_path);
    cv::Mat frame = im.clone();
    int device_id = 1;
    int img_height = frame.rows;
    int img_width = frame.cols;
    unique_ptr<ObjectDetector> maskrcnn = DetectorCreator::create_detector(DetectorType::MaskRcnn, device_id);
    maskrcnn->Init(model_path);

    auto font_face = CV_FONT_HERSHEY_SIMPLEX;
    auto font_scale = 1.2;
    auto thickness = 2;
    auto frame_count = 0;
    auto det_time = 0.0;
    
    int step = 100;
    vector<DetectedObject> detected_objects;
    for(int ii =0; ii < step; ii++) {
        maskrcnn->Detect(im, detected_objects);
        cout<<"Detect "<<ii<<endl;
    }
    int i = 0;
    for (const auto object : detected_objects) {
        auto x1 = static_cast<int>(std::round(object.x1() * img_width));
        auto y1 = static_cast<int>(std::round(object.y1() * img_height));
        auto w = static_cast<int>(std::round(object.w * img_width));
        auto h = static_cast<int>(std::round(object.h * img_height));
        cv::Rect bbox(x1, y1, w, h);
        //cout<<bbox<<endl;
        rectangle(frame, bbox, randomColors[object.id], thickness);
        cv::Mat roi = frame(bbox);
        cv::Mat roi_show;
        cv::Size box_size(static_cast<int>(w), static_cast<int>(h));
        cv::Mat resize_mask;
        cv::resize(object.mask, resize_mask, box_size);
        cv::Mat final_mask;
        cv::threshold(resize_mask, final_mask, 0.5, 255, cv::THRESH_BINARY);
        // string mask_name = string("mask")+to_string(i++)+string(".jpg");
        // cv::imwrite(mask_name, final_mask);
        add_color_mask(roi, final_mask, roi_show, randomColors[object.id]);
        roi_show.copyTo(roi);
        std::stringstream ss;
        ss << object.id << ": " << std::round(object.score * 1000) / 10.0 << "%";
        putText(frame, ss.str(), cv::Point(bbox.x, bbox.y - 10), font_face, font_scale,
                randomColors[object.id], thickness, 8);
    }
    cv::imwrite("res.jpg", frame);
    cout<<"detect done"<<endl;
}

int main(int argc, char **argv)
{
    get_random_colors();
    TestMaskRcnn(argc, argv);
}

