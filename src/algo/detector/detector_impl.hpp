#ifndef _DETECTOR_IMPL_HPP_
#define _DETECTOR_IMPL_HPP_


#include <memory>
#include <map>
#include <vector>
#include <mutex>
#include <string>
#include <opencv2/opencv.hpp>
#include "detector_types.hpp"
#include "detector.hpp"
#include "tracker/tracker.h"


#include "infer/infer.h"
#include "infer/infer_creator.hpp"

class DetectorImpl {
public:
    explicit DetectorImpl();
    ~DetectorImpl()=default;
    long Init(const DetectorParams params);
    long Detect(const cv::Mat &src, vector<Object> &detect_objects);
    long DetectWithTracking(const cv::Mat &src, int64_t frame_key, std::shared_ptr<Tracker> tracker, vector<ObjectResult>& object_results);
private:
    std::unique_ptr<ObjectDetector> detector_;
    DetectorParams params_;

};

#endif //_DETECTOR_IMPL_HPP_

