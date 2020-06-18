#ifndef _DETECTOR_API_
#define _DETECTOR_API_
#include "detector_types.hpp"
#include "video_tracker.hpp"

class DetectorImpl;
class Detector {
public:
    explicit Detector();
    ~Detector();
    long Init(const DetectorParams params);
    long Detect(const cv::Mat &src, vector<Object> &detect_objects);
    long DetectWithTracking(const cv::Mat &src, int64_t frame_key, std::shared_ptr<Tracker> tracker, vector<ObjectResult>& object_results);

    static string version();

private:
    std::unique_ptr<DetectorImpl> impl;
};

#endif