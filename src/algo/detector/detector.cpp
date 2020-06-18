#include "detector.hpp"
#include "detector_impl.hpp"
#include "detector_version.hpp"



string Detector::version()
{
    std::stringstream ss;
    ss << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH << "." << GIT_COMMIT_HASH;
    ss << "(" << BUILD_DATE << ")";
    std::string version_str;
    ss >> version_str; 
    std::cout << "version: " << version_str <<std::endl;
    return version_str;
}

Detector::Detector(){
    impl.reset(new DetectorImpl());
}

Detector::~Detector(){
}

long Detector::Init(const DetectorParams params){
    return impl->Init(params);
}

long Detector::Detect(const cv::Mat &src, vector<Object> &objects){
    return impl->Detect(src, objects);
}

long Detector::DetectWithTracking(const cv::Mat &src, int64_t frame_key, std::shared_ptr<Tracker> tracker, vector<ObjectResult>& object_results)
{
    return impl->DetectWithTracking(src, frame_key, tracker, object_results);
}

