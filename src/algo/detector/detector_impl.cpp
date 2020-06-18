#include "detector_impl.hpp"
#include "utils/logging.hpp"
#include "utils/timer.hpp"

using namespace easycv;
using namespace std;

DetectorImpl::DetectorImpl() 
{

}

long DetectorImpl::Init(const DetectorParams params) 
{
    params_ = params;
    string type_str = params_.model_type;
    DetectorType type;
    if(type_str.compare("SSD")==0)
    {
        type = DetectorType::SSD;
    }
    else if(type_str.compare("FasterRcnn")==0)
    {
        type = DetectorType::FasterRcnn;
    }
    else if(type_str.compare("FasterRcnnFpn")==0)
    {
        type = DetectorType::FasterRcnnFpn;
    }
    else if(type_str.compare("MaskRcnn")==0)
    {
        type = DetectorType::MaskRcnn;
        cout<<"use maskrcnn "<<endl;
    }
    else{
        LOG(ERROR)<<"Not supported model "<<type_str;
        return -1L;
    }
    LOG(INFO)<<"using device "<<params_.device_id;
    detector_ = DetectorCreator::create_detector(type, params_.device_id);
    //cout<<detector_<<endl;
    detector_->Init(params_.model_path);
    if(detector_){
        return 0L;
    }
    return -1L;
}

long DetectorImpl::Detect(const cv::Mat &src, vector<Object> &objects)
{
    vector<DetectedObject> detected_objects;
    detector_->Detect(src, detected_objects);
    for(auto i=0;i<detected_objects.size();i++){
        Object obj;
        obj.confidence = detected_objects[i].score;
        obj.type = detected_objects[i].id;
        obj.mask = detected_objects[i].mask.clone();
        float x1, y1, x2, y2;
        x1 = detected_objects[i].x1();
        y1 = detected_objects[i].y1();
        x2 = detected_objects[i].x2();
        y2 = detected_objects[i].y2();
        obj.norm_rect = cv::Rect2f(x1, y1, x2-x1, y2-y1);
        obj.rect = cv::Rect(x1*src.cols, y1*src.rows, (x2-x1)*src.cols, (y2-y1)*src.rows);
        obj.object_crop = src(obj.rect).clone();
        objects.push_back(obj);
    }
}

long DetectorImpl::DetectWithTracking(const cv::Mat &src, int64_t frame_key, std::shared_ptr<Tracker> tracker, vector<ObjectResult>& object_results){
    object_results.clear();
    Timer detect_timer, tracking_timer;
    if (tracker->GetFrameCount() % tracker->detect_every_n == 0) {
        vector<Object> detected_objects;
        //detect_timer.tic();
        Detect(src, detected_objects);
        //LOG(INFO)<<"detect obj "<<detected_objects.size();
        //LOG(INFO)<<"detect obj done "<<detected_objects.size();
        vector<ObjectResult> detections;
        for (size_t i = 0; i < detected_objects.size(); i++) {
            ObjectResult object_result;
            object_result.detected_object = detected_objects[i];
            object_result.rect = detected_objects[i].rect;
            object_result.norm_rect = detected_objects[i].norm_rect;
            detections.push_back(object_result);
        }
        //LOG(INFO)<<"detections size "<<detections.size();
        //LOG(INFO)<<"detections size "<<detections.size();
        //tracking_timer.tic();
        object_results = tracker->Update(src, detections);
        //tracking_timer.toc("Tracking update");
        //LOG(INFO)<<"Update Done";
    }
    else {
        //tracking_timer.tic();
        object_results = tracker->Predict(src);
        //tracking_timer.toc("Tracking predict");
        //LOG(INFO)<<"Predict Done"<<endl;
    }

    for (size_t i = 0; i < object_results.size(); i++) {
        object_results[i].is_updated = 0;
        object_results[i].source_id = tracker->camera_id;
        object_results[i].frame_key = frame_key;
    }
    tracker->SetCurrentTrackedObjects(object_results);
    return 0L;
}
