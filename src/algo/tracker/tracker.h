#ifndef TRACKING_TRACKER_H
#define TRACKING_TRACKER_H
#include <iostream>
#include <set>
#include <atomic>
#include <memory>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/eigen.hpp>


#include "detector_types.hpp"
#include "kcf.hpp"

using namespace std;

using GpuMat = cv::cuda::GpuMat;


class Tracker{

public:
  Tracker();
  Tracker(int camera_id, int detect_every_n);
  ~Tracker();
  vector<ObjectResult> Predict(const cv::Mat &frame);
  vector<ObjectResult> Update(const cv::Mat &frame, vector<ObjectResult> &obj_results);
  long GetFrameCount();
  void IncreaseFrameCount();
  void SetCurrentTrackedObjects(vector<ObjectResult> &obj_results);
  vector<ObjectResult> GetCurrentTrackedObjects();
  long Release();
  int camera_id;
  int detect_every_n;
  
private:
  float max_iou_distance = 0.7;
  static atomic<long int> track_id;
  float scale_;
  long frame_count;
  vector<KCF> kcf_trackers;
  vector<ObjectResult> current_tracked_objects;
  cv::Mat frame_;
  
  void IoU_cost(vector<cv::Rect> &tracks, vector<cv::Rect> &detections, cv::Mat &cost_matrix);
  void IoU_matching(vector<cv::Rect> &tracks, vector<cv::Rect> &detections, vector<int> &matched_tracks, vector<int> &matched_detections, vector<int> &unmatched_tracks, vector<int> &unmatched_detections);
  float iou(cv::Rect &r1, cv::Rect &r2);
  cv::Rect FloorRectScale(cv::Rect &r, double scale_factor);

  //std::unique_ptr<GPUAllocator> allocator_;
  GpuMat gpu_frame_;
  GpuMat gpu_resized_;
};
#endif
