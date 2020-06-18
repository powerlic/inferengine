#include "tracker.h"
#include "Hungarian.h"
#include <limits>
#include "utils/logging.hpp"

#include "utils/timer.hpp"

#include "tbb/tbb.h"
#include "tbb/concurrent_vector.h"
#include "tbb/tick_count.h"
#include "tbb/parallel_for.h"

using namespace easycv;

atomic<long int> Tracker::track_id(0);

Tracker::Tracker()
{
  max_iou_distance = 0.7;
  frame_count = 0;
  detect_every_n = 5;
  //allocator_.reset(new GPUAllocator(1024 * 1024 * 128));
}

Tracker::~Tracker()
{
  Release();
}

Tracker::Tracker(int camera_id, int detect_every_n)
{
  max_iou_distance = 0.7;
  // track_id = 0;
  frame_count = 0;
  this->detect_every_n = detect_every_n;
  this->camera_id = camera_id;
}

void Tracker::IncreaseFrameCount()
{
  frame_count +=  1;
  if(frame_count >= std::numeric_limits<long>::max())
  {
    frame_count = 0;
  }
}

long Tracker::GetFrameCount()
{
  return frame_count;
}

long Tracker::Release()
{
  return 0;
}

void Tracker::SetCurrentTrackedObjects(vector<ObjectResult> &obj_results){
  current_tracked_objects = obj_results;
}

vector<ObjectResult> Tracker::GetCurrentTrackedObjects()
{
  return current_tracked_objects;
}

vector<ObjectResult> Tracker::Predict(const cv::Mat &frame)
{
  IncreaseFrameCount();
  if(current_tracked_objects.size() == 0 || kcf_trackers.size()==0)
  {
    return current_tracked_objects;
  }
  scale_ = 1.0;
  if(frame.rows > 360)
  {
    scale_ = 360.0f / frame.rows;
  }
  if(scale_ > 0.9999 && scale_ < 1.0001 )
  {
    frame_ = frame;
  }
  else
  {
    //gpu_frame_.upload(frame);
    //cv::cuda::resize(gpu_frame_, gpu_resized_, cv::Size(0, 0), scale_, scale_, cv::INTER_LINEAR);
    cv::resize(frame, frame_, cv::Size(0, 0), scale_, scale_, cv::INTER_LINEAR);
    //gpu_resized_.download(frame_);


  }

  //LOG(INFO)<<"kcf_trackers size"<<kcf_trackers.size();
  // Timer kcf_trackers_timer;
  // kcf_trackers_timer.tic();

  vector<ObjectResult> current_tracked_objects_update;
  vector<KCF> kcf_trackers_update;
 
 
  vector<cv::Rect> tracked_rects;
  vector<bool> iou_meets;
 
  tracked_rects.resize(kcf_trackers.size());
  iou_meets.resize(kcf_trackers.size());

  //#pragma omp parallel for
  // for(int i = 0; i < kcf_trackers.size(); i++)
  // {
  //   cv::Rect tracked_rect = kcf_trackers[i].Update(frame_);
  //   tracked_rect = FloorRectScale(tracked_rect, 1.0/scale_);
  //   float iou_val = iou(current_tracked_objects[i].rect, tracked_rect);
  //   iou_meets[i] = iou_val > 0.4;
  //   tracked_rects[i] = tracked_rect;
  //   //LOG(INFO)<<"threas num "<<omp_get_num_threads();
  // }

  static tbb::affinity_partitioner ap;
  tbb::parallel_for(tbb::blocked_range<size_t>(0, kcf_trackers.size()),
      [this, &iou_meets, &tracked_rects](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); i++) 
      {
        cv::Rect tracked_rect = kcf_trackers[i].Update(frame_);
        tracked_rect = FloorRectScale(tracked_rect, 1.0/scale_);
        float iou_val = iou(current_tracked_objects[i].rect, tracked_rect);
        iou_meets[i] = iou_val > 0.4;
        tracked_rects[i] = tracked_rect;
      }
      }, ap);

  for(size_t i = 0; i < kcf_trackers.size(); i++) {
      if (iou_meets[i]) {
          current_tracked_objects[i].rect = tracked_rects[i];
          current_tracked_objects[i].detected_status = 0;
          current_tracked_objects[i].track_times+=1;
          current_tracked_objects_update.push_back(current_tracked_objects[i]);
          kcf_trackers_update.push_back(kcf_trackers[i]);
      }
  }


  // for(size_t i = 0; i < kcf_trackers.size(); i++)
  // {
  //   cv::Rect tracked_rect = kcf_trackers[i].Update(frame_);
  //   tracked_rect = FloorRectScale(tracked_rect, 1.0/scale_);
  //   float iou_val = iou(current_tracked_objects[i].rect, tracked_rect);
  //   if(iou_val > 0.4)
  //   {
  //     current_tracked_objects[i].rect = tracked_rect;
  //     current_tracked_objects[i].detected_status = 0;
  //     current_tracked_objects_update.push_back(current_tracked_objects[i]);
  //     kcf_trackers_update.push_back(kcf_trackers[i]);
  //   }
  // }


  //kcf_trackers_timer.toc("kcf tracker");
  kcf_trackers = kcf_trackers_update;
  current_tracked_objects = current_tracked_objects_update;
  return current_tracked_objects;
}

vector<ObjectResult> Tracker::Update(const cv::Mat &frame, vector<ObjectResult> &obj_results)
{
  scale_ = 1.0;
  if(frame.rows > 360){
    scale_ = 360.0f / frame.rows;
  }
  //Timer resize_timer;
  //resize_timer.tic();
  if(scale_ > 0.9999 && scale_ < 1.0001 ){
    frame_ = frame;
  }
  else{
    //gpu_frame_.upload(frame);
    cv::cuda::resize(gpu_frame_, gpu_resized_, cv::Size(0, 0), scale_, scale_, cv::INTER_LINEAR);
    cv::resize(frame, frame_, cv::Size(0, 0), scale_, scale_, cv::INTER_LINEAR);
    //gpu_resized_.download(frame_);
  }
  //resize_timer.toc("Resize");

  vector<cv::Rect> tracks;
  for(size_t i = 0; i < kcf_trackers.size(); i++)
  {
    cv::Rect tracked_rect = current_tracked_objects[i].rect;
    tracked_rect = FloorRectScale(tracked_rect, scale_);
    tracks.push_back(tracked_rect);
  }

  vector<cv::Rect> detections;
  for(size_t i = 0; i < obj_results.size(); i++){
    cv::Rect detected_rect = obj_results[i].detected_object.rect;
    detected_rect = FloorRectScale(detected_rect, scale_);
    detections.push_back(detected_rect);
  }

  vector<int> matched_tracks, matched_detections, unmatched_tracks, unmatched_detections;
  //Timer match_timer;
  //match_timer.tic();
  IoU_matching(tracks, detections, matched_tracks, matched_detections, unmatched_tracks, unmatched_detections);
  //match_timer.toc("iou match");
  /*
    mtached_detections detected_status 1
    unmatched_detections detect_status 1
    unmatch_tracks detect_status 0
  */
  Timer matched_detect_timer;
  matched_detect_timer.tic();
  vector<KCF> kcf_trackers_update;
  vector<ObjectResult> current_tracked_objects_update;
  for(size_t i = 0; i < matched_detections.size(); i++) {
    int idx = matched_detections[i];
    KCF kcf("gaussian", "gray");
    kcf.Init(frame_, detections[idx]); //matched tracks
    kcf_trackers_update.push_back(kcf);
    current_tracked_objects[matched_tracks[i]].detected_object = obj_results[idx].detected_object;
    obj_results[idx] = current_tracked_objects[matched_tracks[i]];
    obj_results[idx].lost_times = 0;
    obj_results[idx].detected_status = 1;
    obj_results[idx].track_times += 1;
    obj_results[idx].rect = FloorRectScale(detections[idx], 1.0 / scale_);
    current_tracked_objects_update.push_back(obj_results[idx]);
  }
  matched_detect_timer.toc("matched_detect_timer");
  // cout<<"update:"<<current_tracked_objects_update.size()<<endl;

  //Timer unmatched_detect_timer;
  //unmatched_detect_timer.tic();
  for(size_t i = 0; i < unmatched_detections.size(); i++) {
    int idx = unmatched_detections[i];
    KCF kcf("gaussian", "gray");
    kcf.Init(frame_, detections[idx]); //matched tracks
    kcf_trackers_update.push_back(kcf);
    obj_results[idx].lost_times = 0;
    obj_results[idx].detected_status = 1;
    obj_results[idx].track_id = track_id++;
    obj_results[idx].rect = FloorRectScale(detections[idx], 1.0 / scale_);
    current_tracked_objects_update.push_back(obj_results[idx]);
  }
  //unmatched_detect_timer.toc("unmatched_detect_timer");
  // cout<<"update:"<<current_tracked_objects_update.size()<<endl;

  for(size_t i = 0; i < unmatched_tracks.size(); i++) {
    int idx = unmatched_tracks[i];
    current_tracked_objects[idx].lost_times += 1;
    // cout<<current_tracked_objects[idx].track_id<<", lost time:"<<current_tracked_objects[idx].lost_times<<endl;
    if(current_tracked_objects[idx].lost_times >= 1) {
      continue;
    }
    else{
      kcf_trackers_update.push_back(kcf_trackers[idx]);
      current_tracked_objects[idx].detected_status = 0;
      current_tracked_objects_update.push_back(current_tracked_objects[idx]);
    }
  }
  // cout<<"current:"<<current_tracked_objects.size()<<endl;
  // cout<<"update:"<<current_tracked_objects_update.size()<<endl;
  kcf_trackers = kcf_trackers_update;
  current_tracked_objects = current_tracked_objects_update;

  IncreaseFrameCount();
  return current_tracked_objects;
}

cv::Rect Tracker::FloorRectScale(cv::Rect &r, double scale_factor) 
{
  if (scale_factor > 0.9999 && scale_factor < 1.0001)
    return r;
  return cv::Rect(cvFloor(r.x * scale_factor), cvFloor(r.y * scale_factor), cvFloor(r.width * scale_factor), cvFloor(r.height * scale_factor));
}

void Tracker::IoU_cost(vector<cv::Rect> &tracks, vector<cv::Rect> &detections, cv::Mat &cost_matrix){
  cost_matrix = cv::Mat::zeros(tracks.size(), detections.size(), CV_32FC1);
  for(size_t i = 0; i < tracks.size(); i++){
    for(size_t j = 0; j < detections.size(); j++){
      float cost = iou(tracks[i], detections[j]);
      cost_matrix.at<float>(i, j) = 1.0 - cost;
    }
  }
}

float Tracker::iou(cv::Rect &r1, cv::Rect &r2){
  cv::Rect inter = r1 & r2;
  int area_inter = inter.area();
  if(area_inter <= 0) return 0.0;
  float iou = (float) area_inter / (r1.area() + r2.area() - area_inter);
  return iou;
}

void Tracker::IoU_matching(vector<cv::Rect> &tracks, vector<cv::Rect> &detections, vector<int> &matched_tracks, vector<int> &matched_detections, vector<int> &unmatched_tracks, vector<int> &unmatched_detections)
{
  if(tracks.size() == 0 )
  {
    for(size_t i = 0; i < detections.size(); i++){
      unmatched_detections.push_back(i);
    }
    return;
  }

  if(detections.size() == 0){
    for(int i = 0; i < tracks.size(); i++){
      unmatched_tracks.push_back(i);
    }
    return;
  }
  cv::Mat cost_matrix;
  IoU_cost(tracks, detections, cost_matrix);
  vector<vector<double> > costMatrix;
  costMatrix.resize(cost_matrix.rows);
  for(int i = 0; i < cost_matrix.rows; i++){
    costMatrix[i].resize(cost_matrix.cols);
    for(int j = 0; j < cost_matrix.cols; j++){
      if(cost_matrix.at<float>(i, j) > this->max_iou_distance){
        costMatrix[i][j] =  this->max_iou_distance + 1e-5;
      }
      else{
        costMatrix[i][j] =  cost_matrix.at<float>(i, j);
      }
    }
  }
  HungarianAlgorithm HungAlgo;
  // vector<int> assignment;
  double cost = 99999;
  vector<int> indices;
  cost = HungAlgo.Solve(costMatrix, indices);
  

  for(size_t i = 0; i < indices.size(); i++){
    if(indices[i] < 0){
      unmatched_tracks.push_back(i);
    }
    else{
      if(costMatrix[i][indices[i]] > this->max_iou_distance){
        unmatched_tracks.push_back(i);
        unmatched_detections.push_back(indices[i]);
      }
      else{
        matched_tracks.push_back(i);
        matched_detections.push_back(indices[i]);
      }
    }
  }
  for(size_t i = 0; i < detections.size(); i++){
    if(std::find(indices.begin(), indices.end(), i) == indices.end()){
      unmatched_detections.push_back(i);
    }
  }
}

