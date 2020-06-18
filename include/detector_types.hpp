#pragma once 
#include<string>
#include<opencv2/opencv.hpp>
#include<vector>
#include<map>

#define DetectorAPI

using namespace std;
using std::vector;
using std::string;

struct DetectorAPI Object{
    cv::Mat object_crop;
    cv::Rect rect;
    cv::Rect2f norm_rect;
    float confidence;
    cv::Mat mask;
    int type;//目标类型
    long detected_time;
};

enum DetectorAPI TrackStatus {
    Tentatvie,
    New,
    Confirmed,
    Deleted,
    Finished
};

/// @brief 检测跟踪结果的结构体.
struct DetectorAPI ObjectTrack {
  int camera_id;                          ///< 视频源编号
  long track_id;                          ///< 每个视频源从0自增
  Object selected_object;             ///< 最佳抓拍
  vector<Object> detected_objects;    ///< 保存每张抓拍
  int best_idx;                           ///< detected_faces[best_idx]是选出来的人脸
  bool is_best_idx_updated;               ///< best_idx是否更新过
  TrackStatus status;                     ///< 
  long start_time, end_time;              ///< track的开始和结束时间 
  int time_since_update;
};

//检测跟踪结构体
struct DetectorAPI ObjectResult {
  ObjectResult(): source_id(-1), 
                  track_id(-1), 
                  detected_status(-1), 
                  match_score(-1),
                  match_index(-1), 
                  lost_times(0), 
                  is_updated(0), 
                  track_times(0){};
  int source_id;
  long track_id;
  long frame_key;
  int detected_status;
  cv::Rect rect;
  cv::Rect2f norm_rect;
  Object detected_object;
  Object selected_object;
  float match_score;
  int match_index;
  int lost_times;
  int is_updated;
  int track_times;
  vector<cv::Point> centroid_tracks;//中心点序列，用于计算移动方向
};

class DetectorAPI DetectorParams{
public:
  string model_type;
  string model_path;
  int device_id;
  int video_detect_every_n;

  long Read(const string& config_path);
};

