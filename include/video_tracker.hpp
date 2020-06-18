// Copyright [2017] <Tunicorn Inc.>
#ifndef TRACKER_HPP
#define TRACKER_HPP
#include <vector>
#include <memory>
#include <unordered_map>


class Tracker;
/**
 * @brief 跟踪对象的封装类
 */
class VideoTracker {
public:
    VideoTracker();
    ~VideoTracker();
    
    /**
     * @brief 增加一个跟踪对象
     * @param camera_id 视频源id，如果id已存在，则直接返回
     * @param detect_every_n 跳帧数
     */
    long AddTracker(int camera_id, int detect_every_n = 5);
    
    /**
     * @brief 释放一个跟踪对象
     * @param camera_id 视频源id
     */
    long ReleaseTracker(int camera_id);
    
    /**
     * @brief 获取一个跟踪对象的指针
     * @param camera_id 视频源id
     */
    std::shared_ptr<Tracker> GetTracker(int camera_id);
    
    /**
     * @brief 释放所有跟踪对象
     */
    long Release();
    
private:
    std::unordered_map<int, std::shared_ptr<Tracker> > trackers_;
};



#endif