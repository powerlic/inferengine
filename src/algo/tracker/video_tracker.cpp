#include "video_tracker.hpp"
#include "tracker.h"



VideoTracker::VideoTracker()
{
  
}

VideoTracker::~VideoTracker()
{
  
}

long VideoTracker::AddTracker(int camera_id, int detect_every_n)
{
  if(trackers_.find(camera_id) == trackers_.end())
  {
      trackers_.insert(std::make_pair(camera_id, std::shared_ptr<Tracker>(new Tracker(camera_id, detect_every_n)))); 
  }
  else
  {
    // trackers_[camera_id] = std::shared_ptr<Tracker>(new Tracker(max_age, n_init));
  }
  return 0L; 
}

long VideoTracker::ReleaseTracker(int camera_id)
{
  if(trackers_.find(camera_id) != trackers_.end())
  {
    trackers_.erase(camera_id);
  }
  return 0L;
}

std::shared_ptr<Tracker> VideoTracker::GetTracker(int camera_id)
{
  if(trackers_.find(camera_id) != trackers_.end()){
    return trackers_[camera_id];
  }
  return nullptr;
}

long VideoTracker::Release()
{
  trackers_.clear();
  return 0;
}
