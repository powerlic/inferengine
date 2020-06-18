#include "test_common.hpp"

int test_video_tracking(int argc, char**argv)
{   
    get_random_colors();
    //std::string video_src = "/home/yangyoulei/road.mp4";
    string video_src{argv[1]};
    string video_name = GetVideoName(video_src);
    cv::VideoCapture cap(video_src);
    if (!cap.isOpened()) {
        std::cout << video_src << " cannot be opened" << std::endl;
        exit(-1);
    }
    cout<<"open video"<<video_src<<endl;
    int target_frame_num = atoi(argv[2]);

    DetectorParams parameters;
    parameters.Read(CONFIG_FILE);
    cout<<"using config file "<<CONFIG_FILE<<endl;
    int fps = 3;
    auto out_video_name = video_name + "_res.avi";
    cv::Size demo_size(128, 384);
    cv::VideoWriter video(out_video_name, CV_FOURCC('X', 'V', 'I', 'D'), 3, demo_size, true);
    auto font_face = CV_FONT_HERSHEY_SIMPLEX;
    auto font_scale = 1.2;
    auto frame_count = 0;
    auto save_frame_count = 0;
    auto det_time = 0.0;
    int device_id = 0;
    int frameidx=0;

    Detector video_detector;
    video_detector.Init(parameters);
    cout<<"Init video detector"<<endl;
    //tracker
    VideoTracker tracking_manager;
    tracking_manager.Release();
    tracking_manager.AddTracker(0, parameters.video_detect_every_n);

    double time_total = 0;
    float thickness = 4;
    while(1)
    {
        cv::Mat frame;
        if(!cap.read(frame)) 
        {
            cout<<video_src<< " read frame:"<<frameidx<<" error"<<endl;
                break;
        }
        cv::Mat copy_frame = frame.clone();
        tic();
        cout<<"frame: "<<frameidx<<endl;
        vector<ObjectResult> object_results;
        video_detector.DetectWithTracking(frame, (int64_t) frameidx, tracking_manager.GetTracker(0), object_results);    
        double diff = toc("detect_with_tracking");
        bool has_pedestrian = false;
        time_total += diff;
        for(int i = 0; i < object_results.size(); i++) {
            const auto& obj_result = object_results[i];
            cout<<i<<" trackid:"<<object_results[i].track_id<<", sourceid:"<<object_results[i].source_id<<endl;
            cv::Scalar color = cv::Scalar(0, 255, 255);
            string track_id_text = to_string(object_results[i].track_id);
            Object det_obj = obj_result.detected_object;
            cv::rectangle(frame, object_results[i].rect, randomColors[det_obj.type], thickness);
            string type_text = std::to_string(det_obj.type);
            string label_text = track_id_text+"|"+ type_text;
            cv::putText(frame, label_text, cv::Point(det_obj.rect.x, det_obj.rect.y-10), 
                font_face, font_scale, randomColors[det_obj.type], thickness, 8);
        }
        double avg_time = time_total / (frameidx+1);
        int fps = int(1000 / avg_time);
        char buff[100];
        snprintf(buff, sizeof(buff), "%05d process time: %03d ms, fps: %d", (int)frameidx, int(diff), fps);
        std::string text = buff;
        //cout<<text<<endl;
        cv::putText(frame, text, cv::Point(50, 200), font_face, font_scale, cv::Scalar::all(255), thickness, 8);
        cv::Mat out;
        cv::resize(frame, out, demo_size, cv::INTER_LINEAR);
        video.write(out);
        frameidx += 1;

        if(frameidx > target_frame_num) {
            break;
        }
  }
  cap.release();
  video.release();
}

int test_video_detect(int argc, char**argv){
    get_random_colors();
    //std::string video_src = "/home/yangyoulei/road.mp4";
    string video_src{argv[1]};
    string video_name = GetVideoName(video_src);
    cv::VideoCapture cap(video_src);
    if (!cap.isOpened()) {
        std::cout << video_src << " cannot be opened" << std::endl;
        exit(-1);
    }
    cout<<"open video"<<video_src<<endl;
    int target_frame_num = atoi(argv[2]);

    DetectorParams parameters;
    parameters.Read(CONFIG_FILE);

    auto out_video_name = video_name + "_res.avi";
    cv::Size demo_size(1248, 800);
    cv::VideoWriter video(out_video_name, CV_FOURCC('X', 'V', 'I', 'D'), 25, cv::Size(1248, 800), true);
    auto font_face = CV_FONT_HERSHEY_SIMPLEX;
    auto font_scale = 1.2;
    auto frame_count = 0;
    auto save_frame_count = 0;
    auto det_time = 0.0;
    int device_id = 1;
    int frameidx=0;

    Detector video_detector;
    video_detector.Init(parameters);
    cout<<"Init video detector"<<endl;
    //tracker    
    double time_total = 0;
    float thickness = 4;
    while(1)
    {
        cv::Mat frame;
        if(!cap.read(frame)) {
        cout<<video_src<< " read frame:"<<frameidx<<" error"<<endl;
            break;
        }
        cv::Mat copy_frame = frame.clone();
        tic();
        cout<<"frame: "<<frameidx<<endl;
        vector<Object> objects;
        video_detector.Detect(frame, objects);    
        double diff = toc("detect_with_tracking");
        bool has_pedestrian = false;
        time_total += diff;
        for(int i = 0; i < objects.size(); i++) {
            const auto& obj = objects[i];

            //cout<<i<<" trackid:"<<object_results[i].track_id<<", sourceid:"<<object_results[i].source_id<<endl;
            cv::Scalar color = cv::Scalar(0, 255, 255);
            cv::rectangle(frame, obj.rect, randomColors[obj.type], thickness);
            string type_text = std::to_string(obj.type);
            cv::putText(frame, type_text, cv::Point(obj.rect.x, obj.rect.y-10), 
                font_face, font_scale, randomColors[obj.type], thickness, 8);
        }
        double avg_time = time_total / (frameidx+1);
        int fps = int(1000 / avg_time);
        char buff[100];
        snprintf(buff, sizeof(buff), "%05d process time: %03d ms, fps: %d", (int)frameidx, int(diff), fps);
        std::string text = buff;
        //cout<<text<<endl;
        cv::putText(frame, text, cv::Point(50, 200), font_face, font_scale, cv::Scalar::all(255), thickness, 8);
        cv::Mat out;
        cv::resize(frame, out, demo_size, cv::INTER_LINEAR);
        video.write(out);
        frameidx += 1;

        if(frameidx > target_frame_num) {
            break;
        }
  }
  cap.release();
  video.release();
}

int main(int argc, char**argv)
{
    test_video_tracking(argc, argv);
}