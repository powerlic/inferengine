//
// Created by dl on 2019/12/3.
//
#ifndef FASTERRCNN_H
#define FASTERRCNN_H
#include "infer.h"
#include "torch/torch.h"
#include <opencv2/core/cuda.hpp>

class FasterRcnnParams {
public:
    int num_levels;
    cv::Size input_size;
    int num_classes=3;
    std::vector<int> in_channels;
    torch::Tensor cell_anchors;
    torch::Tensor target_means;
    torch::Tensor target_stds;

    torch::Tensor bbox_head_target_stds;
    torch::Tensor bbox_head_target_means;

    std::vector<float> input_means;
    std::vector<float> input_stds;
    std::vector<int> feature_map_sizes;
    bool rgb;
    bool dynamic_size=false;
    float score_threshold = 0.5;
    float rpn_nms_threshold = 0.7;
    float final_score_threshold = 0.3;
    float final_nms_threshold = 0.5;
    int stride = 16;

    int keep_top= 200;
    int pre_nms_count = 6000;
    int post_nms_count = 100;
    float max_ratio=4.135166556742356;
    int64_t max_proposals = 100;
    int roi_align_feat_size = 14;
    int roi_algin_out_channels = 1024;
};


class FasterRcnn: public ObjectDetector
{
public:
    FasterRcnn(int device_id);
    void Init(const std::string& model_prefix);
    void Detect(const cv::Mat& image, std::vector<DetectedObject>& detected_objects);

protected:
    FasterRcnnParams params_;
    torch::jit::script::Module base_net_;
    torch::jit::script::Module shared_head_;
    torch::jit::script::Module bbox_head_;
    torch::Device device_;
    torch::Tensor all_anchors_;

protected:
    void generate_cell_anchors();
    void generate_all_anchors(cv::Size feature_map_size);
    torch::Tensor bbox_transfrom_torch(torch::Tensor& anchors, torch::Tensor& deltas, 
        torch::Tensor& stds, torch::Tensor& means, int max_w=-1, int max_h=-1);


    cv::cuda::GpuMat gpu_raw_;
    cv::cuda::GpuMat gpu_raw_float_;
    cv::cuda::GpuMat gpu_resize_;
    cv::cuda::Stream cv_cuda_stream_;

};




#endif //FASTERRCNN_H