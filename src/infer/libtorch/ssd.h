//
// Created by dl on 2019/12/3.
//

#ifndef TORCH_SSD_H
#define TORCH_SSD_H
#include "infer.h"
#include "nms_layer.h"
#include "torch/torch.h"
#include <opencv2/core/cuda.hpp>

class SsdParams {
public:
    int num_levels;
    cv::Size input_size;
    int num_classes;
    std::vector<int> in_channels;
    std::vector<float> anchor_strides;
    std::vector<std::vector<float>> anchor_ratios;
    std::vector<std::vector<float>> anchor_scales;
    std::vector<float> anchor_basesizes;
    std::vector<torch::Tensor> cell_anchors;
    std::vector<float> target_means;
    std::vector<float> target_stds;
    std::vector<float> input_means;
    std::vector<float> input_stds;
    std::vector<int> feature_map_sizes;
    bool rgb;
    bool dynamic_size=false;
    float score_threshold = 0.02;
    float nms_threshold = 0.5;
    float final_score_threshold = 0.3;

    int keep_top= 200;
    int level_pos_nms_count = 1000;
    float max_ratio;
    int64_t max_proposals = 1000;
};


class SSD : public ObjectDetector {
public:
    SSD(int device_id);
    void Init(const std::string& model_prefix);
    void Detect(const cv::Mat& image, std::vector<DetectedObject>& detected_objects);
private:
    torch::Tensor bbox_transfrom_torch(torch::Tensor& anchors, torch::Tensor& deltas);

    std::vector<float> get_debug_input();

    torch::Tensor all_anchors_;
    torch::jit::script::Module module_;
    torch::Tensor gt_mask_for_boxes_;
    SsdParams params_;
    torch::Device device_;
    std::vector<int> level_anchor_counts_;
    std::vector<int> level_anchor_offsets_;

    cv::cuda::GpuMat gpu_raw_;
    cv::cuda::GpuMat gpu_raw_float_;
    cv::cuda::GpuMat gpu_resize_;
    cv::cuda::GpuMat gpu_rgb_;
    cv::cuda::Stream cv_cuda_stream_;

    NmsLayer nms_layer_;

};


#endif