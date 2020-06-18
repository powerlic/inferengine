//
// Created by dl on 2019/12/3.
//
#ifndef FASTERRCNNRPN_H
#define FASTERRCNNRPN_H
#include "infer.h"
#include "torch/torch.h"
#include "nms_layer.h"

using namespace std;

class FasterRcnnFpnParams {
    //input setting
    vector<float> input_means;
    vector<float> input_stds;
    bool to_rgb;
    cv::Size img_scale;

    //rpn_head
    vector<float> anchor_scales;
    vector<float> anchor_ratios;
    vector<float> anchor_strides;
    vector<float> rpn_head_target_means;
    vector<float> rpn_head_target_stds;
};

class FasterRcnnFpn final: public ObjectDetector
{
public:
    FasterRcnnFpn(int device_id);
    void Init(const std::string& model_prefix);
    void Detect(const cv::Mat& image, std::vector<DetectedObject>& detected_objects) override;

protected:
    void generate_cell_anchors();

    void get_input_scale(const cv::Mat& frame);

    void boxes_sort(const int num, const float* pred, float* sorted_pred);

    void bbox_transform_inv(torch::Tensor& rois, torch::Tensor& cls_prob, torch::Tensor& bbox_pre,
                            int img_width, int img_height, float scale, float* prob);

    torch::Tensor bbox_transform_torch(torch::Tensor& anchors, torch::Tensor& deltas);
    torch::Tensor bbox_transfrom_torch2(torch::Tensor& rois, torch::Tensor& deltas);

    torch::Tensor mlvl_nms(const torch::Tensor& deltas, const torch::Tensor& scores);
    torch::Tensor target_lvls(torch::Tensor& rois);

    // member
    // network
    torch::jit::script::Module m_feature_rpn;
    torch::jit::script::Module m_boxhead;
    torch::jit::script::Module m_mask;

    // devices type (torch::kCUDA  and torch::kCPU)
    // torch::DeviceType m_deviceType; //设置Device类型
    torch::Device m_device;

    torch::Tensor cell_anchors_p2_;
    torch::Tensor cell_anchors_p3_;
    torch::Tensor cell_anchors_p4_;
    torch::Tensor cell_anchors_p5_;
    torch::Tensor cell_anchors_p6_;

    //p2 - p6
    void generate_all_anchors(std::vector<cv::Size> feature_map_sizes);
    std::vector<int> strides_={4, 8, 16, 32, 64};
    std::vector<torch::Tensor> cell_anchors_;
    //std::vector<torch::Tensor> all_anchors_;
    torch::Tensor all_anchors_;
    vector<int> level_anchor_offsets_;
    vector<int> level_anchor_counts_;
    int level_count_=4;

    int m_cell_anchors;

    float level_nms_threshold = 0.7;
    int level_pre_nms_count=  1000;
    int level_pos_nms_count = 1000;
    int64_t max_proposals_ = 1000;
    float finest_scale = 56;

    float final_nms_threshold_ = 0.5;

    float mask_thresholds_ = 0.5;
    int raw_mask_size_ = 28;

    // scale
    cv::Size input_size_;
    int m_category;
    int m_boxNum;

    // threshold
    float m_score_threshold=0.5;
    int num_classes = 3;

    float max_ratio_ = 4.135166556742356;

    vector<float> get_debug_input();

    cv::Size input_scale_={128, 384};
    cv::Size valid_size_ = {128, 384};
    float scale_fator_;

    NmsLayer nms_layer_;
};




#endif