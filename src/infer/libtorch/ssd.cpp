#include "ssd.h"
#include "torch/script.h"
#include "utils/logging.hpp"
#include "anchors.hpp"
#include "nms.h"
#include <fstream>

#ifdef TIME_LOG
#include "utils/timer.hpp"
#endif

using namespace easycv;
using namespace std;

SSD::SSD(int device_id) : device_(torch::kCUDA, device_id) 
{
}


void SSD::Init(const std::string& model_path) {

    params_.dynamic_size = false;
    params_.num_classes = 81;
    params_.num_levels = 7;
    params_.anchor_strides = {8., 16., 32., 64., 128., 256., 512.};
    params_.input_size = cv::Size(512, 512);
    params_.input_means = {123.675, 116.28, 103.53}; 
    params_.target_means = {0., 0., 0., 0.};
    params_.target_stds = {0.1, 0.1, 0.2, 0.2};
    params_.max_ratio = 0.456;

    torch::Tensor level1_anchors = torch::tensor(
        {-6.,  -6.,  13.,  13.,
        -12., -12.,  19.,  19.,
        -10.,  -3.,  17.,  10.,
        -3., -10.,  10.,  17. }).view({-1,4}).toType(torch::kFloat);

    torch::Tensor level2_anchors = torch::tensor(
        {-18., -18.,  32.,  32.,
        -33., -33.,  48.,  48.,
        -28., -10.,  43.,  25.,
        -10., -28.,  25.,  43.,
        -36.,  -7.,  51.,  22.,
        -7., -36.,  22.,  51.}).view({-1, 4}).toType(torch::kFloat);

    torch::Tensor level3_anchors = torch::tensor(
        {-50., -50.,  82.,  82.,
         -69., -69., 100., 100.,
         -78., -31., 109.,  62.,
         -31., -78.,  62., 109.,
         -99., -22., 130.,  53.,
         -22., -99.,  53., 130.}).view({-1, 4}).toType(torch::kFloat);

    torch::Tensor level4_anchors = torch::tensor(
        {-76.,  -76.,  138.,  138.,
         -94.,  -94.,  157.,  157.,
         -120.,  -44.,  183.,  107.,
         -44., -120.,  107.,  183.,
         -154.,  -30.,  217.,   93.,
         -30., -154.,   93.,  217.}).view({-1, 4}).toType(torch::kFloat);

    torch::Tensor level5_anchors = torch::tensor(
        {-84.,  -84.,  211.,  211.,
         -103., -103.,  230.,  230.,
         -145.,  -41.,  272.,  168.,
         -41., -145.,  168.,  272.,
         -192.,  -21.,  319.,  148.,
         -21., -192.,  148.,  319.}).view({-1, 4}).toType(torch::kFloat);

    torch::Tensor level6_anchors = torch::tensor(
        {-61.,  -61.,  316.,  316.,
         -80.,  -80.,  335.,  335.,
         -139.,   -6.,  394.,  261.,
         -6., -139.,  261.,  394.}).view({-1, 4}).toType(torch::kFloat);

    torch::Tensor level7_anchors = torch::tensor(
        {26.,  26.,  485.,  485.,
         6.,   6., 505., 505.,
         -69.,  93., 580., 418.,
         93., -69., 418., 580.}).view({-1, 4}).toType(torch::kFloat);

    params_.cell_anchors.push_back(level1_anchors);
    params_.cell_anchors.push_back(level2_anchors);
    params_.cell_anchors.push_back(level3_anchors);
    params_.cell_anchors.push_back(level4_anchors);
    params_.cell_anchors.push_back(level5_anchors);
    params_.cell_anchors.push_back(level6_anchors);
    params_.cell_anchors.push_back(level7_anchors);

    params_.feature_map_sizes = {64, 32, 16, 8, 4, 2, 1};
    int offset = 0;
    std::vector<torch::Tensor> level_anchors_list;
    for(auto i = 0; i <7; i++) {
        float ctr = ( params_.anchor_strides[i] - 1 )/2; 
        cv::Size size(params_.feature_map_sizes[i], params_.feature_map_sizes[i]);
        auto level_anchors = easycv::generate_total_anchors(
            params_.cell_anchors[i], size, params_.anchor_strides[i]
        );
        level_anchors_list.push_back(level_anchors);
        level_anchor_counts_.push_back(level_anchors.size(0));
        level_anchor_offsets_.push_back(offset);
        offset += level_anchors.size(0);
    }
    all_anchors_ = torch::cat(level_anchors_list, 0);
    all_anchors_ = all_anchors_.to(device_);

    std::cout<<all_anchors_.dim()<<std::endl;
    for(auto j=0;j<all_anchors_.dim();j++){
         std::cout<<all_anchors_.size(j)<<" ";
    }

    //nms_layer_ = NmsLayer();

    module_ = torch::jit::load(model_path);
    module_.eval();
    module_.to(device_);
    getchar();
}

void SSD::Detect(const cv::Mat& image, std::vector<DetectedObject>& detected_objects)
{
#ifdef TIME_LOG
    Timer total_timer;
    total_timer.tic();
#endif
    torch::NoGradGuard no_grad_guard;
    
    gpu_raw_.upload(image, cv_cuda_stream_);
    gpu_raw_.convertTo(gpu_raw_float_, CV_32FC3, 1, 0, cv_cuda_stream_);
    cv::cuda::resize(gpu_raw_float_, gpu_resize_, params_.input_size, 0, 0, cv::INTER_LINEAR, cv_cuda_stream_);
    cv::cuda::cvtColor(gpu_resize_, gpu_rgb_, cv::COLOR_BGR2RGB, 0, cv_cuda_stream_);
    cv::Mat rgb_img;
    gpu_rgb_.download(rgb_img, cv_cuda_stream_);

    // cv::Mat raw_float_;
    // cv::Mat rgb_img;
    // image.convertTo(raw_float_, CV_32FC3, 1, 0);
    // cv::resize(raw_float_, rgb_img, params_.input_size, 0, 0, cv::INTER_LINEAR);

    torch::Tensor tensor_image = torch::from_blob(rgb_img.data, {1, params_.input_size.height, params_.input_size.width, 3}, torch::kFloat);
    tensor_image = tensor_image.to(device_);
    tensor_image = tensor_image.permute({0, 3, 1, 2});

    tensor_image[0][0].sub_(params_.input_means[0]);
    tensor_image[0][1].sub_(params_.input_means[1]);
    tensor_image[0][2].sub_(params_.input_means[2]);

    // vector<float> img_data = get_debug_input();
    // torch::Tensor tensor_image = torch::tensor(img_data).reshape({1, 3, 512, 512}).to(torch::kFloat);
    // tensor_image = tensor_image.to(device_);

    auto base = module_.forward({tensor_image}).toTuple();

    torch::Tensor scores = base->elements()[0].toTensor().view({-1, params_.num_classes});
    torch::Tensor deltas = base->elements()[1].toTensor().view({-1, 4});


    // ofstream out_f("ssd_scores.txt");
    // torch::Tensor p = scores.to(torch::kCPU);
    // float *ps = p.data<float>();
    // out_f.setf(ios::fixed, ios::floatfield);  // 设定为 fixed 模式，以小数点表示浮点数
    // out_f.precision(4);  // 设置精度 2
    // for(int i = 0; i < scores.size(0); i++)
    // {
    //     int dim = scores.size(1);
    //     for(int j = 0; j < dim; j++)
    //     {
    //         out_f<<ps[dim*i+j]<<" ";
    //         //cout<<ps[dim*i+j]<<" ";
    //     }
    //     out_f<<std::endl;
    // }
    // out_f<<std::endl;
    // out_f.close();

    //torch::Tensor pred_boxes = bbox_transfrom_torch(all_anchors_, deltas);
    // std::cout<<"scores "<<scores.dim()<<std::endl;
    // for(auto j=0;j<scores.dim();j++){
    //      std::cout<<scores.size(j)<<" ";
    // }
    // std::cout<<std::endl;
    vector<torch::Tensor> final_instances; 
    // std::cout<<"scores"<<endl;
    // std::cout<<scores<<endl;
    //cout<<"params_.score_threshold "<<params_.score_threshold<<endl;
    auto pred_scores = scores.slice(1, 1, params_.num_classes, 1);
    auto gt_mask = pred_scores > params_.score_threshold;
    //auto multi_boxes = pred_boxes.view({pred_boxes.size(0), -1, 4}).expand({pred_boxes.size(0), params_.num_classes-1, 4});

    auto expand_all_anchors = all_anchors_.view({all_anchors_.size(0), -1, 4}).expand({all_anchors_.size(0), params_.num_classes-1, 4});
    auto expand_all_deltas  = deltas.view({deltas.size(0), -1, 4}).expand({deltas.size(0), params_.num_classes-1, 4});



    gt_mask_for_boxes_ = gt_mask.view({gt_mask.size(0),gt_mask.size(1),1}).expand({gt_mask.size(0),gt_mask.size(1),4});
    auto selected_deltas = expand_all_deltas.masked_select(gt_mask_for_boxes_).view({-1, 4});


    auto selected_anchors = expand_all_anchors.masked_select(gt_mask_for_boxes_).view({-1, 4});
    auto pred_boxes = bbox_transfrom_torch(selected_anchors, selected_deltas);

    auto valid_scores = pred_scores.masked_select(gt_mask);
    //auto valid_boxes  = multi_boxes.masked_select(gt_mask.view({gt_mask.size(0),gt_mask.size(1),1}).expand({{gt_mask.size(0),gt_mask.size(1),4}})).view({-1, 4});
    auto valid_boxes = pred_boxes;
    //auto valid_labels =  gt_mask.nonzero();

    auto labels = gt_mask.nonzero().select(1, 1);

    auto max_cord = valid_boxes.max();

    auto offsets = labels.view({-1, 1}).expand({valid_scores.size(0), 4}).mul(max_cord);
    auto valid_boxes_with_offset = valid_boxes.add(offsets);

 
    auto nms_input = torch::cat({valid_boxes_with_offset, valid_scores.view({-1,1})}, 1);
    
    //auto nms_ids = nms_cuda(nms_input, params_.nms_threshold);
    auto nms_ids = nms_layer_.nms_cuda(nms_input, params_.nms_threshold);

    auto nms_output_boxes = valid_boxes.index_select(0, nms_ids);
    auto nms_output_scores = valid_scores.index_select(0, nms_ids);
    auto nms_output_labels = labels.index_select(0, nms_ids);

    if(nms_output_labels.size(0)> params_.keep_top){
        auto total_scores_tuple = torch::sort(nms_output_scores, 0, 1);
        auto total_order_single = std::get<1>(total_scores_tuple);
        auto select_inds = total_order_single.slice(0, 0, params_.keep_top);
        nms_output_boxes = nms_output_boxes.index_select(0, select_inds);
        nms_output_scores = nms_output_scores.index_select(0, select_inds);
        nms_output_labels = nms_output_labels.index_select(0, select_inds);
    }

    auto final_preds = torch::cat({nms_output_boxes, nms_output_scores.view({-1,1}), nms_output_labels.view({-1, 1}).to(torch::kFloat)}, 1).to(torch::kCPU);


    auto clamp = [](float x, float min, float max)->float{
        if (x > max)
            return max;
        if (x < min)
            return min;
        return x;
    };
    auto pdata = final_preds.data<float>();
    for(auto i=0; i<final_preds.size(0);i++)
    {
        float x1 = clamp(pdata[6*i+0], 0, params_.input_size.width-1)/params_.input_size.width;
        float y1 = clamp(pdata[6*i+1], 0, params_.input_size.height-1)/params_.input_size.height;
        float x2 = clamp(pdata[6*i+2], 0, params_.input_size.width-1)/params_.input_size.width;
        float y2 = clamp(pdata[6*i+3], 0, params_.input_size.height-1)/params_.input_size.height; 
        float score = pdata[6*i+4];
        if(score < params_.final_score_threshold) continue;
        int label = static_cast<int>(pdata[6*i+5]);
        float w = x2 - x1;
        float h = y2 - y1;
        float c_x = (x1 + x2)/2;
        float c_y = (y1 + y2)/2;
        if(x2>x1 && y2>y1) 
        {
            detected_objects.push_back(DetectedObject(c_x, c_y, w, h, label, score));
        }
    } 

#ifdef TIME_LOG
    total_timer.toc("total time:");
#endif
}


torch::Tensor SSD::bbox_transfrom_torch(torch::Tensor& anchors, torch::Tensor& deltas)
{
    auto std = torch::tensor(params_.target_stds).to(device_);
    std = std.repeat({1, static_cast<int>(deltas.size(1)/4)});
    auto denorm_deletas = deltas.mul(std);
  
    int W = deltas.size(1);
    auto dx = denorm_deletas.slice(1, 0, W, 4);
    auto dy = denorm_deletas.slice(1, 1, W, 4);
    auto dw = denorm_deletas.slice(1, 2, W, 4).clamp(-params_.max_ratio, params_.max_ratio);
    auto dh = denorm_deletas.slice(1, 3, W, 4).clamp(-params_.max_ratio, params_.max_ratio);

    auto widths  =  anchors.select(1, 2).sub(anchors.select(1, 0)).unsqueeze(1).expand_as(dx).add(1.0);
    auto heights =  anchors.select(1, 3).sub(anchors.select(1, 1)).unsqueeze(1).expand_as(dy).add(1.0);
    auto ctr_x   =  anchors.select(1, 0).unsqueeze(1).expand_as(dw).add(widths.mul(0.5));
    auto ctr_y   =  anchors.select(1, 1).unsqueeze(1).expand_as(dw).add(heights.mul(0.5));

    auto pred_ctr_x = dx.mul(widths).add(ctr_x);
    auto pred_ctr_y = dy.mul(heights).add(ctr_y);
    auto pred_w = torch::exp(dw).mul(widths);
    auto pred_h = torch::exp(dh).mul(heights);

    auto proposals_x1 = pred_ctr_x.sub(pred_w.mul(0.5));
    auto proposals_y1 = pred_ctr_y.sub(pred_h.mul(0.5));
    auto proposals_x2 = pred_ctr_x.add(pred_w.mul(0.5)).sub(1);
    auto proposals_y2 = pred_ctr_y.add(pred_h.mul(0.5)).sub(1);

    auto proposals = torch::stack({proposals_x1, proposals_y1, proposals_x2, proposals_y2}, -1).view_as(deltas);
    //cout<<"proposals "<<endl;
    //cout<<proposals<<endl;
    return proposals;

}