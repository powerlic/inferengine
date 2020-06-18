//
// Created by dl on 2019/12/3.
//

#include "fasterrcnnfpn.h"
#include "roi_align.h"
#include "nms.h"
#include "torch/script.h"

#include <string>
#include <fstream>

FasterRcnnFpn::FasterRcnnFpn(int device_id):m_device(torch::kCUDA, device_id)
{
    m_cell_anchors = 3;
    input_size_ = cv::Size(128, 384);
    nms_layer_ = NmsLayer();
}

void FasterRcnnFpn::Init(const std::string& model_prefix)
{   
    LOG(INFO)<<"load model from "<<model_prefix<<endl;
    string base_model = model_prefix + "/faster_rcnn_backbone.pt";
    m_feature_rpn = torch::jit::load(base_model);
    m_feature_rpn.to(m_device);
    m_feature_rpn.eval();
    LOG(INFO)<<"load backbone done "<<model_prefix<<endl;

    string box_head_model = model_prefix + "/faster_rcnn_boxhead.pt";
    m_boxhead = torch::jit::load(box_head_model);
    m_boxhead.to(m_device);
    m_boxhead.eval();
    
    LOG(INFO)<<"load boxhead done "<<model_prefix<<endl;

    generate_cell_anchors();

    m_category = 2;
    m_boxNum = 300;
    m_score_threshold = 0.5;
}

void FasterRcnnFpn::generate_cell_anchors()
{   
    cell_anchors_p2_ = torch::ones({m_cell_anchors, 4}).toType(torch::kFloat);
    cell_anchors_p3_ = torch::ones({m_cell_anchors, 4}).toType(torch::kFloat);
    cell_anchors_p4_ = torch::ones({m_cell_anchors, 4}).toType(torch::kFloat);
    cell_anchors_p5_ = torch::ones({m_cell_anchors, 4}).toType(torch::kFloat);
    cell_anchors_p6_ = torch::ones({m_cell_anchors, 4}).toType(torch::kFloat);

    cell_anchors_p2_.select(0, 0) = torch::tensor({-21, -9, 24, 12}).toType(torch::kFloat);
    cell_anchors_p2_.select(0, 1) = torch::tensor({-14, -14, 17, 17}).toType(torch::kFloat);
    cell_anchors_p2_.select(0, 2) = torch::tensor({-9, -21, 12, 24}).toType(torch::kFloat);

    //need to check
    cell_anchors_p3_.select(0, 0) = torch::tensor({-41, -19, 48, 26}).toType(torch::kFloat);
    cell_anchors_p3_.select(0, 1) = torch::tensor({-28, -28, 35, 35}).toType(torch::kFloat);
    cell_anchors_p3_.select(0, 2) = torch::tensor({-19, -41, 26, 48}).toType(torch::kFloat);

    cell_anchors_p4_.select(0, 0) = torch::tensor({-83, -37, 98, 52}).toType(torch::kFloat);
    cell_anchors_p4_.select(0, 1) = torch::tensor({-56, -56, 71, 71}).toType(torch::kFloat);
    cell_anchors_p4_.select(0, 2) = torch::tensor({-37, -83, 52, 98}).toType(torch::kFloat);

    cell_anchors_p5_.select(0, 0) = torch::tensor({-165, -75, 196, 106}).toType(torch::kFloat);
    cell_anchors_p5_.select(0, 1) = torch::tensor({-112, -112, 143, 143}).toType(torch::kFloat);
    cell_anchors_p5_.select(0, 2) = torch::tensor({-75, -165, 106, 196}).toType(torch::kFloat);

    cell_anchors_p6_.select(0, 0) = torch::tensor({-330, -149, 393, 212}).toType(torch::kFloat);
    cell_anchors_p6_.select(0, 1) = torch::tensor({-224, -224, 287, 287}).toType(torch::kFloat);
    cell_anchors_p6_.select(0, 2) = torch::tensor({-149, -330, 212, 393}).toType(torch::kFloat);

    cell_anchors_.push_back(cell_anchors_p2_);
    cell_anchors_.push_back(cell_anchors_p3_);
    cell_anchors_.push_back(cell_anchors_p4_);
    cell_anchors_.push_back(cell_anchors_p5_);
    cell_anchors_.push_back(cell_anchors_p6_);
}

torch::Tensor FasterRcnnFpn::bbox_transform_torch(torch::Tensor& anchors, torch::Tensor& deltas)
{
    // std::cout<<"anchors dim "<<anchors.dim()<<std::endl;
    // for(auto i=0; i<anchors.dim(); i++){
    //     std::cout<<" "<<anchors.size(i)<<" ";
    // }
    // std::cout<<std::endl;

    //cout<<"anchors "<<endl;
    //cout<<anchors<<endl;

    //cout<<"deltas"<<endl;
    //cout<<deltas<<endl;
    auto widths = anchors.select(1, 2).sub(anchors.select(1, 0)).add(1.0);
    auto heights = anchors.select(1, 3).sub(anchors.select(1, 1)).add(1.0);
    auto ctr_x = anchors.select(1, 0).add(widths.mul(0.5));
    auto ctr_y = anchors.select(1, 1).add(heights.mul(0.5));

    auto dx = deltas.select(1, 0);
    auto dy = deltas.select(1, 1);
    auto dw = deltas.select(1, 2).clamp_(-max_ratio_, max_ratio_);
    auto dh = deltas.select(1, 3).clamp_(-max_ratio_, max_ratio_);

    auto pred_ctr_x = dx.mul(widths).add(ctr_x);
    auto pred_ctr_y = dy.mul(heights).add(ctr_y);
    auto pred_w = torch::exp(dw).mul(widths);
    auto pred_h = torch::exp(dh).mul(heights);

    auto proposals_x1 = (pred_ctr_x.unsqueeze(1).sub(pred_w.unsqueeze(1).mul(0.5)));
    auto proposals_y1 = (pred_ctr_y.unsqueeze(1).sub(pred_h.unsqueeze(1).mul(0.5)));
    auto proposals_x2 = (pred_ctr_x.unsqueeze(1).add(pred_w.unsqueeze(1).mul(0.5))).sub(1.0);
    auto proposals_y2 = (pred_ctr_y.unsqueeze(1).add(pred_h.unsqueeze(1).mul(0.5))).sub(1.0);

    proposals_x1.clamp_(0, valid_size_.width-1);
    proposals_x2.clamp_(0, valid_size_.width-1);
    proposals_y1.clamp_(0, valid_size_.height-1);
    proposals_y2.clamp_(0, valid_size_.height-1);

    auto proposals = torch::cat({proposals_x1, proposals_y1, proposals_x2, proposals_y2}, 1);
    //cout<<"proposals"<<endl;
    //cout<<proposals<<endl;
    return proposals;
}


torch::Tensor FasterRcnnFpn::bbox_transfrom_torch2(torch::Tensor& rois, torch::Tensor& deltas)
{

    auto std = torch::tensor({0.1, 0.1, 0.2, 0.2}).to(m_device);
    std = std.repeat({1, static_cast<int>(deltas.size(1)/4)});
    auto denorm_deletas = deltas.mul(std);
  
    int W = deltas.size(1);
    auto dx = denorm_deletas.slice(1, 0, W, 4);
    auto dy = denorm_deletas.slice(1, 1, W, 4);
    auto dw = denorm_deletas.slice(1, 2, W, 4).clamp(-max_ratio_, max_ratio_);
    auto dh = denorm_deletas.slice(1, 3, W, 4).clamp(-max_ratio_, max_ratio_);

    auto widths  =  rois.select(1, 3).sub(rois.select(1, 1)).unsqueeze(1).expand_as(dx).add(1.0);
    auto heights =  rois.select(1, 4).sub(rois.select(1, 2)).unsqueeze(1).expand_as(dy).add(1.0);
    auto ctr_x   =  rois.select(1, 1).unsqueeze(1).expand_as(dw).add(widths.mul(0.5));
    auto ctr_y   =  rois.select(1, 2).unsqueeze(1).expand_as(dw).add(heights.mul(0.5));

    auto pred_ctr_x = dx.mul(widths).add(ctr_x);
    auto pred_ctr_y = dy.mul(heights).add(ctr_y);
    auto pred_w = torch::exp(dw).mul(widths);
    auto pred_h = torch::exp(dh).mul(heights);

    // cout<<"ctr_x ";
    // cout<<ctr_x<<endl;
    // cout<<"crt_x "<<endl;
    // cout<<"crt_y "<<endl;
    auto proposals_x1 = pred_ctr_x.sub(pred_w.mul(0.5));
    auto proposals_y1 = pred_ctr_y.sub(pred_h.mul(0.5));
    auto proposals_x2 = pred_ctr_x.add(pred_w.mul(0.5)).sub(1);
    auto proposals_y2 = pred_ctr_y.add(pred_h.mul(0.5)).sub(1);

    auto proposals = torch::stack({proposals_x1, proposals_y1, proposals_x2, proposals_y2}, -1).view_as(deltas);
    // cout<<"proposals "<<endl;
    // cout<<proposals<<endl;
    return proposals;
}

void FasterRcnnFpn::generate_all_anchors(std::vector<cv::Size> feature_map_sizes)
{
    // p2-p6 anchors
    assert(feature_map_sizes.size()==5);
    std::vector<torch::Tensor> all_anchors;
    int offset=0;
    level_anchor_counts_.clear();
    level_anchor_offsets_.clear();
    for(auto i=0; i<feature_map_sizes.size(); i++) {
        cv::Size feature_size = feature_map_sizes[i];
        //std::cout<<"size "<<feature_size.height<<" "<<feature_size.width<<std::endl;
        auto shift_x1 = torch::arange(0, feature_size.width).mul(strides_[i]);
        auto shift_y1 = torch::arange(0, feature_size.height).mul(strides_[i]);
        auto shift_y11_x11 = torch::meshgrid({shift_y1, shift_x1});
        shift_y11_x11[0] = torch::reshape(shift_y11_x11[0], {-1, 1});
        shift_y11_x11[1] = torch::reshape(shift_y11_x11[1], {-1, 1});
        auto shifts = torch::cat({shift_y11_x11[1], shift_y11_x11[0], shift_y11_x11[1], shift_y11_x11[0]}, 1).contiguous();
        //std::cout<<"shifts "<<shifts.numel()<<std::endl;
        int total = feature_size.width*feature_size.height;
        //std::cout<<"total "<<total<<std::endl;
        auto anchors = cell_anchors_[i].expand({total, m_cell_anchors, 4}).add(shifts.view({total, 1, 4}).expand({total, m_cell_anchors, 4}));
        //std::cout<<"anchors "<<anchors.numel()<<std::endl;
        level_anchor_offsets_.push_back(offset);
        offset += anchors.numel()/4;
        level_anchor_counts_.push_back(anchors.numel()/4);
        anchors = anchors.view({1, total * m_cell_anchors, 4});
        //cout<<"level "<<i<<endl;
        //cout<<anchors<<endl;
        all_anchors.push_back(std::move(anchors));
        //anchor_count_offsets_.push_back(offset);
    }
    all_anchors_ = torch::cat({all_anchors[0], all_anchors[1], all_anchors[2], all_anchors[3], all_anchors[4]}, 1).contiguous().view({-1, 4}).to(m_device);
}

torch::Tensor FasterRcnnFpn::target_lvls(torch::Tensor& rois)
{
    auto widths = rois.select(1, 2).sub(rois.select(1, 0));
    auto heights = rois.select(1, 3).sub(rois.select(1, 1));
    auto areas = widths.mul(heights).add(1.0);
    auto scale = torch::sqrt(areas);
    auto target_levels = torch::floor(torch::log2(scale.div(finest_scale).add(1e-6))).toType(torch::kInt64);
    target_levels = target_levels.clamp(0, level_count_-1).toType(torch::kInt64);
    //cout<<"target_levels"<<endl;
    //cout<<target_levels<<endl;
    //cout<<"target_levels done"<<endl;
    return target_levels;
}

torch::Tensor FasterRcnnFpn::mlvl_nms(const torch::Tensor& deltas, const torch::Tensor& scores)
{
    assert(level_anchor_counts_.size()==5);
    assert(level_anchor_offsets_.size()==5);

    vector<torch::Tensor> level_proposals;
    for(int i=0;i<level_anchor_counts_.size();i++){
        
        int start = level_anchor_offsets_[i];
        int count = level_anchor_counts_[i];
        //cout<<start<<" "<<count<<endl;
        torch::Tensor level_deltas = deltas.slice(0, start, start+count, 1);
        torch::Tensor level_anchors = all_anchors_.slice(0, start, start+count, 1);
        torch::Tensor level_scores = scores.slice(0, start, start+count, 1);
        //torch::Tensor nms_input = torch::cat({level_boxes, level_scores}, 1);
        //cout<<"level_scores input "<<level_scores.size(0)<<endl;
        //cout<<"level_deltas input "<<level_deltas.size(0)<<" "<<level_deltas.size(1)<<endl;
        //cout<<"level_anchors input "<<level_anchors.size(0)<<" "<<level_anchors.size(1)<<endl;
        if(level_scores.size(0) > level_pre_nms_count) 
        {
            //cout<<"level "<<i<<endl;
            auto scores_tuple = torch::sort(level_scores, 0, 1);
            auto order_single = std::get<1>(scores_tuple).select(1, 0);
            order_single = order_single.slice(0, 0, level_pre_nms_count, 1);
            // nms_input = nms_input.index_select(0, order_single);
            level_deltas  = level_deltas.index_select(0, order_single);
            level_anchors = level_anchors.index_select(0, order_single);
            level_scores  = level_scores.index_select(0, order_single);
        }
        auto level_preds = bbox_transform_torch(level_anchors, level_deltas);
        auto nms_input = torch::cat({level_preds, level_scores.view({-1,1})}, 1);
        //cout<<"nms input "<<nms_input.size(0)<<" "<<nms_input.size(1)<<endl;
        //auto nms_ids = nms_cuda(nms_input, level_nms_threshold);
        auto nms_ids = nms_layer_.nms_cuda(nms_input, level_nms_threshold);
        auto top_nms_output = nms_input.index_select(0, nms_ids);
        if(top_nms_output.size(0)> level_pos_nms_count){
            top_nms_output = top_nms_output.slice(0, 0, level_pos_nms_count, 1);
        }
        level_proposals.push_back(top_nms_output);
    }

    auto proposals = torch::cat(level_proposals, 0);
    //cout<<"proposals before select "<<proposals.size(0)<<endl;
    auto total_scores = proposals.select(1, 4);
    auto total_scores_tuple = torch::sort(total_scores, 0, 1);
    int num = std::min(max_proposals_, proposals.size(0));
    auto total_order_single = std::get<1>(total_scores_tuple);
    auto select_inds = total_order_single.slice(0, 0, num);
    proposals = proposals.index_select(0, select_inds);
    // cout<<"total proposals "<<endl;
    // cout<<proposals<<endl;
    return proposals;
}


void FasterRcnnFpn::Detect(const cv::Mat& image, std::vector<DetectedObject>& detected_objects)
{
    torch::NoGradGuard no_grad_guard;
    cv::Mat f_img;
    image.convertTo(f_img, CV_32FC3, 1, 0);

     // resize
    cv::Mat bgr_img;
    cv::resize(f_img, bgr_img, input_size_);
    cv::Mat show;

    cv::Mat rgb_img;
    cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);

    torch::Tensor tensor_image = torch::from_blob(rgb_img.data, {1, input_size_.height, input_size_.width, 3}, torch::kFloat);
    tensor_image = tensor_image.permute({0, 3, 1, 2});
    tensor_image[0][0] = tensor_image[0][0].sub_(123.675);
    tensor_image[0][1] = tensor_image[0][1].sub_(116.28);
    tensor_image[0][2] = tensor_image[0][2].sub_(103.53);
    tensor_image[0][0] = tensor_image[0][0].div_(58.395);
    tensor_image[0][1] = tensor_image[0][1].div_(57.12);
    tensor_image[0][2] = tensor_image[0][2].div_(57.375);
    // // timer.toc("----process");
    tensor_image = tensor_image.to(m_device);

    // vector<float> debug_data = get_debug_input();
    // torch::Tensor debug_tensor = torch::from_blob(debug_data.data(), {1, 3, 768, 1344}, torch::kFloat);
    // debug_tensor = debug_tensor.to(m_device);
    //cout<<debug_tensor<<endl;
    auto base = m_feature_rpn.forward({tensor_image}).toTuple();

    
    //auto features =  base->elements()[0].toTensor();
    std::vector<torch::Tensor> level_features;
    // for(auto i=0;i<5;i++) {
    //     level_features.push_back(base->elements()[i].toTensor());
    // }
    torch::Tensor p2_feature = base->elements()[0].toTensor();
    torch::Tensor p3_feature = base->elements()[1].toTensor();
    torch::Tensor p4_feature = base->elements()[2].toTensor();
    torch::Tensor p5_feature = base->elements()[3].toTensor();
    torch::Tensor p6_feature = base->elements()[4].toTensor();

    torch::Tensor rpn_deltas = base->elements()[5].toTensor().view({-1, 4});
    torch::Tensor rpn_scores = base->elements()[6].toTensor().view({-1, 1});

    //cout<<"p2_feature"<<endl;
    //cout<<p2_feature<<endl;
    // cout<<"rpn_deltas "<<endl;
    // cout<<rpn_deltas<<endl;
    // cout<<"rpn_scores "<<endl;
    // cout<<rpn_scores<<endl;
    std::vector<cv::Size> feat_sizes = {
        {p2_feature.size(3), p2_feature.size(2)},
        {p3_feature.size(3), p3_feature.size(2)},
        {p4_feature.size(3), p4_feature.size(2)},
        {p5_feature.size(3), p5_feature.size(2)},
        {p6_feature.size(3), p6_feature.size(2)}
    };
    generate_all_anchors(feat_sizes);
    auto proposals = mlvl_nms(rpn_deltas, rpn_scores);
    //cout<<proposals;
    auto target_levels = target_lvls(proposals);
    //cout<<target_levels;

    std::vector<torch::Tensor> level_roi_fetures;
    std::vector<torch::Tensor> rearragned_rois_list;
    for(long i=0; i <level_count_;i++){
        auto inds = torch::nonzero(target_levels == i).squeeze(1);
        if(inds.size(0) > 0) {
            // std::cout<<"inds dim "<<inds.dim()<<std::endl;
            // for(auto ii=0; ii<inds.dim(); ii++) {
            //     std::cout<<" "<<inds.size(ii)<<" ";
            // } 
            // std::cout<<std::endl;
            auto level_proposals = proposals.index_select(0, inds);

            auto x1 = level_proposals.select(1, 0).clamp(0, input_size_.width-1).view({-1,1});
            auto y1 = level_proposals.select(1, 1).clamp(0, input_size_.height-1).view({-1,1});
            auto x2 = level_proposals.select(1, 2).clamp(0, input_size_.width-1).view({-1,1});
            auto y2 = level_proposals.select(1, 3).clamp(0, input_size_.height-1).view({-1,1});
            auto label = level_proposals.select(1, 4).view({-1,1});

            level_proposals =  torch::cat({label, x1, y1, x2, y2}, 1);

            rearragned_rois_list.push_back(level_proposals);      
            torch::Tensor level_roi_feature = torch::zeros({level_proposals.size(0), 256, 7, 7}).toType(torch::kFloat).to(m_device);

            if(i==0) {
                //cout<<"level_proposals 0 "<<level_proposals.size(0)<<endl;
                //cout<<level_proposals<<endl;
                ROIAlign_forwardV1(p2_feature, level_proposals, 7, 7, 0.25, 2, level_roi_feature);
                //cout<<"level 0 roi feature"<<endl;
                //cout<<level_roi_feature<<endl;
            }
            else if(i==1){
                //cout<<"level_proposals 1 "<<level_proposals.size(0)<<endl;
                ROIAlign_forwardV1(p3_feature, level_proposals,  7, 7, 0.125, 2, level_roi_feature);
            }
            else if(i==2){
                //cout<<"level_proposals 2 "<<level_proposals.size(0)<<endl;
                ROIAlign_forwardV1(p4_feature, level_proposals, 7, 7,  0.0625, 2, level_roi_feature);
            }
            else if(i==3){
                //cout<<"level_proposals 3 "<<level_proposals.size(0)<<endl;
                ROIAlign_forwardV1(p5_feature, level_proposals,  7, 7, 0.03125, 2, level_roi_feature);
            }
            level_roi_fetures.push_back(level_roi_feature);
        }
    }
    torch::Tensor roi_features = torch::cat(level_roi_fetures, 0);
    //stage2 box refine
    torch::Tensor rearragned_roi = torch::cat(rearragned_rois_list, 0);
    auto boxhead_output = m_boxhead.forward({roi_features}).toTuple();
    torch::Tensor multi_scores = boxhead_output->elements()[0].toTensor();
    torch::Tensor bbox_pred = boxhead_output->elements()[1].toTensor();

    torch::Tensor refine_pred_boxes = bbox_transfrom_torch2(rearragned_roi, bbox_pred);
    int  num_classes  = multi_scores.size(1);
    auto multi_boxes  = refine_pred_boxes.view({refine_pred_boxes.size(0), -1, 4});
    // nms
    vector<torch::Tensor> final_instances; 
    for(int i=1; i<num_classes; i++) 
    {
        auto instances_cls_probs =  multi_scores.select(1, i);
        auto instances_pred_bboxes = multi_boxes.select(1, i);
        auto ids = torch::nonzero(instances_cls_probs > m_score_threshold).squeeze(1);
        if(ids.size(0) > 0) {
            auto selected_instances_cls_probs = instances_cls_probs.index_select(0, ids).view({-1, 1});
            auto selected_instances_pred_bboxes = instances_pred_bboxes.index_select(0, ids).view({-1, 4}).toType(torch::kFloat);
            auto instances = torch::cat({selected_instances_pred_bboxes, selected_instances_cls_probs}, 1);
            auto nms_ids = nms_cuda(instances, final_nms_threshold_);
            auto final_cls_instances = instances.index_select(0, nms_ids);
            auto label = torch::tensor({i}).expand({final_cls_instances.size(0)}).view({-1,1}).toType(torch::kFloat).to(m_device);
            final_cls_instances = torch::cat({final_cls_instances, label}, 1);
            final_instances.push_back(final_cls_instances);
        }
    }
    torch::Tensor final_preds = torch::cat({final_instances}, 0);
    final_preds = final_preds.to(torch::kCPU);
    float *pdata = final_preds.data<float>();

    auto clamp = [](float x, float min, float max)->float{
        if (x > max)
            return max;
        if (x < min)
            return min;
        return x;
    };
    
    for(auto i=0; i<final_preds.size(0);i++)
    {
        float x1 = clamp(pdata[6*i+0], 0, input_size_.width-1)/input_size_.width;
        float y1 = clamp(pdata[6*i+1], 0, input_size_.height-1)/input_size_.height;
        float x2 = clamp(pdata[6*i+2], 0, input_size_.width-1)/input_size_.width;
        float y2 = clamp(pdata[6*i+3], 0, input_size_.height-1)/input_size_.height; 
        float score = pdata[6*i+4];
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
    
}