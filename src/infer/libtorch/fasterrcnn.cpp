#include "fasterrcnn.h"

#include "roi_align.h"
#include "nms.h"
#include "torch/script.h"
#include <cuda_runtime_api.h>

#include <string>

#ifdef TIME_LOG
#include "utils/timer.hpp"
#endif

#include "utils/logging.hpp"


using namespace easycv;
using namespace std;

std::vector<float> get_debug_input()
{
    std::ifstream infile("/home/chengli12/git/mmdetection/tools/faster_rcnn.txt");
    std::string line;
    vector<float> data_list;
    while (std::getline(infile, line))
    {
        data_list.push_back(atof(line.c_str()));
    }
    infile.close();
    return data_list;
}

FasterRcnn::FasterRcnn(int device_id):device_(torch::kCUDA, device_id)
{
    LOG(INFO) << "faster rcnn "<<device_;
    //cudaSetDevice(device_id);
    //cv::cuda::setDevice(device_id);
}

void FasterRcnn::Init(const std::string& model_prefix)
{

    LOG(INFO)<<"load faster rcnn from "<<model_prefix;
    std::string base_model = model_prefix + "/fasterrcnn_backbone.pt";

    base_net_ = torch::jit::load(base_model);
    base_net_.to(device_);
    base_net_.eval();
    LOG(INFO)<<"load basenet done";


    std::string shared_head_model = model_prefix + "/fasterrcnn_shared_head.pt";
    shared_head_ = torch::jit::load(shared_head_model);
    shared_head_.to(device_);
    shared_head_.eval();
    LOG(INFO)<<"load shared head done";


    std::string bbox_head_model = model_prefix + "/fasterrcnn_bbox_head.pt";
    bbox_head_ = torch::jit::load(bbox_head_model);
    bbox_head_.to(device_);
    bbox_head_.eval();
    LOG(INFO)<<"load bbox head done";

    // std::string box_head_model = model_prefix + "/faster_rcnn_boxhead.pt";
    // box_head_ = torch::jit::load(box_head_model);
    // box_head_.to(device_);
    // box_head_.eval();
    // LOG(INFO)<<"load head done";

    // generate_cell_anchors();
    // m_score_threshold = 0.5;

    //params_.input_size = cv::Size(1248, 800);
    params_.input_size = cv::Size(128, 384);
    params_.input_means = {102.9801, 115.9465, 122.7717};
    params_.cell_anchors = torch::tensor({-15.,   -3.,   30.,   18.,
                                          -37.,  -15.,   52.,   30.,
                                          -83.,  -37.,   98.,   52.,
                                         -173.,  -83.,  188.,   98.,
                                         -354., -173.,  369.,  188.,
                                           -8.,   -8.,   23.,   23.,
                                          -24.,  -24.,   39.,   39.,
                                          -56.,  -56.,   71.,   71.,
                                         -120., -120.,  135.,  135.,
                                         -248., -248.,  263.,  263.,
                                           -3.,  -15.,   18.,   30.,
                                          -15.,  -37.,   30.,   52.,
                                          -37.,  -83.,   52.,   98.,
                                          -83., -173.,   98.,  188.,
                                         -173., -354.,  188.,  369.}).view({-1, 4}).to(device_);
    params_.target_means = torch::tensor({0., 0., 0., 0.}).to(device_);
    params_.target_stds = torch::tensor({1., 1., 1., 1.}).to(device_);

    params_.bbox_head_target_stds = torch::tensor({0.1, 0.1, 0.2, 0.2}).to(device_);
    params_.bbox_head_target_means = torch::tensor({0., 0., 0., 0.}).to(device_);

}

void FasterRcnn::generate_cell_anchors()
{


}

void FasterRcnn::Detect(const cv::Mat& image, std::vector<DetectedObject>& detected_objects)
{

#ifdef TIME_LOG
    Timer total_timer;
    total_timer.tic();
    Timer timer;
    timer.tic();
#endif
    torch::NoGradGuard no_grad_guard;
   
    gpu_raw_.upload(image, cv_cuda_stream_);
    gpu_raw_.convertTo(gpu_raw_float_, CV_32FC3, 1, 0, cv_cuda_stream_);
    cv::cuda::resize(gpu_raw_float_, gpu_resize_, params_.input_size, 0, 0, cv::INTER_LINEAR, cv_cuda_stream_);
    cv::Mat input_img;
    gpu_resize_.download(input_img, cv_cuda_stream_);
    torch::Tensor tensor_image = torch::from_blob(input_img.data, {1, params_.input_size.height, params_.input_size.width, 3}, torch::kFloat);
    tensor_image = tensor_image.to(device_);
    tensor_image = tensor_image.permute({0, 3, 1, 2});

    tensor_image[0][0].sub_(params_.input_means[0]);
    tensor_image[0][1].sub_(params_.input_means[1]);
    tensor_image[0][2].sub_(params_.input_means[2]);
    

    // vector<float> img_data = get_debug_input();
    // torch::Tensor tensor_image = torch::tensor(img_data).reshape({1, 3, 768, 1344}).to(torch::kFloat);
    // tensor_image = tensor_image.to(device_);

#ifdef TIME_LOG
    timer.toc("preprocess ");
#endif
    auto base = base_net_.forward({tensor_image}).toTuple();
    torch::Tensor feat =  base->elements()[0].toTensor();
    torch::Tensor rpn_deltas = base->elements()[1].toTensor();
    torch::Tensor rpn_scores = base->elements()[2].toTensor().view({-1, 1});

    // cout<<"rpn_scores"<<endl;
    // cout<<rpn_scores<<endl;

    // ofstream out_f("rpn_deltas.txt");
    // torch::Tensor p = rpn_deltas.to(torch::kCPU);
    // float *ps = p.data<float>();
    // out_f.setf(ios::fixed, ios::floatfield);  // 设定为 fixed 模式，以小数点表示浮点数
    // out_f.precision(4);  // 设置精度 2
    // for(int i = 0; i < rpn_deltas.size(0); i++)
    // {
    //     int dim = rpn_deltas.size(1);
    //     for(int j = 0; j < dim; j++)
    //     {
    //         out_f<<ps[dim*i+j]<<" ";
    //         //cout<<ps[dim*i+j]<<" ";
    //     }
    //     out_f<<std::endl;
    // }
    // out_f<<std::endl;
    // out_f.close();


#ifdef TIME_LOG
    timer.tic();
#endif
    cv::Size feat_size(feat.size(3), feat.size(2));
    generate_all_anchors(feat_size);


    // cout<<"all_anchors"<<endl;
    // cout<<all_anchors_<<endl;
    // std::cout<<"feat_size "<<all_anchors_.dim()<<std::endl;
    // for(auto j=0;j<all_anchors_.dim();j++){
    //      std::cout<<all_anchors_.size(j)<<" ";
    // }
    // std::cout<<std::endl;
    torch::Tensor rpn_pred_boxes;
    int clamp_w = params_.input_size.width-1;
    int clamp_h = params_.input_size.height-1;
    if(rpn_scores.size(0)>params_.pre_nms_count)
    {
        auto scores_tuple = torch::sort(rpn_scores, 0, 1);
        auto order_single = std::get<1>(scores_tuple).select(1, 0);
        order_single = order_single.slice(0, 0, params_.pre_nms_count, 1);
        auto deltas  = rpn_deltas.index_select(0, order_single);
        auto anchors = all_anchors_.index_select(0, order_single);
        rpn_scores = rpn_scores.index_select(0, order_single);

        rpn_pred_boxes = bbox_transfrom_torch(anchors, deltas, params_.target_stds, params_.target_means, clamp_w, clamp_h);
    }
    else
    {
        rpn_pred_boxes = bbox_transfrom_torch(all_anchors_, rpn_deltas, params_.target_stds, params_.target_means, clamp_w, clamp_h);
    }

    // cout<<"rpn_pred_boxes "<<endl;
    // cout<<rpn_pred_boxes<<endl;

    auto nms_input = torch::cat({rpn_pred_boxes.to(torch::kFloat), rpn_scores}, 1);

    auto nms_ids = nms_cuda(nms_input, params_.rpn_nms_threshold);
    auto top_nms_output = nms_input.index_select(0, nms_ids);
    if(top_nms_output.size(0)> params_.post_nms_count) {
        top_nms_output = top_nms_output.slice(0, 0, params_.post_nms_count, 1);
    }

    auto x1 = top_nms_output.select(1, 0).clamp(0, params_.input_size.width-1).view({-1,1});
    auto y1 = top_nms_output.select(1, 1).clamp(0, params_.input_size.height-1).view({-1,1});
    auto x2 = top_nms_output.select(1, 2).clamp(0, params_.input_size.width-1).view({-1,1});
    auto y2 = top_nms_output.select(1, 3).clamp(0, params_.input_size.height-1).view({-1,1});
    auto s = top_nms_output.select(1, 4).view({-1,1});

    auto proposals =  torch::cat({s, x1, y1, x2, y2}, 1);
    // cout<<"proposals "<<endl;
    // cout<<proposals<<endl;


    torch::Tensor roi_features = torch::zeros({proposals.size(0), params_.roi_algin_out_channels, 
        params_.roi_align_feat_size, params_.roi_align_feat_size}).toType(torch::kFloat).to(device_);

    ROIAlign_forwardV1(feat, proposals, params_.roi_align_feat_size, params_.roi_align_feat_size, 1./params_.stride, 2, roi_features);

    // cout<<"roi_features "<<endl;
    // cout<<roi_features<<endl;

    // std::cout<<"roi_features "<<roi_features.dim()<<std::endl;
    // for(auto j=0;j<roi_features.dim();j++){
    //      std::cout<<roi_features.size(j)<<" ";
    // }
    // std::cout<<std::endl;

    auto shared_roi_feats = shared_head_.forward({roi_features}).toTensor();
    // std::cout<<"shared_roi_feats "<<shared_roi_feats.dim()<<std::endl;
    // for(auto j=0;j<shared_roi_feats.dim();j++){
    //      std::cout<<shared_roi_feats.size(j)<<" ";
    // }G
    // std::cout<<std::endl; 

    // cout<<"shared_roi_feats "<<endl;
    // cout<<shared_roi_feats<<endl;


#ifdef TIME_LOG
    timer.toc("roi align ");
#endif

    //stage 2 bbox
    auto bbox_head_output = bbox_head_.forward({shared_roi_feats}).toTuple();
    torch::Tensor bbox_scores = bbox_head_output->elements()[0].toTensor();
    torch::Tensor bbox_deltas = bbox_head_output->elements()[1].toTensor();

    // std::cout<<"bbox_deltas "<<bbox_deltas.dim()<<std::endl;
    // for(auto j=0;j<bbox_deltas.dim();j++){
    //      std::cout<<bbox_deltas.size(j)<<" ";
    // }
    // std::cout<<std::endl; 

    // std::cout<<"top_nms_output "<<top_nms_output.dim()<<std::endl;
    // for(auto j=0;j<top_nms_output.dim();j++){
    //      std::cout<<top_nms_output.size(j)<<" ";
    // }
    // std::cout<<std::endl; 

    torch::Tensor refine_pred_boxes = bbox_transfrom_torch(top_nms_output, bbox_deltas, params_.bbox_head_target_stds, params_.bbox_head_target_means);
    refine_pred_boxes = refine_pred_boxes.view({refine_pred_boxes.size(0), params_.num_classes, 4});
    // nms
    vector<torch::Tensor> final_instances; 
    for(int i=1; i<params_.num_classes; i++) 
    {
        auto instances_cls_probs  =  bbox_scores.select(1, i);
        auto instances_pred_bboxes = refine_pred_boxes.select(1, i);
        auto ids = torch::nonzero(instances_cls_probs > params_.score_threshold).squeeze(1);
    
        if(ids.size(0) > 0) {
            auto selected_instances_cls_probs = instances_cls_probs.index_select(0, ids).view({-1, 1});
            auto selected_instances_pred_bboxes = instances_pred_bboxes.index_select(0, ids).view({-1, 4}).toType(torch::kFloat);
            auto instances = torch::cat({selected_instances_pred_bboxes, selected_instances_cls_probs}, 1);
            auto nms_ids = nms_cuda(instances, params_.final_nms_threshold);
            auto final_cls_instances = instances.index_select(0, nms_ids);
            auto label = torch::tensor({i}).expand({final_cls_instances.size(0)}).view({-1,1}).toType(torch::kFloat).to(device_);
            final_cls_instances = torch::cat({final_cls_instances, label}, 1);
            final_instances.push_back(final_cls_instances);
        }
    }
    torch::Tensor final_preds = torch::cat({final_instances}, 0).to(torch::kCPU);

    auto clamp = [](float x, float min, float max)->float{
        if (x > max)
            return max;
        if (x < min)
            return min;
        return x;
    };

    float* pdata = final_preds.data<float>(); 

    for(auto i=0; i<final_preds.size(0);i++)
    {
        float x1 = clamp(pdata[6*i+0], 0, params_.input_size.width-1)/params_.input_size.width;
        float y1 = clamp(pdata[6*i+1], 0, params_.input_size.height-1)/params_.input_size.height;
        float x2 = clamp(pdata[6*i+2], 0, params_.input_size.width-1)/params_.input_size.width;
        float y2 = clamp(pdata[6*i+3], 0, params_.input_size.height-1)/params_.input_size.height; 
        float score = pdata[6*i+4];
        int label = static_cast<int>(pdata[6*i+5]);
        float w = x2 - x1;
        float h = y2 - y1;
        float c_x = (x1 + x2)/2;
        float c_y = (y1 + y2)/2;
        //pasrse mask
        // string save_name = string("mask")+to_string("i.jpg");
        // cv::imwrite(save_name, mask);
        if(x2>x1 && y2>y1) 
        {
            detected_objects.push_back(DetectedObject(c_x, c_y, w, h, label, score));
        }
    }
#ifdef TIME_LOG
    total_timer.toc("total time:");
#endif
}


void FasterRcnn::generate_all_anchors(cv::Size feature_map_size)
{
    // p2-p6 anchors
    int num_cell_anchors = params_.cell_anchors.size(0);
    auto shift_x1 = torch::arange(0, feature_map_size.width).mul(params_.stride).to(device_);
    auto shift_y1 = torch::arange(0, feature_map_size.height).mul(params_.stride).to(device_);
    auto shift_y11_x11 = torch::meshgrid({shift_y1, shift_x1});
    shift_y11_x11[0] = torch::reshape(shift_y11_x11[0], {-1, 1});
    shift_y11_x11[1] = torch::reshape(shift_y11_x11[1], {-1, 1});
    auto shifts = torch::cat({shift_y11_x11[1], shift_y11_x11[0], shift_y11_x11[1], shift_y11_x11[0]}, 1).contiguous();
    int total = feature_map_size.width*feature_map_size.height;
    all_anchors_ = params_.cell_anchors.expand({total, num_cell_anchors, 4}).
        add(shifts.view({total, 1, 4}).expand({total, num_cell_anchors, 4})).
        view({total * num_cell_anchors, 4});
}

torch::Tensor FasterRcnn::bbox_transfrom_torch(torch::Tensor& anchors, torch::Tensor& deltas, 
    torch::Tensor& stds, torch::Tensor& means, int max_w, int max_h)
{
    auto expand_stds    = stds.repeat({1, static_cast<int>(deltas.size(1)/4)});
    auto expand_means   = means.repeat({1, static_cast<int>(deltas.size(1)/4)});
    // std::cout<<"expand_stds "<<expand_stds.dim()<<std::endl;
    // for(auto j=0;j<expand_stds.dim();j++){
    //      std::cout<<expand_stds.size(j)<<" ";
    // }
    // std::cout<<std::endl;

    auto denorm_deletas = deltas.mul(expand_stds).add(expand_means);
  
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

    if(max_w>0 && max_h>0){
        proposals_x1.clamp_(0, max_w);
        proposals_y1.clamp_(0, max_h);
        proposals_x2.clamp_(0, max_w);
        proposals_y2.clamp_(0, max_h);
    }

    auto proposals = torch::stack({proposals_x1, proposals_y1, proposals_x2, proposals_y2}, -1).view_as(deltas);
    //cout<<"proposals "<<endl;
    //cout<<proposals<<endl;
    return proposals;
}