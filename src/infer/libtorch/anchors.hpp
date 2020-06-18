//
// Created by dl on 2019/12/3.
//

#ifndef ANCHORS_H
#define ANCHORS_H
#include "torch/torch.h"
#include <vector>
#include <opencv2/opencv.hpp>

namespace easycv {

torch::Tensor generate_total_anchors(torch::Tensor &cell_anchors_, cv::Size featmap_size, float stride);
/*
class AnchorGenerator {
public:
    AnchorGenerator(std::vector<float> anchor_scales, std::vector<float> anchor_ratios, float base_size, float ctr, bool scale_major=false)
    { 
        anchor_scales_ = torch::tensor(anchor_scales).toType(torch::kFloat);
        anchor_ratios_  = torch::tensor(anchor_ratios).toType(torch::kFloat);
        base_size_ = base_size;
        ctr_ = ctr;
        scale_major_ = scale_major;
    }

    ~AnchorGenerator()= default;

    void generate_base_anchors(){
        float w = base_size_;
        float h = base_size_;
        float x_ctr, y_ctr;
        if(ctr_<0) {
             x_ctr = 0.5 * (w - 1);
             y_ctr = 0.5 * (h - 1);
        }
        else{
             x_ctr = ctr_;
             y_ctr = ctr_;
        }

        torch::Tensor h_ratios = torch::sqrt(anchor_ratios_);
        torch::Tensor w_ratios = torch::tensor({1.0}).toType(torch::kFloat).expand_as(h_ratios).div(h_ratios);

        torch::Tensor ws;
        torch::Tensor hs;
        if(scale_major_){
            //TO DO
        }
        else{
            ws = w_ratios.expand({anchor_scales_.size(0), w_ratios.size(0)}).mul(
                anchor_scales_.view({-1,1}).expand({anchor_scales_.size(0), w_ratios.size(0)}).mul(base_size_)
                ).view({-1});
            hs = h_ratios.expand({anchor_scales_.size(0), h_ratios.size(0)}).mul(
                anchor_scales_.view({-1,1}).expand({anchor_scales_.size(0), h_ratios.size(0)}).mul(base_size_)
                ).view({-1});
        }

        // cell_anchors_ = torch::stack(
        //     {
        //         ws.sub(1.0).mul(-0.5).add(x_ctr),
        //         hs.sub(1.0).mul(-0.5).add(y_ctr),
        //         ws.sub(1.0).mul(0.5).add(x_ctr),
        //         hs.sub(1.0).mul(0.5).add(y_ctr)
        //     }, 1
        // ).round();
    }


public:
    //torch::Tensor cell_anchors_;

private:
    torch::Tensor anchor_scales_;
    torch::Tensor anchor_ratios_;
    float base_size_;
    float ctr_;
    bool scale_major_;
};
*/

}
#endif 