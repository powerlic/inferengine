#include "anchors.hpp"

namespace easycv {

torch::Tensor generate_total_anchors(torch::Tensor &cell_anchors_, cv::Size featmap_size, float stride){
    int cell_anchors_count = cell_anchors_.size(0);
    auto shift_x1 = torch::arange(0, featmap_size.width).mul(stride);
    auto shift_y1 = torch::arange(0, featmap_size.height).mul(stride);
    auto shift_y11_x11 = torch::meshgrid({shift_y1, shift_x1});
    shift_y11_x11[0] = torch::reshape(shift_y11_x11[0], {-1, 1});
    shift_y11_x11[1] = torch::reshape(shift_y11_x11[1], {-1, 1});
    auto shifts = torch::cat({shift_y11_x11[1], shift_y11_x11[0], shift_y11_x11[1], shift_y11_x11[0]}, 1).contiguous();
    int total = featmap_size.width*featmap_size.height;
    auto anchors = cell_anchors_.expand({total, cell_anchors_count, 4}).
        add(shifts.view({total, 1, 4}).expand({total, cell_anchors_count, 4}));
    return anchors.view({-1, 4});
}

}