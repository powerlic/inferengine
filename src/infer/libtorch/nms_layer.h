#ifndef  _NMS_LAYER_H_
#define _NMS_LAYER_H_

#include "torch/torch.h"

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

class NmsLayer {

public:
    NmsLayer();
    ~NmsLayer();
    at::Tensor nms_cpu(const at::Tensor& dets, const float nms_threshold);
    at::Tensor nms_cuda(const at::Tensor& dets, const float nms_threshold);

private:
    unsigned long long* mask_dev_;
    int max_box_num_;
};

#endif //_NMS_LAYER_H_