#ifndef  _NMS_H_
#define _NMS_H_

#include "torch/torch.h"


#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

at::Tensor nms_cpu(const at::Tensor& dets, const float threshold);

at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);

// at::Tensor nms(const at::Tensor& dets, const float threshold) {
//   CHECK_CUDA(dets);
//   if (dets.numel() == 0)
//     return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
//   return nms_cuda(dets, threshold);
// }

#endif //_NMS_CUDA_H_