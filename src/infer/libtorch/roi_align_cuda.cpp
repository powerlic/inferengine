#include "roi_align.h"
#include <ATen/ATen.h>
#include <cmath>
#include <vector>

int ROIAlignForwardLaucher(const torch::Tensor features, const torch::Tensor rois,
                           const float spatial_scale, const int sample_num,
                           const int channels, const int height,
                           const int width, const int num_rois,
                           const int pooled_height, const int pooled_width,
                           torch::Tensor output);


torch::Tensor ROIAlignForwardV2Laucher(const torch::Tensor& input,
                                    const torch::Tensor& rois,
                                    const float spatial_scale,
                                    const int pooled_height,
                                    const int pooled_width,
                                    const int sampling_ratio, bool aligned);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int ROIAlign_forwardV1(torch::Tensor features, torch::Tensor rois, int pooled_height,
                       int pooled_width, float spatial_scale, int sample_num,
                       torch::Tensor output) {
  CHECK_INPUT(features);
  CHECK_INPUT(rois);
  CHECK_INPUT(output);
  at::DeviceGuard guard(features.device());

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 5) {
    printf("wrong roi size\n");
    return 0;
  }

  int num_channels = features.size(1);
  int data_height = features.size(2);
  int data_width = features.size(3);

  ROIAlignForwardLaucher(features, rois, spatial_scale, sample_num,
                         num_channels, data_height, data_width, num_rois,
                         pooled_height, pooled_width, output);

  return 1;
}

// Interface for Python
torch::Tensor ROIAlign_forwardV2(const torch::Tensor& input,
                                     const torch::Tensor& rois,
                                     const float spatial_scale,
                                     const int pooled_height,
                                     const int pooled_width,
                                     const int sampling_ratio, bool aligned) {
  if (input.type().is_cuda()) {
    return ROIAlignForwardV2Laucher(input, rois, spatial_scale, pooled_height,
                                    pooled_width, sampling_ratio, aligned);

  }
}


