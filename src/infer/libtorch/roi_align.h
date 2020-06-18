#ifndef _ROI_ALIGN_H_
#define _ROI_ALIGN_H_

//#include <ATen/ATen.h>
#include "torch/torch.h"
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


int ROIAlign_forwardV1(torch::Tensor features, torch::Tensor rois, int pooled_height,
                       int pooled_width, float spatial_scale, int sample_num,
                       torch::Tensor output);  


torch::Tensor ROIAlign_forwardV2(const torch::Tensor& input,
                                     const torch::Tensor& rois,
                                     const float spatial_scale,
                                     const int pooled_height,
                                     const int pooled_width,
                                     const int sampling_ratio, bool aligned);


                            


#endif //_ROI_ALIGN_H_