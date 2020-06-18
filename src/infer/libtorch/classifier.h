#ifndef _CLASSIFIER_H
#define _CLASSIFIER_H

#include "infer.h"
#include "torch/torch.h"

using namespace std;

class TorchClassifier: public Classifier {
public:
    TorchClassifier(int device_id);

    void Init(const std::string& model_prefix);

    void Classify(const std::vector<cv::Mat>& images,
                  std::vector<std::vector<float> >& results,
                  int batch_limit = 32);

protected:
    vector<float> mean_;
    vector<float> std_;
    cv::Size cv_size_;
    torch::Device device_;
    torch::jit::script::Module module_;

    void BatchPredict(const std::vector<cv::Mat>& images,
                      std::vector<std::vector<float> >& results);
};

#endif