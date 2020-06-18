#include "classifier.h"
#include "torch/script.h"

TorchClassifier::TorchClassifier(int device_id) :
    device_(torch::kCUDA, device_id) {}

void TorchClassifier::Init(const std::string& model_prefix) {
    mean_ = {0.485, 0.456, 0.406};
    std_ = {0.225, 0.225, 0.225};
    cv_size_ = cv::Size(224, 224);
    module_ = torch::jit::load(model_prefix);
    module_.to(device_);
}

void TorchClassifier::BatchPredict(const std::vector<cv::Mat>& images,
                                       std::vector<std::vector<float> >& results) {
    torch::NoGradGuard no_grad_guard;

    int num = images.size();
    torch::Tensor tensor_image = torch::empty({num, cv_size_.height, cv_size_.width, 3});
    for (int i = 0; i < num; i++) {
        cv::Mat img_resized;
        cv::resize(images[i], img_resized, cv_size_);
        cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);
        tensor_image.select(0, i) = torch::from_blob(img_resized.data,
            {cv_size_.height, cv_size_.width, 3}, torch::kByte);
    }
    tensor_image = tensor_image.to(device_);
    tensor_image = tensor_image.permute({0, 3, 1, 2});
    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);
    for (int i = 0; i < num; i++) {
        tensor_image[i][0] = tensor_image[i][0].sub_(mean_[0]).div_(std_[0]);
        tensor_image[i][1] = tensor_image[i][1].sub_(mean_[1]).div_(std_[1]);
        tensor_image[i][2] = tensor_image[i][2].sub_(mean_[2]).div_(std_[2]);
    }

    torch::Tensor output = module_.forward({tensor_image}).toTensor();
    output = output.to(torch::kCPU);
    int o_dim = output.size(1);
    float* ptr = output.data<float>();
    for (int i = 0; i < num; i++) {
        float* begin = ptr + i * o_dim;
        vector<float> single_pred(begin, begin + o_dim);
        results.emplace_back(single_pred);
    }
}

void TorchClassifier::Classify(const std::vector<cv::Mat>& images,
                                   std::vector<std::vector<float> >& results,
                                   int batch_limit) {
    int batch_size = images.size();
    for (int i = 0; i < batch_size; i += batch_limit) {
        int endpoint = (i + batch_limit > batch_size) ? batch_size : (i + batch_limit);
        vector<cv::Mat> imgs_batch(images.begin() + i, images.begin() + endpoint);
        vector<vector<float> > sub_results;
        BatchPredict(imgs_batch, sub_results);
        results.insert(results.end(), sub_results.begin(), sub_results.end());
    }
}
