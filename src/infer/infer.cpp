#include "infer.h"


/*----------------------------------Detected Object ---------------------------*/
BoundingBox::BoundingBox() {
    cx = 0.0f;
    cy = 0.0f;
    w = 0.0f;
    h = 0.0f;
}
BoundingBox::BoundingBox(const float cx, const float cy, const float w, const float h) : cx(cx), cy(cy), w(w), h(h) {
}
void BoundingBox::Scale(const float scale_factor) {
    w *= scale_factor;
    h *= scale_factor;
}
DetectedObject::DetectedObject() {
    id = -1;
    score = -1.0f;
}
DetectedObject::DetectedObject(const BoundingBox& box, const int id, const float score) : DetectedObject(
    box.cx, box.cy, box.w, box.h, id, score) {
}
DetectedObject::DetectedObject(const float cx, const float cy, const float w, const float h, const int id,
                               const float score) : BoundingBox(cx, cy, w, h),
                                                    id(id), score(score) {
}
DetectedObject::DetectedObject(const float cx, const float cy, const float w, const float h, const int id,
                               const float score, const cv::Mat& mask_) : BoundingBox(cx, cy, w, h),
                                                    id(id), score(score) {
    mask = mask_.clone();
}

ObjectDetector::ObjectDetector() = default;

ObjectDetector::~ObjectDetector() = default;


Classifier::Classifier() = default;

Classifier::~Classifier() = default;