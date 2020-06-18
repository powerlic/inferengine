#pragma once
#ifndef _INFER_H_
#define _INFER_H_

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

/**********************************************************************************************/ /**
 * @class   BoundingBox
 *
 * @brief   一个目标的外接矩形
 **************************************************************************************************/
class BoundingBox {
public:
    BoundingBox();
    BoundingBox(float cx, float cy, float w, float h);

    float cx; //外接矩形的中心点 x 坐标
    float cy; //外接矩形的中心点 y 坐标
    float w; //外接矩形的宽
    float h; //外接矩形的高
    float x1() const {
        return cx - w / 2;
    } // 获取外接矩形的左上角 x 坐标
    float y1() const {
        return cy - h / 2;
    } // 获取外接矩形的左上角 y 坐标
    float x2() const {
        return cx + w / 2;
    } // 获取外接矩形的右下角 x 坐标
    float y2() const {
        return cy + h / 2;
    } // 获取外接矩形的右下角 y 坐标

    /**********************************************************************************************/ /**
     * @fn  void BoundingBox::Scale(float scale_factor);
     *
     * @brief   对外接矩形进行缩放
     *
     * @param   scale_factor    缩放因子
     **************************************************************************************************/
    void Scale(float scale_factor);
};

/**********************************************************************************************/ /**
 * @class   DetectedObject
 *
 * @brief   一个检出目标，包含目标的外接矩形、类别 ID 和置信度
 **************************************************************************************************/
class DetectedObject : public BoundingBox {
public:
    DetectedObject();
    explicit DetectedObject(const BoundingBox& box, int id = -1, float score = -1.0f);
    DetectedObject(float cx, float cy, float w, float h, int id, float score);
    DetectedObject(float cx, float cy, float w, float h, int id, float score, const cv::Mat &mask);

    int id; // 目标的类别 ID
    float score; // 目标的置信度
    cv::Mat mask;
};

/**********************************************************************************************//**
 * @class   ObjectDetector
 *
 * @brief   目标检测基类，定义各类目标检测算法的通用接口
 **************************************************************************************************/
class ObjectDetector {
public:
    ObjectDetector();
    virtual ~ObjectDetector();
    virtual void Init(const std::string& model_prefix) = 0;
    virtual void Detect(const cv::Mat& image, std::vector<DetectedObject>& detected_objects) = 0;
};

/**********************************************************************************************//**
 * @class   Classifier
 *
 * @brief   分类器基类， 定义分类器基本接口
 **************************************************************************************************/
class Classifier {
public:
    Classifier();
    virtual ~Classifier();
    virtual void Init(const std::string& model_prefix) = 0;
    virtual void Classify(const std::vector<cv::Mat>& images,
                          std::vector<std::vector<float> >& results,
                          int batch_limit = 32) = 0;
};

#endif  // _TUNI_INFER_H_
