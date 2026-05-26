#ifndef __RGBD_SLAM_NODE_HPP__
#define __RGBD_SLAM_NODE_HPP__

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <mutex>
#include <deque>
#include <array>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include "System.h"
#include "Frame.h"
#include "Map.h"
#include "Tracking.h"

#include "utility.hpp"

// PSD 해상도: H=96, W=128  (학습 데이터 기준)
static constexpr int PSD_H = 96;
static constexpr int PSD_W = 128;
static constexpr int PSD_C = 3;   // keypoints, inliers, mappoints
static constexpr int PSD_FRAMES = 4;

// RTS 피처 16개 (훈련 데이터와 동일):
// Brightness, Contrast, Entropy, Laplacian, AvgMPDepth, VarMPDepth,
// PrePOKeyMapLoss, PostPOOutlier, MatchedInlier,
// DX, DY, DZ, Yaw, Pitch, Roll, local_visual_BA_Err
static constexpr int RTS_FEATURES = 16;
static constexpr int RTS_WINDOW   = 30;

class RgbdSlamNode : public rclcpp::Node
{
public:
    RgbdSlamNode(ORB_SLAM3::System* pSLAM);

    ~RgbdSlamNode();

private:
    using ImageMsg = sensor_msgs::msg::Image;

    void GrabRGB(const ImageMsg::SharedPtr msgRGB);
    void GrabDepth(const ImageMsg::SharedPtr msgD);

    void ExtractAndWriteFeatures(const cv::Mat& colorImg,
                                 const std::vector<cv::KeyPoint>& trackedKps,
                                 const std::vector<ORB_SLAM3::MapPoint*>& trackedMPs,
                                 int trackState,
                                 const Sophus::SE3f& Tcw,
                                 int prePOMatches,
                                 float localBAErr);

    ORB_SLAM3::System* m_SLAM;

    std::mutex color_mutex;
    ImageMsg::SharedPtr latest_color;

    rclcpp::Subscription<ImageMsg>::SharedPtr rgb_sub;
    rclcpp::Subscription<ImageMsg>::SharedPtr depth_sub;

    std::deque<std::array<float, PSD_C * PSD_H * PSD_W>> psd_buf_;
    std::deque<std::array<float, RTS_FEATURES>> rts_buf_;

    Sophus::SE3f last_Tcw_;
    bool has_last_pose_ = false;
};

#endif
