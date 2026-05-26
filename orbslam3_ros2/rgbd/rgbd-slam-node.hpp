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

// RTS 피처 수 (논문 기준 11개)
static constexpr int RTS_FEATURES = 11;
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

    // RTS + PSD 추출 및 /dev/shm 기록
    void ExtractAndWriteFeatures(const cv::Mat& colorImg,
                                 const std::vector<cv::KeyPoint>& trackedKps,
                                 const std::vector<ORB_SLAM3::MapPoint*>& trackedMPs,
                                 int trackState,
                                 const Sophus::SE3f& Tcw);

    ORB_SLAM3::System* m_SLAM;

    std::mutex color_mutex;
    ImageMsg::SharedPtr latest_color;

    rclcpp::Subscription<ImageMsg>::SharedPtr rgb_sub;
    rclcpp::Subscription<ImageMsg>::SharedPtr depth_sub;

    // PSD 4-frame 슬라이딩 버퍼: [frame][channel][H][W]
    std::deque<std::array<float, PSD_C * PSD_H * PSD_W>> psd_buf_;

    // RTS 30-frame 슬라이딩 버퍼
    std::deque<std::array<float, RTS_FEATURES>> rts_buf_;

    // 직전 pose (relative pose 계산용)
    Sophus::SE3f last_Tcw_;
    bool has_last_pose_ = false;
};

#endif
