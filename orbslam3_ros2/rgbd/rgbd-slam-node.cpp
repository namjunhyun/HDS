#include "rgbd-slam-node.hpp"
#include <opencv2/core/core.hpp>
#include <cmath>
#include <cstdio>
#include <numeric>

RgbdSlamNode::RgbdSlamNode(ORB_SLAM3::System* pSLAM)
:   Node("ORB_SLAM3_ROS2"),
    m_SLAM(pSLAM)
{
    auto sensor_qos = rclcpp::SensorDataQoS();

    rgb_sub = this->create_subscription<ImageMsg>(
        "camera/color/image_raw", sensor_qos,
        std::bind(&RgbdSlamNode::GrabRGB, this, std::placeholders::_1));

    depth_sub = this->create_subscription<ImageMsg>(
        "camera/depth/image_rect_raw", sensor_qos,
        std::bind(&RgbdSlamNode::GrabDepth, this, std::placeholders::_1));
}

RgbdSlamNode::~RgbdSlamNode()
{
    m_SLAM->Shutdown();
    m_SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

void RgbdSlamNode::GrabRGB(const ImageMsg::SharedPtr msgRGB)
{
    std::lock_guard<std::mutex> lock(color_mutex);
    latest_color = msgRGB;
}

void RgbdSlamNode::GrabDepth(const ImageMsg::SharedPtr msgD)
{
    ImageMsg::SharedPtr color_msg;
    {
        std::lock_guard<std::mutex> lock(color_mutex);
        color_msg = latest_color;
    }

    if (!color_msg) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "No color image yet");
        return;
    }

    double color_t = Utility::StampToSec(color_msg->header.stamp);
    double depth_t = Utility::StampToSec(msgD->header.stamp);
    double dt = std::abs(color_t - depth_t);
    if (dt > 0.2) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Timestamp gap too large: %.3fs", dt);
        return;
    }
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Processing frame, dt=%.3fs", dt);

    cv_bridge::CvImageConstPtr cv_ptrRGB;
    cv_bridge::CvImageConstPtr cv_ptrD;

    try {
        cv_ptrRGB = cv_bridge::toCvShare(color_msg);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    try {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    Sophus::SE3f Tcw = m_SLAM->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image, depth_t);

    int state     = m_SLAM->GetTrackingState();
    auto trackedKps = m_SLAM->GetTrackedKeyPointsUn();
    auto trackedMPs = m_SLAM->GetTrackedMapPoints();

    ExtractAndWriteFeatures(cv_ptrRGB->image, trackedKps, trackedMPs, state, Tcw);
}

// ── RTS + PSD 추출 및 /dev/shm 기록 ──────────────────────────────────────────
void RgbdSlamNode::ExtractAndWriteFeatures(
    const cv::Mat& colorImg,
    const std::vector<cv::KeyPoint>& trackedKps,
    const std::vector<ORB_SLAM3::MapPoint*>& trackedMPs,
    int trackState,
    const Sophus::SE3f& Tcw)
{
    const float IMG_W = (float)colorImg.cols;
    const float IMG_H = (float)colorImg.rows;
    const float sx    = PSD_W / IMG_W;
    const float sy    = PSD_H / IMG_H;

    // ── PSD 프레임 생성 ───────────────────────────────────────────────────
    std::array<float, PSD_C * PSD_H * PSD_W> psd_frame{};
    // layout: [channel * PSD_H * PSD_W + y * PSD_W + x]
    // ch0 = keypoints (response), ch1 = inliers (0/1), ch2 = mappoints (depth)

    if (trackState == 2) {  // TRACKING_OK
        size_t n = std::min(trackedKps.size(), trackedMPs.size());
        for (size_t i = 0; i < n; i++) {
            int px = (int)(trackedKps[i].pt.x * sx);
            int py = (int)(trackedKps[i].pt.y * sy);
            if (px < 0 || px >= PSD_W || py < 0 || py >= PSD_H) continue;

            int idx = py * PSD_W + px;

            // ch0: keypoint response (normalized /1000)
            psd_frame[0 * PSD_H * PSD_W + idx] =
                std::min(trackedKps[i].response / 1000.0f, 1.0f);

            // ch1: inlier (1 if map point exists)
            if (trackedMPs[i]) {
                psd_frame[1 * PSD_H * PSD_W + idx] = 1.0f;

                // ch2: map point depth (normalized /10m)
                auto pos = trackedMPs[i]->GetWorldPos();
                float depth = pos.norm();
                psd_frame[2 * PSD_H * PSD_W + idx] =
                    std::min(depth / 10.0f, 1.0f);
            }
        }
    }

    // 4-frame 버퍼 업데이트
    psd_buf_.push_back(psd_frame);
    if ((int)psd_buf_.size() > PSD_FRAMES)
        psd_buf_.pop_front();

    // /dev/shm/psd_buffer.bin 기록: shape (4, 3, 96, 128) float32
    if ((int)psd_buf_.size() == PSD_FRAMES) {
        FILE* f = fopen("/dev/shm/psd_buffer.bin", "wb");
        if (f) {
            for (auto& fr : psd_buf_)
                fwrite(fr.data(), sizeof(float), fr.size(), f);
            fclose(f);
        }
    }

    // ── RTS 피처 계산 ─────────────────────────────────────────────────────
    std::array<float, RTS_FEATURES> rts{};

    // 이미지 기반 피처
    cv::Mat gray;
    cv::cvtColor(colorImg, gray, cv::COLOR_BGR2GRAY);
    cv::Mat gray_f;
    gray.convertTo(gray_f, CV_32F, 1.0 / 255.0);

    // 0: Brightness
    rts[0] = (float)cv::mean(gray_f)[0];

    // 1: Contrast (RMS)
    cv::Scalar mean_v, std_v;
    cv::meanStdDev(gray_f, mean_v, std_v);
    rts[1] = (float)std_v[0];

    // 2: Entropy (Shannon)
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    float total = (float)(gray.rows * gray.cols);
    float entropy = 0.0f;
    for (int b = 0; b < histSize; b++) {
        float p = hist.at<float>(b) / total;
        if (p > 0) entropy -= p * std::log2(p);
    }
    rts[2] = entropy / 8.0f;  // normalize to [0,1]

    // 3: Laplacian variance
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar lap_mean, lap_std;
    cv::meanStdDev(lap, lap_mean, lap_std);
    rts[3] = std::min((float)(lap_std[0] * lap_std[0]) / 1000.0f, 1.0f);

    // SLAM tracking 피처
    int n_inliers = 0;
    int n_outliers = 0;
    if (trackState == 2) {
        size_t n = std::min(trackedKps.size(), trackedMPs.size());
        for (size_t i = 0; i < n; i++) {
            if (trackedMPs[i]) n_inliers++;
            else n_outliers++;
        }
    }
    // 4: MatchedInliers (normalized by 500)
    rts[4] = std::min(n_inliers / 500.0f, 1.0f);
    // 5: Outliers ratio
    rts[5] = (n_inliers + n_outliers > 0) ?
              (float)n_outliers / (n_inliers + n_outliers) : 0.0f;

    // 6,7,8: RelativeTranslation (x,y,z)
    // 9,10: RelativeRotation (yaw, pitch) - simplified from SE3
    if (trackState == 2 && has_last_pose_) {
        Sophus::SE3f rel = last_Tcw_.inverse() * Tcw;
        auto t = rel.translation();
        rts[6]  = std::max(-1.0f, std::min(1.0f, t.x()));
        rts[7]  = std::max(-1.0f, std::min(1.0f, t.y()));
        rts[8]  = std::max(-1.0f, std::min(1.0f, t.z()));
        auto q  = rel.unit_quaternion();
        // roll, pitch, yaw from quaternion (simplified)
        float siny_cosp = 2.0f * (q.w() * q.z() + q.x() * q.y());
        float cosy_cosp = 1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z());
        rts[9]  = std::atan2(siny_cosp, cosy_cosp);  // yaw
        float sinp = 2.0f * (q.w() * q.y() - q.z() * q.x());
        rts[10] = std::asin(std::max(-1.0f, std::min(1.0f, sinp)));  // pitch
    }

    if (trackState == 2)
        last_Tcw_ = Tcw;
    has_last_pose_ = (trackState == 2);

    // RTS 버퍼 업데이트
    rts_buf_.push_back(rts);
    if ((int)rts_buf_.size() > RTS_WINDOW)
        rts_buf_.pop_front();

    // /dev/shm/rts_buffer.bin 기록: shape (30, 11) float32
    if ((int)rts_buf_.size() == RTS_WINDOW) {
        FILE* f = fopen("/dev/shm/rts_buffer.bin", "wb");
        if (f) {
            for (auto& row : rts_buf_)
                fwrite(row.data(), sizeof(float), RTS_FEATURES, f);
            fclose(f);
        }
    }
}
