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

    int state        = m_SLAM->GetTrackingState();
    auto trackedKps  = m_SLAM->GetTrackedKeyPointsUn();
    auto trackedMPs  = m_SLAM->GetTrackedMapPoints();
    int prePOMatches = m_SLAM->GetPrePOMatches();
    float localBAErr = m_SLAM->GetLocalBAError();

    ExtractAndWriteFeatures(cv_ptrRGB->image, trackedKps, trackedMPs, state, Tcw, prePOMatches, localBAErr);
}

// ── RTS 16개 + PSD 추출 및 /dev/shm 기록 ─────────────────────────────────────
// 훈련 데이터(EuRoC)와 동일한 피처 순서 및 정규화:
// [0] Brightness/160   [1] Contrast(raw)  [2] Entropy/8     [3] Laplacian/90
// [4] AvgMPDepth*1.2   [5] VarMPDepth     [6] PrePOKeyMapLoss (prePOMatches)
// [7] PostPOOutlier(count) [8] MatchedInlier/400
// [9] DX [10] DY [11] DZ [12] Yaw [13] Pitch [14] Roll  [15] local_visual_BA_Err
void RgbdSlamNode::ExtractAndWriteFeatures(
    const cv::Mat& colorImg,
    const std::vector<cv::KeyPoint>& trackedKps,
    const std::vector<ORB_SLAM3::MapPoint*>& trackedMPs,
    int trackState,
    const Sophus::SE3f& Tcw,
    int prePOMatches,
    float localBAErr)
{
    const float IMG_W = (float)colorImg.cols;
    const float IMG_H = (float)colorImg.rows;
    const float sx    = PSD_W / IMG_W;
    const float sy    = PSD_H / IMG_H;

    // ── PSD 프레임 생성 ───────────────────────────────────────────────────
    std::array<float, PSD_C * PSD_H * PSD_W> psd_frame{};

    if (trackState == 2) {
        size_t n = std::min(trackedKps.size(), trackedMPs.size());
        for (size_t i = 0; i < n; i++) {
            int px = (int)(trackedKps[i].pt.x * sx);
            int py = (int)(trackedKps[i].pt.y * sy);
            if (px < 0 || px >= PSD_W || py < 0 || py >= PSD_H) continue;

            int idx = py * PSD_W + px;
            psd_frame[0 * PSD_H * PSD_W + idx] =
                std::min(trackedKps[i].response / 1000.0f, 1.0f);

            if (trackedMPs[i]) {
                psd_frame[1 * PSD_H * PSD_W + idx] = 1.0f;
                auto pos = trackedMPs[i]->GetWorldPos();
                float depth = pos.norm();
                psd_frame[2 * PSD_H * PSD_W + idx] = std::min(depth / 10.0f, 1.0f);
            }
        }
    }

    psd_buf_.push_back(psd_frame);
    if ((int)psd_buf_.size() > PSD_FRAMES)
        psd_buf_.pop_front();

    if ((int)psd_buf_.size() == PSD_FRAMES) {
        FILE* f = fopen("/dev/shm/psd_buffer.bin", "wb");
        if (f) {
            for (auto& fr : psd_buf_)
                fwrite(fr.data(), sizeof(float), fr.size(), f);
            fclose(f);
        }
    }

    // ── RTS 16개 피처 ─────────────────────────────────────────────────────
    std::array<float, RTS_FEATURES> rts{};

    cv::Mat gray;
    cv::cvtColor(colorImg, gray, cv::COLOR_BGR2GRAY);

    // [0] Brightness: mean of gray [0,255] / 160
    rts[0] = (float)cv::mean(gray)[0] / 160.0f;

    // [1] Contrast: std of gray [0,255] (raw, no normalization)
    cv::Scalar mean_v, std_v;
    cv::meanStdDev(gray, mean_v, std_v);
    rts[1] = (float)std_v[0];

    // [2] Entropy: Shannon entropy / 8
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
    rts[2] = entropy / 8.0f;

    // [3] Laplacian variance / 90
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar lap_mean, lap_std;
    cv::meanStdDev(lap, lap_mean, lap_std);
    rts[3] = (float)(lap_std[0] * lap_std[0]) / 90.0f;

    // AvgMPDepth, VarMPDepth, n_inliers, n_outliers
    int n_inliers = 0, n_outliers = 0;
    float depth_sum = 0.0f, depth_sq_sum = 0.0f;
    if (trackState == 2) {
        size_t n = std::min(trackedKps.size(), trackedMPs.size());
        for (size_t i = 0; i < n; i++) {
            if (trackedMPs[i]) {
                float d = trackedMPs[i]->GetWorldPos().norm();
                depth_sum    += d;
                depth_sq_sum += d * d;
                n_inliers++;
            } else {
                n_outliers++;
            }
        }
    }
    float avg_depth = (n_inliers > 0) ? depth_sum / n_inliers : 0.0f;
    float var_depth = (n_inliers > 1) ?
        (depth_sq_sum / n_inliers - avg_depth * avg_depth) : 0.0f;

    // [4] AvgMPDepth * 1.2
    rts[4] = avg_depth * 1.2f;

    // [5] VarMPDepth (raw variance)
    rts[5] = var_depth;

    // [6] PrePOKeyMapLoss: map-matched keypoints before pose optimization
    rts[6] = (float)prePOMatches;

    // [7] PostPOOutlier: raw outlier count
    rts[7] = (float)n_outliers;

    // [8] MatchedInlier / 400
    rts[8] = (float)n_inliers / 400.0f;

    // [9-14] DX,DY,DZ,Yaw,Pitch,Roll: relative pose (raw, radians)
    if (trackState == 2 && has_last_pose_) {
        Sophus::SE3f rel = last_Tcw_.inverse() * Tcw;
        auto t = rel.translation();
        rts[9]  = t.x();
        rts[10] = t.y();
        rts[11] = t.z();

        auto q = rel.unit_quaternion();
        float sinr_cosp = 2.0f * (q.w() * q.x() + q.y() * q.z());
        float cosr_cosp = 1.0f - 2.0f * (q.x() * q.x() + q.y() * q.y());
        float roll  = std::atan2(sinr_cosp, cosr_cosp);
        float sinp  = 2.0f * (q.w() * q.y() - q.z() * q.x());
        float pitch = std::asin(std::max(-1.0f, std::min(1.0f, sinp)));
        float siny_cosp = 2.0f * (q.w() * q.z() + q.x() * q.y());
        float cosy_cosp = 1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z());
        float yaw   = std::atan2(siny_cosp, cosy_cosp);

        rts[12] = yaw;
        rts[13] = pitch;
        rts[14] = roll;
    }

    // [15] local_visual_BA_Err: mean chi2 from last local bundle adjustment
    rts[15] = localBAErr;

    if (trackState == 2)
        last_Tcw_ = Tcw;
    has_last_pose_ = (trackState == 2);

    rts_buf_.push_back(rts);
    if ((int)rts_buf_.size() > RTS_WINDOW)
        rts_buf_.pop_front();

    // /dev/shm/rts_buffer.bin: shape (30, 16) float32
    if ((int)rts_buf_.size() == RTS_WINDOW) {
        FILE* f = fopen("/dev/shm/rts_buffer.bin", "wb");
        if (f) {
            for (auto& row : rts_buf_)
                fwrite(row.data(), sizeof(float), RTS_FEATURES, f);
            fclose(f);
        }
    }
}
