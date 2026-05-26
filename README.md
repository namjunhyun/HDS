# HDS: Hybrid DeepSEE — Human-in-the-Loop Neuro-Symbolic Drift Anticipation for Visual SLAM

**Paper**: *Human-in-the-Loop Neuro-Symbolic Drift Anticipation for Reliable Visual SLAM*  
Junhyun Nam, Wonse Jo — Incheon National University, 2025

---

## Overview

HDS integrates three components to proactively estimate V-SLAM drift risk in real time:

| Component | Description |
|-----------|-------------|
| **DeepSEE** | IMU + PSD (Point Spatial Distribution) → neural drift risk `r̂` |
| **Hardware Guardrail G** | Image quality (brightness, entropy, blur, feature count) → visual degradation |
| **Symbolic Layer** | LLM (Claude) translates operator text input → `Δ_sym` |

**Final score:** `r̃ = clip(r̂ + Δ_sym + W_HW · G, 0, 1)`

---

## System Architecture

```
G1 Robot (ROS2 Foxy, Jetson)
  └─ RealSense D435i → /camera/imu, /camera/color/image_raw/compressed
         │
         ▼  (CycloneDDS, 100Mbps Ethernet)
External PC (ROS2 Jazzy, Ubuntu 24.04)
  ├─ image_transport republish ×2  → /local/color/image_raw, /local/depth/image_rect_raw
  ├─ ORB-SLAM3 (RGBD)              → PSD + RTS → /dev/shm/psd_buffer.bin, rts_buffer.bin
  ├─ hds_ros_bridge.py (Python 3.12) → /dev/shm/imu_buffer.npy, latest_frame.jpg
  └─ hds_g1_local.py  (conda deepsee, Python 3.9) → HDS score overlay on camera feed
```

---

## Setup

### 1. Python 환경 (conda)

```bash
conda create -n deepsee python=3.9
conda activate deepsee
pip install -r DeepSEE/requirements.txt
pip install anthropic python-dotenv
```

### 2. API 키 설정

```bash
cp DeepSEE/.env.example DeepSEE/.env
# DeepSEE/.env 파일에 ANTHROPIC_API_KEY 입력
```

### 3. 모델 가중치

학습된 `SupervisedFinetune_1_best_model.pth`를 `DeepSEE/runs/` 폴더에 배치하세요.

### 4. ORB-SLAM3 수정 적용

`orbslam3_ros2/rgbd/` 의 `rgbd-slam-node.cpp`, `rgbd-slam-node.hpp`를  
`ros2_ws/src/orbslam3_ros2/src/rgbd/`에 복사 후 빌드:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws && colcon build
```

---

## 실행 순서

**터미널 1 — 컬러 이미지 변환:**
```bash
source ~/ros2_ws/install/setup.bash
ros2 run image_transport republish \
  --ros-args -p in_transport:=compressed -p out_transport:=raw \
  -r /in/compressed:=/camera/color/image_raw/compressed \
  -r /out:=/local/color/image_raw
```

**터미널 2 — 뎁스 이미지 변환:**
```bash
source ~/ros2_ws/install/setup.bash
ros2 run image_transport republish \
  --ros-args -p in_transport:=compressed -p out_transport:=raw \
  -r /in/compressed:=/camera/depth/image_rect_raw/compressed \
  -r /out:=/local/depth/image_rect_raw
```

**터미널 3 — ORB-SLAM3 (PSD/RTS 자동 추출):**
```bash
~/scripts/run_slam.sh
```

**터미널 4 — HDS ROS Bridge (IMU + 카메라 → /dev/shm):**
```bash
source ~/ros2_ws/install/setup.bash
python3 DeepSEE/hds_ros_bridge.py
```

**터미널 5 — HDS 추론:**
```bash
conda activate deepsee
export DISPLAY=:1
python3 DeepSEE/hds_g1_local.py
```

---

## HDS Score 구성

- **DS** (DeepSEE): IMU + PSD → neural drift prediction, running normalization
- **G**: 이미지 품질 (어두움·블러·특징점 부족) → hardware guardrail
- **hw**: `max(0, G - 0.20) × 0.3` → G가 임계값 초과 시만 기여
- **Symbolic** `Δ_sym`: 운영자 텍스트 입력 → Claude LLM → 위험 가산

---

## Troubleshooting

| 문제 | 해결 |
|------|------|
| ORB-SLAM3 초기화 안 됨 | `nFeatures ≥ 1000` 확인 (yaml) |
| Pangolin 검은 화면 | `export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json` |
| rclpy import 실패 (conda) | hds_ros_bridge.py는 시스템 Python 3.12으로 실행 |
| 네트워크 포화 | compressed topic 파이프라인 확인, 해상도 424×240@15Hz 사용 |
