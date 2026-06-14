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

> **학습/평가 데이터**: DeepSEE는 **SenseTime(실측) + 합성 환경 9종**으로 학습됨.
> EuRoC / TUM-VI / OpenLORIS 는 **zero-shot** 평가 대상(학습 미사용).

---

## Repository Structure

```
HDS_repo/
├─ DeepSEE/
│  ├─ hds_g1_local.py          # 메인 실시간 추론 (DS+G+Symbolic 융합)
│  ├─ hds_ros_bridge.py        # 센서 → /dev/shm
│  ├─ make_ds_calib.py         # DS 출력 고정 캘리브 생성 → runs/ds_calib.npz
│  ├─ models/ · ts2vec/ · runs/
│  └─ eval/                    # 오프라인 평가 파이프라인 (논문용)
├─ orbslam3_ros2/rgbd/         # ORB-SLAM3 RGBD 수정본 (RTS 16채널 + PSD 추출)
├─ experiments/g1_closedloop/  # E2 클로즈드루프 (궤적평가·완화정책)
└─ docs/                       # 아키텍처 + 실험 설계 + human study 키트
```

## Evaluation

- **문서**: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) · [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md)
- **오프라인 평가** (`DeepSEE/eval/`): EuRoC zero-shot lead-time / EWR / precision ablation + in-domain 검증 + 부트스트랩 CI — `python3 DeepSEE/eval/eval_offline.py`
- **게이트 (LLM vs 규칙)**: `gate_llm_vs_rules.py` → 논문화 키트 [`docs/human_study/`](docs/human_study/)
- **E2 클로즈드루프**: `experiments/g1_closedloop/` (`traj_eval.py --selftest`, `mitigation_policy.py --selftest`)

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

학습된 `SupervisedFinetune_1_best_model.pth`(SenseTime+합성 학습), `pretrained_model.pkl`(TS2Vec),
`rts_norm_*.npy`(입력 정규화)를 `DeepSEE/runs/` 에 배치하세요. DS 출력 캘리브는
`python3 DeepSEE/make_ds_calib.py` 로 `runs/ds_calib.npz` 생성.

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

- **DS** (DeepSEE): IMU + PSD → neural drift prediction, 고정 캘리브(`runs/ds_calib.npz`)로 [0,1] 사상
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
