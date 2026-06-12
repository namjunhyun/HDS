# HDS 시스템 아키텍처

**HDS (Hybrid DeepSEE)** — Human-in-the-Loop Neuro-Symbolic Drift Anticipation for Visual SLAM
Junhyun Nam, Wonse Jo — Incheon National University

---

## 1. 한눈에 보기

HDS는 Visual SLAM의 **드리프트(궤적 누적오차) 위험을 사전에** 추정한다. 세 신호를 융합:

```
r̃ = clip( DS  +  Δ_sym  +  W_HW · G ,  0, 1 )     # THRESHOLD=0.6 초과 시 ALERT
```

| 구성요소 | 입력 | 처리 | 의미 |
|---|---|---|---|
| **DS** (DeepSEE) | IMU/SLAM 16채널 RTS + PSD(특징점 공간분포) | TS2Vec(16→64) → Timesformer + Cross-Attention 회귀 | 신경망 드리프트 위험 (선행연구) |
| **G** (Hardware Guardrail) | 카메라 프레임 | 밝기·엔트로피·블러·특징점수 가중합 | 영상 품질 저하 (반응형) |
| **Δ_sym** (Symbolic) | 운영자 자연어 | Claude(`claude-sonnet-4-6`) → `risk_delta` JSON | 경로 사전지식 (선행형, 본 연구 기여) |

**기여 구분**: DeepSEE는 선행연구(SEESys). 본 논문의 신규 기여는 **G + Symbolic 레이어(휴먼-인-더-루프)** 와 그 융합.

---

## 2. 데이터 흐름 (멀티-프로세스 / 멀티-언어)

```
G1 로봇 (Jetson, ROS2 Foxy) ── RealSense D435i ──┐
                                                  ▼ (CycloneDDS, 100Mbps)
외부 PC (ROS2 Jazzy, Ubuntu 24.04)
  ├ image_transport republish ×2  → /local/color, /local/depth
  ├ ORB-SLAM3 (RGBD, C++ 수정본)   → PSD + RTS  → /dev/shm/{psd_buffer.bin, rts_buffer.bin}
  ├ hds_ros_bridge.py (Py3.12)     → IMU + 프레임 → /dev/shm/{imu_buffer.npy, latest_frame.jpg}
  └ hds_g1_local.py  (Py3.9 conda) → DS·G·Δ_sym 융합 → HDS 점수 오버레이
```

IPC는 `/dev/shm` 램디스크 바이너리 파일로 처리 — C++/Python 및 ROS 버전 차이를 디커플링.

### RTS 16채널 (ORB-SLAM3 → rts_buffer.bin, shape (30,16))
`[0]Brightness [1]Contrast [2]Entropy [3]Laplacian [4]AvgMPDepth [5]VarMPDepth
 [6]PrePOKeyMapLoss [7]PostPOOutlier [8]MatchedInlier [9-11]DX,DY,DZ [12-14]Yaw,Pitch,Roll
 [15]local_visual_BA_Err`
PSD: shape (4,3,96,128) — 4프레임 × {특징점응답, 맵포인트마스크, 깊이} × 96×128 그리드.

---

## 3. 핵심 파일

### 배포 파이프라인 (논문 대상)
| 파일 | 역할 |
|---|---|
| `DeepSEE/hds_g1_local.py` | 메인 추론 루프 (DS+G+Symbolic 융합, 오버레이, 로깅) |
| `DeepSEE/hds_ros_bridge.py` | 센서 → /dev/shm |
| `DeepSEE/models/DeepSEEModels.py` | DeepSEE 모델 구조 (Timesformer + CrossAttention) |
| `orbslam3_ros2/rgbd/rgbd-slam-node.cpp` | RTS 16채널 + PSD 추출 (학습=SenseTime+합성과 동일 정규화) |
| `DeepSEE/make_ds_calib.py` | DS 출력 → [0,1] 고정 캘리브레이터 생성 |

### 모델 가중치 (`DeepSEE/runs/`)
| 파일 | 내용 | 출처 |
|---|---|---|
| `SupervisedFinetune_1_best_model.pth` | DeepSEE 본체 | **SenseTime+합성 학습, `May18_14-44-57` run fold1** |
| `pretrained_model.pkl` | TS2Vec 인코더 | 동상 |
| `rts_norm_{lower,upper,std}.npy` | RTS 입력 정규화 | ⚠ **`May13` run** (모델과 다른 run — 동일성 검증 필요) |
| `ds_calib.npz` | DS 출력 캘리브 (lo=2.563, hi=4.344) | May18 Y_est p1/p99 |

### 오프라인 평가 (`DeepSEE/eval/`)
| 파일 | 역할 |
|---|---|
| `make_annotations.py` | 운영자 어노테이션 생성 (이미지 단서만, GT 누수 없음) |
| `score_annotations.py` | 텍스트 → Claude(temp=0) risk_delta, 캐싱 |
| `eval_offline.py` | DS/DS+G/DS+Sym/Full ablation, EWR·precision·lead·AUC, in-domain+CI |

---

## 4. 데이터셋 (중요)

| 데이터셋 | 용도 | 비고 |
|---|---|---|
| **SenseTime (실측)** | **학습** + in-domain 평가 | 6-fold CV (A0–A7, B0–B7) |
| **합성 환경 9종** | **학습** | LivingRoom/Hall/Lab/Apartment/FireStation×3/AbandonedFactory |
| EuRoC | **zero-shot** 평가 | MAV 비행, Vicon/Leica GT (정밀 lead-time 검증대) |
| TUM-VI / OpenLORIS | **zero-shot** 평가 | 미사용 결과 — 재실행 필요 |

라벨: `Y = log1p(10⁴ · clip(RelativeError, 0.001, 0.02))`, 범위 [2.40, 5.30]. **높을수록 드리프트 큼.**

---

## 5. 점수 산출 세부

- **DS 정규화**: 배포=고정 캘리브(`ds_calib.npz`, 결정론·무지연). 평가=causal 정규화(과거만, 누수 없음).
  (구버전의 rolling min-max+EMA는 상대값·지연 문제로 폐기.)
- **Δ_sym (gap-based)**: `min(max(0, 0.65 − DS), risk_delta)` — DS가 이미 높으면 가산 안 함(이중계산 방지).
- **G 기여**: `max(0, G − 0.20) · 0.3` — 임계 초과분만.
- **LLM 폴백**: API 불가 시 키워드 규칙(실시간 안전 보장).

자세한 평가 방법론과 결과는 [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) 참조.
