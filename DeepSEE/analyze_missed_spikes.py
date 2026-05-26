"""
DeepSEE가 놓친 GT 스파이크 구간 분석
- 각 시퀀스별로 GT spike 중 DeepSEE가 못잡는 구간 출력
- 해당 구간의 cam0 이미지 경로 (시작/중간/끝 3장) 출력
- 그래프 저장
"""

import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob

RUN_DIR  = "/home/junhyun/SEESys/DeepSEE/Training/runs/Apr04_00-06-40_AHRI-Junhyun"
GT_BASE  = "/home/junhyun/SEESys/Datasets/SenseTime/EuRoC"
OUT_DIR  = "/home/junhyun/SEESys/HDS_output/missed_spike_analysis"
SEQS     = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
THRESHOLD = 0.6   # high-risk 판단 임계값

os.makedirs(OUT_DIR, exist_ok=True)

# cam0 이미지 목록 로드
def get_cam_images(seq):
    cam_dir = f"{GT_BASE}/{seq}/mav0/cam0/data"
    imgs = sorted(glob(f"{cam_dir}/*.png"))
    return imgs

# model frame → cam frame 인덱스 매핑 (선형 보간)
def model_to_cam(model_idx, n_model, n_cam):
    return int(round(model_idx / n_model * n_cam))

def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def find_spikes(signal, threshold=0.6, min_gap=5):
    """threshold 넘는 연속 구간을 이벤트로 묶어 반환 (start, end, peak)"""
    above = signal > threshold
    events = []
    in_event = False
    start = 0
    for i, v in enumerate(above):
        if v and not in_event:
            in_event = True
            start = i
        elif not v and in_event:
            in_event = False
            events.append((start, i, np.argmax(signal[start:i]) + start))
    if in_event:
        events.append((start, len(signal), np.argmax(signal[start:]) + start))
    # 너무 짧은 이벤트 제거
    events = [(s, e, p) for s, e, p in events if e - s >= min_gap]
    return events

print("=" * 70)
print("DeepSEE 놓친 GT 스파이크 분석")
print("=" * 70)

for fold_idx, seq in enumerate(SEQS):
    gt_raw   = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_gt.npy",  allow_pickle=True)
    est_raw  = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_est.npy", allow_pickle=True)

    gt   = norm(np.concatenate([np.array(x).flatten() for x in gt_raw]))
    pred = norm(np.concatenate([np.array(x).flatten() for x in est_raw]))
    n    = len(gt)

    cam_imgs = get_cam_images(seq)
    n_cam    = len(cam_imgs)

    # GT 스파이크 이벤트
    gt_events = find_spikes(gt, THRESHOLD)
    # DeepSEE 예측 스파이크 이벤트
    pred_events = find_spikes(pred, THRESHOLD)

    # GT 스파이크 중 DeepSEE가 못잡는 것 (GT 이벤트 구간 내 pred max < threshold)
    missed = []
    for s, e, p in gt_events:
        pred_max_in_zone = pred[s:e].max() if e > s else 0.0
        if pred_max_in_zone < THRESHOLD:
            missed.append((s, e, p, pred_max_in_zone))

    print(f"\n[{seq}]  n_model={n}, n_cam={n_cam}")
    print(f"  GT 이벤트 수: {len(gt_events)},  DeepSEE 놓침: {len(missed)}")

    for i, (s, e, p, pred_max) in enumerate(missed):
        # cam 인덱스 계산
        cam_s   = model_to_cam(s, n, n_cam)
        cam_e   = model_to_cam(e, n, n_cam)
        cam_mid = model_to_cam((s + e) // 2, n, n_cam)

        img_s   = cam_imgs[min(cam_s,   n_cam-1)]
        img_mid = cam_imgs[min(cam_mid, n_cam-1)]
        img_e   = cam_imgs[min(cam_e,   n_cam-1)]

        print(f"\n  [놓친 구간 #{i+1}]  model_frame {s}~{e}  (cam {cam_s}~{cam_e})")
        print(f"    GT peak: {gt[p]:.3f},  DeepSEE max in zone: {pred_max:.3f}")
        print(f"    이미지 (시작): {img_s}")
        print(f"    이미지 (중간): {img_mid}")
        print(f"    이미지 (끝):   {img_e}")
        print(f"    → 어노테이션 cam 구간: ({max(0,cam_s-100)}, {cam_e})")
        print(f"      (100프레임=5초 앞당겨서 proactive하게)")

    # 그래프 저장
    fig, ax = plt.subplots(figsize=(14, 4))
    t = np.arange(n)
    ax.plot(t, gt,   color='black',  lw=1.2, label='GT', alpha=0.8)
    ax.plot(t, pred, color='blue',   lw=1.0, label='DeepSEE', alpha=0.7)
    ax.axhline(THRESHOLD, color='gray', ls='--', lw=0.8, alpha=0.5)

    # GT 이벤트 초록 배경
    for s, e, p in gt_events:
        ax.axvspan(s, e, alpha=0.15, color='green')

    # 놓친 구간 빨간 배경
    for s, e, p, _ in missed:
        ax.axvspan(s, e, alpha=0.35, color='red', label='_missed')

    # 더미 핸들
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor='green', alpha=0.3, label='GT spike'))
    handles.append(Patch(facecolor='red',   alpha=0.5, label='Missed by DeepSEE'))
    ax.legend(handles=handles, fontsize=8)
    ax.set_title(f"{seq}  |  GT spikes={len(gt_events)}, Missed={len(missed)}")
    ax.set_xlabel("Model frame")
    ax.set_ylabel("Normalized score")
    ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/{seq}_missed.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → 그래프: {out_path}")

print("\n" + "=" * 70)
print("완료. OUT_DIR:", OUT_DIR)
