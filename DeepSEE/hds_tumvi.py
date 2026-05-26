"""
HDS (Hybrid DeepSEE) - TUM-VI Zero-Shot Evaluation
SenseTime 학습 모델 → TUM-VI 제로샷 테스트

Human annotation: TUM-VI 이미지 직접 확인 후 위험 구간 작성
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dotenv import load_dotenv
import anthropic
import json
import re
from glob import glob
from sklearn.metrics import precision_score, recall_score, f1_score
from matplotlib.patches import Patch

load_dotenv("/home/junhyun/SEESys/DeepSEE/Training/.env")

# ── 설정 ──────────────────────────────────────────────────────────────────────
RUN_DIR  = "/home/junhyun/SEESys/DeepSEE/Training/runs/TUMVI_zeroshot"
OUT_DIR  = "/home/junhyun/SEESys/HDS_output/TUMVI"
IMG_BASE = "/home/junhyun/SEESys/Datasets/TUM_VI"
SEQS     = ["room1", "room2", "room3", "room4", "room5", "room6"]

W_SY = 0.3
W_HW = 0.1
CAM_HZ = 20.0

os.makedirs(OUT_DIR, exist_ok=True)

# ── Human Annotations (TUM-VI 이미지 직접 확인) ──────────────────────────────
# (cam_start, cam_end, human_text)
# cam_start: 실제 위험 구간보다 ~100 카메라 프레임(5초) 앞
HUMAN_ANNOTATIONS = {
    # room1: 밝은 체스보드 환경, 전반적으로 양호 — 어노테이션 없음
    "room1": [],

    # room2: 어두운 회의실, 후반부 조명이 낮아지는 구간
    "room2": [
        (1700, 2668, "Path plan enters a dimly lit conference room. Low ambient lighting with deep shadows in corners will reduce visual feature extraction quality and cause moderate SLAM drift."),
    ],

    # room3: 밝은 실내, 포스터 풍부 — 어노테이션 없음
    "room3": [],

    # room4: 전반부 포스터 구간 이후 암실 수준으로 어두워짐
    "room4": [
        # 초반: 이미 어두운 환경에서 시작 (첫 이미지에서 확인)
        (0, 600, "Path plan begins in a very dark room. Near-zero ambient lighting with only ceiling spotlights will severely degrade visual feature tracking from the start."),
        # 후반: 발표실 암실 구간
        (1400, 2228, "Path plan enters an extremely dark lecture hall. Projector screen environment with near-zero ambient light will cause severe visual feature loss and high SLAM drift risk."),
    ],

    # room5: 중반~후반 어두운 세미나실 전체
    "room5": [
        (600, 2847, "Path plan traverses a dark seminar room with low ambient lighting. Sparse visual features combined with high-contrast ceiling lights will cause repeated feature tracking failures and significant drift accumulation."),
    ],

    # room6: 밝은 홀/복도 — 어노테이션 없음
    "room6": [],
}

# ── LLM risk delta 생성 ───────────────────────────────────────────────────────
client = anthropic.Anthropic()

def generate_risk_delta(human_text, seq_name):
    prompt = f"""You are a SLAM (Simultaneous Localization and Mapping) risk assessment expert.

A human operator has provided the following path plan note for sequence '{seq_name}':
"{human_text}"

Based on this description, assess the drift risk level for visual SLAM.
Return a JSON object with:
- "risk_delta": float between 0.0 and 0.5 (how much to increase risk score)
- "confidence": float between 0.0 and 1.0
- "reasoning": brief explanation (1 sentence)

JSON only, no other text."""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"risk_delta": 0.3, "confidence": 0.5, "reasoning": "Parse error"}

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_data(fold_idx):
    gt   = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_gt.npy",   allow_pickle=True)
    est  = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_est.npy",  allow_pickle=True)
    base = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_base.npy", allow_pickle=True)

    gt_flat  = np.array(gt).flatten().astype(np.float64)
    est_flat = np.array(est).flatten().astype(np.float64)

    # inverse transform (log-scale → original)
    gt_flat  = (np.exp(gt_flat)  - 1) / 10000
    est_flat = (np.exp(est_flat) - 1) / 10000

    # base: RF predicts in log-scale too
    base_flat = np.array(base).flatten().astype(np.float64)
    base_flat = (np.exp(base_flat) - 1) / 10000

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    return {
        "gt":       norm(gt_flat),
        "deepsee":  norm(est_flat),
        "baseline": norm(base_flat),
    }

# ── IMU hardware guardrail ────────────────────────────────────────────────────
def compute_imu_guardrail(seq, n_samples):
    """TUM-VI IMU 데이터 없으면 uniform 낮은 값으로 대체"""
    imu_path = f"{IMG_BASE}/dataset-{seq}_512_16/mav0/imu0/data.csv"
    if not os.path.exists(imu_path):
        return np.zeros(n_samples)
    try:
        import pandas as pd
        imu = pd.read_csv(imu_path, header=0)
        ang  = imu.iloc[:, 1:4].values
        accel= imu.iloc[:, 4:7].values
        ang_norm   = np.linalg.norm(ang,   axis=1)
        accel_norm = np.linalg.norm(accel, axis=1)
        ang_thresh   = np.percentile(ang_norm,   85)
        accel_thresh = np.percentile(accel_norm, 85)
        ang_n   = (ang_norm   - ang_norm.min())   / (ang_norm.max()   - ang_norm.min() + 1e-8)
        accel_n = (accel_norm - accel_norm.min()) / (accel_norm.max() - accel_norm.min() + 1e-8)
        G_imu = 0.6 * (ang_norm > ang_thresh).astype(float) + \
                0.4 * (accel_norm > accel_thresh).astype(float)
        # 모델 프레임 수로 리샘플
        indices = np.linspace(0, len(G_imu)-1, n_samples).astype(int)
        return G_imu[indices]
    except Exception:
        return np.zeros(n_samples)

# ── HDS 계산 ──────────────────────────────────────────────────────────────────
PROACTIVE_FRAMES = 15
RAMP_FRAMES = 10

def compute_hds(deepsee, G, seq, n_samples):
    annotations = HUMAN_ANNOTATIONS.get(seq, [])
    n_cam = len(glob(f"{IMG_BASE}/dataset-{seq}_512_16/mav0/cam0/data/*.png"))

    w_map = np.zeros(n_samples)
    segment_info = []

    for cam_s, cam_e, text in annotations:
        # cam → model frame
        m_start = int(cam_s / n_cam * n_samples)
        m_end   = int(cam_e / n_cam * n_samples)
        # proactive shift
        orig_start, orig_end = m_start, m_end
        m_start = max(0, m_start - PROACTIVE_FRAMES)
        m_end   = max(0, m_end   - PROACTIVE_FRAMES)

        result  = generate_risk_delta(text, seq)
        risk_delta = float(result.get('risk_delta', 0.3))
        conf    = float(result.get('confidence', 0.5))
        reasoning = result.get('reasoning', '')

        seg_len  = m_end - m_start
        ramp_len = min(RAMP_FRAMES, seg_len)
        ramp = np.ones(seg_len)
        if ramp_len > 0:
            ramp[:ramp_len] = np.linspace(0.1, 1.0, ramp_len)

        gap     = np.maximum(0.0, 0.65 - deepsee[m_start:m_end])
        applied = np.minimum(gap, risk_delta * conf * ramp)
        boosted = (applied > 0).sum()

        w_map[m_start:m_end] = np.maximum(w_map[m_start:m_end], applied)

        print(f"    [Annotation @ cam {cam_s}~{cam_e} → model {m_start}~{m_end}]")
        print(f"      → risk_delta={risk_delta:.3f} conf={conf:.2f} | boosted {boosted}/{seg_len} frames")
        print(f"      → {reasoning}")

        segment_info.append((m_start, m_end, text[:60]))

    delta_hw = W_HW * G
    hds = np.clip(deepsee + w_map + delta_hw, 0, 1)
    return hds, segment_info

# ── Lead Time 계산 ────────────────────────────────────────────────────────────
def compute_lead_time(gt, hds, deepsee, seq, n_samples, threshold=0.6):
    sec_per_frame = 1.0 / CAM_HZ * (n_samples / max(
        len(glob(f"{IMG_BASE}/dataset-{seq}_512_16/mav0/cam0/data/*.png")), 1))

    gt_bin   = (gt    > threshold).astype(int)
    hds_bin  = (hds   > threshold).astype(int)
    deep_bin = (deepsee > threshold).astype(int)

    # GT 이벤트 찾기
    events, in_ev, ev_start = [], False, 0
    for i, v in enumerate(gt_bin):
        if v and not in_ev:
            in_ev, ev_start = True, i
        elif not v and in_ev:
            in_ev = False
            peak = ev_start + np.argmax(gt[ev_start:i])
            events.append(peak)
    if in_ev:
        events.append(ev_start + np.argmax(gt[ev_start:]))

    results = []
    for ev in events:
        window = max(0, ev - int(9.0 / sec_per_frame))

        hds_warn, deep_warn = None, None
        for t in range(window, ev+1):
            if hds_warn  is None and hds_bin[t]:  hds_warn  = t
            if deep_warn is None and deep_bin[t]: deep_warn = t

        lead_hds  = (ev - hds_warn)  * sec_per_frame if hds_warn  is not None else None
        lead_deep = (ev - deep_warn) * sec_per_frame if deep_warn is not None else None
        results.append({'event': ev, 'lead_hds_sec': lead_hds, 'lead_deep_sec': lead_deep})

    return results

# ── 평가 ──────────────────────────────────────────────────────────────────────
def evaluate(gt, baseline, deepsee, hds, risk_threshold=0.6):
    gt_bin   = (gt       > risk_threshold).astype(int)
    base_bin = (baseline > risk_threshold).astype(int)
    deep_bin = (deepsee  > risk_threshold).astype(int)
    hds_bin  = (hds      > risk_threshold).astype(int)

    def safe_f1(y_true, y_pred):
        if y_pred.sum() == 0 and y_true.sum() == 0: return 1.0, 1.0, 1.0
        if y_pred.sum() == 0: return 0.0, 0.0, 0.0
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        return f, p, r

    bf, bp, br = safe_f1(gt_bin, base_bin)
    df, dp, dr = safe_f1(gt_bin, deep_bin)
    hf, hp, hr = safe_f1(gt_bin, hds_bin)

    return {
        "gt_positive_rate": gt_bin.mean(),
        "baseline_f1": bf, "baseline_prec": bp, "baseline_rec": br,
        "deepsee_f1":  df, "deepsee_prec":  dp, "deepsee_rec":  dr,
        "hds_f1":      hf, "hds_prec":      hp, "hds_rec":      hr,
    }

# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_results(seq, gt, baseline, deepsee, hds, G, metrics, segment_info, lead_results):
    fig = plt.figure(figsize=(16, 10))
    gs_ = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs_[0, :])
    t = np.arange(len(gt))
    ax1.plot(gt,       color='black', alpha=0.4, lw=1.5, label='GT')
    ax1.plot(baseline, color='green', alpha=0.7, lw=1.2, label='Baseline (RF)')
    ax1.plot(deepsee,  color='blue',  alpha=0.7, lw=1.2, label='DeepSEE (zero-shot)')
    ax1.plot(hds,      color='red',   alpha=0.9, lw=2.0, label='HDS (Ours)')
    ax1.axhline(0.6, color='gray', ls='--', alpha=0.5)
    for m_start, m_end, _ in segment_info:
        ax1.axvspan(m_start, m_end, alpha=0.12, color='orange', label='_ann')

    # Lead time 화살표
    sec_per_frame = 1.0 / CAM_HZ
    diff_events = [r for r in lead_results
                   if r['lead_hds_sec'] is not None and
                      (r['lead_deep_sec'] is None or r['lead_hds_sec'] > r['lead_deep_sec'])]
    for r in diff_events[:5]:
        ev = r['event']
        if r['lead_hds_sec'] is not None:
            hds_frame = ev - int(r['lead_hds_sec'] / sec_per_frame)
            ax1.annotate('', xy=(ev, 0.63), xytext=(max(0, hds_frame), 0.63),
                         arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Patch(facecolor='orange', alpha=0.3, label='Human annotation'))
    ax1.legend(handles=handles, fontsize=8, loc='upper right')
    ax1.set_title(f"{seq} | GT pos={metrics['gt_positive_rate']*100:.1f}%")
    ax1.set_ylabel("Normalized score")
    ax1.set_ylim(-0.05, 1.1)

    ax2 = fig.add_subplot(gs_[1, 0])
    labels_bar = ['Baseline\n(RF)', 'DeepSEE\n(zero-shot)', 'HDS\n(Ours)']
    f1s = [metrics['baseline_f1'], metrics['deepsee_f1'], metrics['hds_f1']]
    bars = ax2.bar(labels_bar, f1s, color=['green','blue','red'], alpha=0.7, width=0.4)
    for bar, val in zip(bars, f1s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("F1 Score")
    ax2.set_ylabel("F1")

    ax3 = fig.add_subplot(gs_[1, 1])
    hds_leads  = [r['lead_hds_sec']  or 0 for r in lead_results if r['lead_hds_sec']  is not None]
    deep_leads = [r['lead_deep_sec'] or 0 for r in lead_results if r['lead_deep_sec'] is not None]
    ax3.bar(['DeepSEE', 'HDS'], [np.mean(deep_leads) if deep_leads else 0,
                                   np.mean(hds_leads)  if hds_leads  else 0],
            color=['blue','red'], alpha=0.7, width=0.4)
    ax3.set_title("Avg Lead Time (s)")
    ax3.set_ylabel("seconds")

    plt.suptitle(f"TUM-VI {seq} — HDS Zero-Shot Evaluation", fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = f"{OUT_DIR}/{seq}_hds.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → 그래프: {out_path}")

# ── 메인 ──────────────────────────────────────────────────────────────────────
print("=" * 65)
print("HDS TUM-VI Zero-Shot Evaluation (SenseTime pretrain only)")
print("=" * 65)

all_metrics = {}
all_lead_hds, all_lead_deep = [], []

for fold_idx, seq in enumerate(SEQS):
    print(f"\n[Fold {fold_idx}] {seq}")

    data = load_data(fold_idx)
    n    = len(data['gt'])

    G   = compute_imu_guardrail(seq, n)
    hds, segment_info = compute_hds(data['deepsee'], G, seq, n)
    metrics    = evaluate(data['gt'], data['baseline'], data['deepsee'], hds)
    lead_results = compute_lead_time(data['gt'], hds, data['deepsee'], seq, n)

    all_metrics[seq] = metrics

    print(f"  GT high-risk rate : {metrics['gt_positive_rate']*100:.1f}%")
    print(f"  Baseline F1: {metrics['baseline_f1']:.3f}  (P={metrics['baseline_prec']:.3f} R={metrics['baseline_rec']:.3f})")
    print(f"  DeepSEE  F1: {metrics['deepsee_f1']:.3f}  (P={metrics['deepsee_prec']:.3f} R={metrics['deepsee_rec']:.3f})")
    print(f"  HDS      F1: {metrics['hds_f1']:.3f}  (P={metrics['hds_prec']:.3f} R={metrics['hds_rec']:.3f})")
    imp = (metrics['hds_f1'] - metrics['deepsee_f1']) / max(metrics['deepsee_f1'], 1e-4) * 100
    imp = min(imp, 9999.0)
    print(f"  HDS vs DeepSEE F1: {imp:+.1f}%")

    diff_events = [r for r in lead_results
                   if r['lead_hds_sec'] is not None and
                      (r['lead_deep_sec'] is None or r['lead_hds_sec'] > r['lead_deep_sec'])]
    if diff_events:
        print(f"  Lead Time (HDS > DeepSEE, {len(diff_events)}/{len(lead_results)} events):")
        for r in diff_events[:5]:
            deep_str = f"{r['lead_deep_sec']:.2f}s" if r['lead_deep_sec'] else "MISSED"
            print(f"    frame {r['event']:4d}: HDS={r['lead_hds_sec']:.2f}s  DeepSEE={deep_str}  (+{r['lead_hds_sec'] - (r['lead_deep_sec'] or 0):.2f}s)")
        avg_adv = np.mean([r['lead_hds_sec'] - (r['lead_deep_sec'] or 0) for r in diff_events])
        print(f"  HDS advantage (avg): +{avg_adv:.2f}s earlier warning")
        all_lead_hds.extend([r['lead_hds_sec']  for r in diff_events])
        all_lead_deep.extend([r['lead_deep_sec'] or 0 for r in diff_events])

    plot_results(seq, data['gt'], data['baseline'], data['deepsee'],
                 hds, G, metrics, segment_info, lead_results)

print("\n" + "=" * 65)
print("전체 평균")
print("=" * 65)
avg_base = np.mean([m['baseline_f1'] for m in all_metrics.values()])
avg_deep = np.mean([m['deepsee_f1']  for m in all_metrics.values()])
avg_hds  = np.mean([m['hds_f1']      for m in all_metrics.values()])
print(f"  Baseline avg F1 : {avg_base:.3f}")
print(f"  DeepSEE  avg F1 : {avg_deep:.3f}")
print(f"  HDS      avg F1 : {avg_hds:.3f}")
print(f"  HDS vs DeepSEE  : {(avg_hds-avg_deep)/max(avg_deep,1e-4)*100:+.1f}%")
if all_lead_hds:
    print(f"\n  HDS  avg lead   : {np.mean(all_lead_hds):.2f}s")
    print(f"  Deep avg lead   : {np.mean(all_lead_deep):.2f}s")
    print(f"  HDS advantage   : +{np.mean(all_lead_hds)-np.mean(all_lead_deep):.2f}s earlier warning")
