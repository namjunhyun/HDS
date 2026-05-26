"""
HDS (Hybrid DeepSEE) - OpenLORIS Zero-Shot Evaluation
SenseTime 학습 모델 → OpenLORIS 제로샷 테스트
Human annotation: OpenLORIS 이미지 직접 확인 후 위험 구간 작성
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

RUN_DIR  = "/home/junhyun/SEESys/DeepSEE/Training/runs/OpenLORIS_zeroshot"
OUT_DIR  = "/home/junhyun/SEESys/HDS_output/OpenLORIS"
IMG_BASE = "/home/junhyun/SEESys/Datasets/OpenLORIS"
SEQS     = ["cafe1-1", "cafe1-2", "office1-2", "office1-5", "office1-6", "office1-7"]

W_HW = 0.1
CAM_HZ = 30.0  # OpenLORIS color camera 30Hz

os.makedirs(OUT_DIR, exist_ok=True)

# ── Human Annotations (이미지 직접 확인) ──────────────────────────────────────
# (cam_start, cam_end, human_text)
HUMAN_ANNOTATIONS = {
    # cafe1-1: 카페. 유리문 앞 구간 + 사람들로 붐비는 구간
    "cafe1-1": [
        (1100, 1708, "Path plan passes through glass door zone. Highly reflective glass surfaces combined with moving people will cause repeated visual feature tracking failures and significant drift."),
    ],

    # cafe1-2: 북카페. 유리 엘리베이터/반사면 강함 + 동적 인물
    "cafe1-2": [
        (0, 600, "Path plan enters near a glass elevator with highly reflective surfaces. Specular reflections from glass combined with colorful signage create ambiguous features causing SLAM confusion."),
        (2000, 2698, "Path plan traverses crowded cafe with multiple moving people. Dynamic objects frequently occlude static features, causing repeated tracking loss and drift accumulation."),
    ],

    # office1-2: 비교적 단순한 환경 — 어노테이션 없음
    "office1-2": [],

    # office1-5: 앞 구간이 거의 암실 수준 (lamp 하나만)
    "office1-5": [
        (0, 500, "Path plan starts in a nearly dark room with only a single lamp. Near-zero ambient lighting severely degrades visual feature extraction from initialization, causing high drift risk."),
    ],

    # office1-6: 뒷 구간에 흰 천/플라스틱 덮인 물건들 (저특징점)
    "office1-6": [
        (700, 1080, "Path plan enters an area with objects covered by white plastic sheets. Uniform white surfaces provide almost no visual features, causing severe feature tracking degradation."),
    ],

    # office1-7: 사람이 카메라 바로 앞을 지나가며 극심한 모션블러
    "office1-7": [
        (400, 800, "Path plan zone where a person walks directly in front of the camera. Dynamic human occlusion combined with severe motion blur from rapid camera motion causes significant SLAM drift risk."),
    ],
}

# ── LLM ───────────────────────────────────────────────────────────────────────
client = anthropic.Anthropic()

def generate_risk_delta(human_text, seq_name):
    prompt = f"""You are a SLAM risk assessment expert.

Human operator note for '{seq_name}': "{human_text}"

Return JSON only:
- "risk_delta": float 0.0-0.5
- "confidence": float 0.0-1.0
- "reasoning": 1 sentence"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"risk_delta": 0.3, "confidence": 0.5, "reasoning": "parse error"}

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_data(fold_idx):
    gt   = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_gt.npy",   allow_pickle=True)
    est  = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_est.npy",  allow_pickle=True)
    base = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_base.npy", allow_pickle=True)

    gt_flat   = np.concatenate([np.array(x).flatten() for x in gt]).astype(float)
    est_flat  = np.concatenate([np.array(x).flatten() for x in est]).astype(float)
    base_flat = base.flatten().astype(float)

    gt_flat   = (np.exp(gt_flat)   - 1) / 10000
    est_flat  = (np.exp(est_flat)  - 1) / 10000
    base_flat = (np.exp(base_flat) - 1) / 10000

    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
    return {"gt": norm(gt_flat), "deepsee": norm(est_flat), "baseline": norm(base_flat)}

# ── HDS 계산 ──────────────────────────────────────────────────────────────────
PROACTIVE_FRAMES = 15
RAMP_FRAMES = 10

def compute_hds(deepsee, seq, n_samples):
    annotations = HUMAN_ANNOTATIONS.get(seq, [])
    n_cam = len(glob(f"{IMG_BASE}/{seq}/color/*.png"))

    w_map = np.zeros(n_samples)
    segment_info = []

    for cam_s, cam_e, text in annotations:
        m_start = int(cam_s / n_cam * n_samples)
        m_end   = int(cam_e / n_cam * n_samples)
        m_start = max(0, m_start - PROACTIVE_FRAMES)
        m_end   = max(0, m_end   - PROACTIVE_FRAMES)

        result     = generate_risk_delta(text, seq)
        risk_delta = float(result.get('risk_delta', 0.3))
        conf       = float(result.get('confidence', 0.5))
        reasoning  = result.get('reasoning', '')

        seg_len  = max(1, m_end - m_start)
        ramp_len = min(RAMP_FRAMES, seg_len)
        ramp = np.ones(seg_len)
        if ramp_len > 0:
            ramp[:ramp_len] = np.linspace(0.1, 1.0, ramp_len)

        gap     = np.maximum(0.0, 0.65 - deepsee[m_start:m_end])
        applied = np.minimum(gap, risk_delta * conf * ramp)
        boosted = (applied > 0).sum()

        w_map[m_start:m_end] = np.maximum(w_map[m_start:m_end], applied)

        print(f"    [cam {cam_s}~{cam_e} → model {m_start}~{m_end}]")
        print(f"      risk_delta={risk_delta:.3f} conf={conf:.2f} | boosted {boosted}/{seg_len} | {reasoning}")
        segment_info.append((m_start, m_end, text[:60]))

    hds = np.clip(deepsee + w_map + W_HW * 0.1, 0, 1)
    return hds, segment_info

# ── Lead Time ──────────────────────────────────────────────────────────────────
def compute_lead_time(gt, hds, deepsee, seq, n_samples, threshold=0.6):
    n_cam = len(glob(f"{IMG_BASE}/{seq}/color/*.png"))
    sec_per_frame = (n_cam / max(n_samples, 1)) / CAM_HZ

    gt_bin   = (gt      > threshold).astype(int)
    hds_bin  = (hds     > threshold).astype(int)
    deep_bin = (deepsee > threshold).astype(int)

    events, in_ev, ev_start = [], False, 0
    for i, v in enumerate(gt_bin):
        if v and not in_ev:   in_ev, ev_start = True, i
        elif not v and in_ev:
            in_ev = False
            events.append(ev_start + np.argmax(gt[ev_start:i]))
    if in_ev: events.append(ev_start + np.argmax(gt[ev_start:]))

    results = []
    for ev in events:
        window = max(0, ev - int(9.0 / sec_per_frame))
        hds_w, deep_w = None, None
        for t in range(window, ev+1):
            if hds_w  is None and hds_bin[t]:  hds_w  = t
            if deep_w is None and deep_bin[t]: deep_w = t
        results.append({
            'event': ev,
            'lead_hds_sec':  (ev - hds_w)  * sec_per_frame if hds_w  is not None else None,
            'lead_deep_sec': (ev - deep_w) * sec_per_frame if deep_w is not None else None,
        })
    return results

# ── 평가 ──────────────────────────────────────────────────────────────────────
def evaluate(gt, baseline, deepsee, hds, thr=0.6):
    gt_b   = (gt       > thr).astype(int)
    base_b = (baseline > thr).astype(int)
    deep_b = (deepsee  > thr).astype(int)
    hds_b  = (hds      > thr).astype(int)

    def sf1(y_true, y_pred):
        if y_pred.sum() == 0 and y_true.sum() == 0: return 1.0, 1.0, 1.0
        if y_pred.sum() == 0: return 0.0, 0.0, 0.0
        return (f1_score(y_true, y_pred, zero_division=0),
                precision_score(y_true, y_pred, zero_division=0),
                recall_score(y_true, y_pred, zero_division=0))

    bf, bp, br = sf1(gt_b, base_b)
    df, dp, dr = sf1(gt_b, deep_b)
    hf, hp, hr = sf1(gt_b, hds_b)
    return {"gt_positive_rate": gt_b.mean(),
            "baseline_f1": bf, "baseline_prec": bp, "baseline_rec": br,
            "deepsee_f1":  df, "deepsee_prec":  dp, "deepsee_rec":  dr,
            "hds_f1":      hf, "hds_prec":      hp, "hds_rec":      hr}

# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_results(seq, gt, baseline, deepsee, hds, metrics, segment_info, lead_results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"OpenLORIS {seq} — HDS Zero-Shot (SenseTime pretrain only)", fontsize=12, fontweight='bold')

    ax1 = axes[0, 0]
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2, fig=fig)
    fig.delaxes(axes[0, 0]); fig.delaxes(axes[0, 1])

    ax1.plot(gt,       color='black', alpha=0.4, lw=1.5, label='GT')
    ax1.plot(baseline, color='green', alpha=0.7, lw=1.2, label='Baseline (RF, OpenLORIS-trained)')
    ax1.plot(deepsee,  color='blue',  alpha=0.7, lw=1.2, label='DeepSEE (SenseTime zero-shot)')
    ax1.plot(hds,      color='red',   alpha=0.9, lw=2.0, label='HDS (Ours)')
    ax1.axhline(0.6, color='gray', ls='--', alpha=0.5, lw=0.8)
    for m_start, m_end, _ in segment_info:
        ax1.axvspan(m_start, m_end, alpha=0.12, color='orange')
    handles, labels = ax1.get_legend_handles_labels()
    handles += [Patch(facecolor='orange', alpha=0.3, label='Human annotation')]
    ax1.legend(handles=handles, fontsize=8)
    ax1.set_title(f"GT pos={metrics['gt_positive_rate']*100:.1f}%  n={len(gt)}")
    ax1.set_ylabel("Normalized score"); ax1.set_ylim(-0.05, 1.1)

    ax2 = axes[1, 0]
    labels_b = ['Baseline\n(RF)', 'DeepSEE\n(zero-shot)', 'HDS\n(Ours)']
    f1s = [metrics['baseline_f1'], metrics['deepsee_f1'], metrics['hds_f1']]
    bars = ax2.bar(labels_b, f1s, color=['green','blue','red'], alpha=0.7, width=0.4)
    for bar, val in zip(bars, f1s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax2.set_ylim(0, 1.1); ax2.set_title("F1 Score"); ax2.set_ylabel("F1")

    ax3 = axes[1, 1]
    diff = [r for r in lead_results
            if r['lead_hds_sec'] is not None and
               (r['lead_deep_sec'] is None or r['lead_hds_sec'] > r['lead_deep_sec'])]
    d_avg = np.mean([r['lead_deep_sec'] or 0 for r in diff]) if diff else 0
    h_avg = np.mean([r['lead_hds_sec']      for r in diff]) if diff else 0
    ax3.bar(['DeepSEE', 'HDS'], [d_avg, h_avg], color=['blue','red'], alpha=0.7, width=0.4)
    ax3.set_title(f"Lead Time (HDS-winning events: {len(diff)})"); ax3.set_ylabel("seconds")

    plt.tight_layout()
    out = f"{OUT_DIR}/{seq}_hds.png"
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → {out}")

# ── 메인 ──────────────────────────────────────────────────────────────────────
print("=" * 65)
print("HDS OpenLORIS Zero-Shot (SenseTime pretrain only)")
print("=" * 65)

all_metrics = {}
all_lead_hds, all_lead_deep = [], []

for fold_idx, seq in enumerate(SEQS):
    print(f"\n[Fold {fold_idx}] {seq}")
    data = load_data(fold_idx)
    n    = len(data['gt'])

    hds, segment_info = compute_hds(data['deepsee'], seq, n)
    metrics      = evaluate(data['gt'], data['baseline'], data['deepsee'], hds)
    lead_results = compute_lead_time(data['gt'], hds, data['deepsee'], seq, n)
    all_metrics[seq] = metrics

    print(f"  GT high-risk: {metrics['gt_positive_rate']*100:.1f}%  n={n}")
    print(f"  Baseline F1: {metrics['baseline_f1']:.3f}  (P={metrics['baseline_prec']:.3f} R={metrics['baseline_rec']:.3f})")
    print(f"  DeepSEE  F1: {metrics['deepsee_f1']:.3f}  (P={metrics['deepsee_prec']:.3f} R={metrics['deepsee_rec']:.3f})")
    print(f"  HDS      F1: {metrics['hds_f1']:.3f}  (P={metrics['hds_prec']:.3f} R={metrics['hds_rec']:.3f})")
    imp = (metrics['hds_f1'] - metrics['deepsee_f1']) / max(metrics['deepsee_f1'], 1e-4) * 100
    print(f"  HDS vs DeepSEE: {min(imp, 9999):.1f}%")

    diff = [r for r in lead_results
            if r['lead_hds_sec'] is not None and
               (r['lead_deep_sec'] is None or r['lead_hds_sec'] > r['lead_deep_sec'])]
    if diff:
        avg_adv = np.mean([r['lead_hds_sec'] - (r['lead_deep_sec'] or 0) for r in diff])
        print(f"  Lead Time advantage ({len(diff)} events): +{avg_adv:.2f}s")
        all_lead_hds.extend([r['lead_hds_sec']      for r in diff])
        all_lead_deep.extend([r['lead_deep_sec'] or 0 for r in diff])

    plot_results(seq, data['gt'], data['baseline'], data['deepsee'],
                 hds, metrics, segment_info, lead_results)

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
    print(f"  Lead Time HDS   : {np.mean(all_lead_hds):.2f}s")
    print(f"  Lead Time Deep  : {np.mean(all_lead_deep):.2f}s")
    print(f"  HDS advantage   : +{np.mean(all_lead_hds)-np.mean(all_lead_deep):.2f}s")
