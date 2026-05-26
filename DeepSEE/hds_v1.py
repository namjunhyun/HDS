"""
HDS (Hybrid DeepSEE) - v1
r̃_t = r̂_t + Δ_symbolic(t) + Δ_hw(t)

- r̂_t       : DeepSEE neural drift prediction
- Δ_symbolic : LLM-generated symbolic rules from human path-plan context
               → 위험 구간 진입 전부터 적용 (proactive, path-plan based)
- Δ_hw       : Hardware guardrail (IMU spec threshold), 전 구간 적용

핵심 contribution: 경로가 사전에 계획된 로봇에서,
사람이 경로 상의 위험 구간을 미리 파악 → LLM이 symbolic constraint 생성
→ DeepSEE보다 먼저 drift risk 경보 (Lead Time)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dotenv import load_dotenv
import anthropic
import json
import re

import datetime
load_dotenv("/home/junhyun/SEESys/DeepSEE/Training/.env")

# ── 설정 ──────────────────────────────────────────────────────────────────
RUN_DIR  = "/home/junhyun/SEESys/DeepSEE/Training/runs/Apr04_00-06-40_AHRI-Junhyun"
_ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR  = f"/home/junhyun/SEESys/HDS_output/rerun_{_ts}"
SEQS     = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
GT_BASE  = "/home/junhyun/SEESys/Datasets/SenseTime/EuRoC"

W_SY = 0.3
W_HW = 0.0   # guardrail 제거 — symbolic only

# EuRoC 카메라 20Hz → 1 model frame ≈ 0.05s * (n_cam/n_model)
CAM_HZ = 20.0

# ── Spearman 상관계수 기반 이미지 품질 가중치 ─────────────────────────────────
# EuRoC RelativeError 와의 Spearman r 값 (euroc_error_analysis.py 결과)
#   Entropy   r = -0.335 → 낮은 엔트로피 → 높은 위험
#   Brightness r = -0.305 → 어두울수록 → 높은 위험
#   Laplacian  r = +0.262 → 높은 Laplacian → 높은 위험 (고속 움직임 상관)
# |r| 값을 정규화하여 guardrail 가중치로 사용
_W_ENTROPY = 0.335
_W_BRIGHT  = 0.305
_W_LAP     = 0.262
_W_TOTAL   = _W_ENTROPY + _W_BRIGHT + _W_LAP   # 0.902
W_ENTROPY  = _W_ENTROPY / _W_TOTAL              # ≈ 0.371
W_BRIGHT   = _W_BRIGHT  / _W_TOTAL              # ≈ 0.338
W_LAP      = _W_LAP     / _W_TOTAL              # ≈ 0.291

CSV_DIR = "/home/junhyun/SEESys/DeepSEE/data"

os.makedirs(OUT_DIR, exist_ok=True)

# ── LLM 응답 로그 ─────────────────────────────────────────────────────────
LLM_LOG = []  # 실행 중 누적, 종료 시 JSON 저장

# ── 사람 입력 어노테이션 (경로 계획 기반, 위험 구간 진입 전 입력) ──────────
# (cam_start, cam_end, human_text)
# cam_start: 실제 위험 구간보다 ~100 카메라 프레임(5초) 앞
# → 사람이 경로 맵 보고 "이 구간 위험" 미리 입력하는 시뮬레이션
HUMAN_ANNOTATIONS = {
    # ── MH_01_easy ────────────────────────────────────────────────────────
    "MH_01_easy": [
        # cam 2219~2357: 어두운 파이프/기계실, 반사성 금속 (이미지 직접 확인)
        (2219, 2357, "Path plan indicates a dark machinery room with dense metallic pipes ahead. Low illumination combined with specular reflections will significantly degrade visual feature tracking."),
    ],
    # ── MH_03_medium ──────────────────────────────────────────────────────
    "MH_03_medium": [
        # cam 47~176: 어두운 공장, 측면 조명만 (이미지 직접 확인)
        (47,  176,  "Path plan enters a poorly lit industrial zone with only lateral lighting. Reduced illumination and dark shadows will significantly degrade feature detection."),
        (500,  870,  "Path plan shows a rapid maneuver zone ahead. Expect heavy camera shake and feature tracking instability."),
        (1150, 1320, "Upcoming low-light corridor in path plan. Poor illumination will reduce visual features."),
        (2070, 2240, "Path plan indicates unstable motion zone ahead. Camera blur expected."),
    ],
    # ── MH_04_difficult ───────────────────────────────────────────────────
    "MH_04_difficult": [
        # cam 77~231: 위에서 내려다보는 저조도 기계 구역 (이미지 직접 확인)
        (77,   231,  "Path plan shows downward-looking viewpoint over dark industrial machinery at mission start. Low contrast scene with limited visual features anticipated."),
        # cam 990~1443: 매우 어두운 구간 (이미지 직접 확인 - 거의 실루엣만)
        (990,  1443, "Path plan enters extremely dark zone ahead. Near-zero ambient lighting — almost silhouette-only visibility. Severe drift risk."),
        # cam 1358~1589: 반사성 금속 탱크/파이프 밀집 (이미지 직접 확인)
        (1358, 1589, "Path plan navigates through dense metallic tank and pipe cluster. Highly reflective surfaces and complex geometry cause severe feature confusion and mismatching."),
        # cam 1683~1826: 다시 저조도 구간 (이미지 직접 확인)
        (1630, 1826, "Path plan shows another dark low-visibility zone. Poor ambient lighting will cause repeated visual feature degradation."),
    ],
    # ── MH_05_difficult ───────────────────────────────────────────────────
    "MH_05_difficult": [
        (0,    420,  "Unstable initial flight phase in path plan. High vibration expected."),
        (1070, 1530, "Dark low-feature zone detected ahead in path plan. Severe drift risk anticipated."),
        (1920, 2060, "Sudden trajectory change in path plan. Rapid motion change expected."),
    ],
}

CAM_FRAME_COUNT = {
    "MH_01_easy":      2912,
    "MH_02_easy":      3014,
    "MH_03_medium":    2700,
    "MH_04_difficult": 2033,
    "MH_05_difficult": 2273,
}

# ── 1. DeepSEE 예측값 로드 ────────────────────────────────────────────────
def load_deepsee_predictions():
    results = {}
    for fold_idx, seq in enumerate(SEQS):
        gt   = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_gt.npy",   allow_pickle=True)
        est  = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_est.npy",  allow_pickle=True)
        base = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_base.npy", allow_pickle=True)

        gt_flat   = np.concatenate([np.array(x).flatten() for x in gt])
        est_flat  = np.concatenate([np.array(x).flatten() for x in est])
        base_flat = np.array(base).flatten()

        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        results[seq] = {
            "gt":       norm(gt_flat),
            "deepsee":  norm(est_flat),
            "baseline": norm(base_flat),
        }
    return results

# ── 2. Image Quality Guardrail (Spearman 상관계수 기반) ──────────────────────
def compute_hardware_guardrail(seq_name, n_samples):
    """
    IMU 대신 이미지 품질 지표를 Spearman |r| 가중치로 결합한 guardrail.
    G = W_ENTROPY*(1-entropy_norm) + W_BRIGHT*(1-bright_norm) + W_LAP*lap_norm

    방향 근거 (EuRoC RelativeError vs 각 지표의 Spearman r):
      Entropy   r=-0.335: 낮은 엔트로피 → 높은 오차  → (1-norm) 사용
      Brightness r=-0.305: 낮은 밝기    → 높은 오차  → (1-norm) 사용
      Laplacian  r=+0.262: 높은 Laplacian → 높은 오차 →  norm   사용
    """
    csv_path = f"{CSV_DIR}/data_EuRoC_{seq_name}_0.csv"
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Brightness', 'Entropy', 'Laplacian'])

    bright = df['Brightness'].values
    entropy = df['Entropy'].values
    lap     = df['Laplacian'].values

    t     = np.arange(len(bright))
    t_new = np.linspace(0, len(bright) - 1, n_samples)

    bright_r  = np.interp(t_new, t, bright)
    entropy_r = np.interp(t_new, t, entropy)
    lap_r     = np.interp(t_new, t, lap)

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    bright_norm  = norm(bright_r)
    entropy_norm = norm(entropy_r)
    lap_norm     = norm(lap_r)

    # 각 지표가 위험에 기여하는 방향: 낮은 밝기/엔트로피, 높은 Laplacian → 위험 증가
    G = (W_ENTROPY * (1 - entropy_norm) +
         W_BRIGHT  * (1 - bright_norm)  +
         W_LAP     * lap_norm)

    return G, bright_norm, entropy_norm

# ── 3. LLM Symbolic Rule 생성 ─────────────────────────────────────────────
def generate_symbolic_rules_from_human(human_text, seq_name):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = f"""You are a SLAM drift risk expert. A human operator reviewing the robot's path plan says:

"{human_text}"

Sequence: {seq_name}

Based on this proactive path-plan observation, decide how much to increase the drift risk score (0~1 scale) for the upcoming zone.

Return ONLY a JSON dict:
{{
  "risk_delta": <float 0.0-0.5>,
  "reason": "<one sentence>"
}}

Guidelines:
- Minor upcoming degradation: 0.05~0.15
- Moderate (low light or fast motion): 0.15~0.30
- Severe (dark + no features + rapid motion): 0.30~0.50"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.content[0].text.strip()
    raw = re.sub(r'```[a-z]*\n?', '', raw).strip().rstrip('`')
    result = json.loads(raw)

    # 로그 누적
    LLM_LOG.append({
        "seq":        seq_name,
        "human_text": human_text,
        "prompt":     prompt,
        "response":   result,
        "timestamp":  datetime.datetime.now().isoformat(),
    })
    return result

# ── 4. HDS 융합 ───────────────────────────────────────────────────────────
def compute_hds_score(deepsee_pred, G, seq_name, n_samples):
    delta_symbolic = np.zeros(n_samples)
    annotations    = HUMAN_ANNOTATIONS.get(seq_name, [])
    n_cam          = CAM_FRAME_COUNT[seq_name]
    segment_info   = []

    for cam_start, cam_end, human_text in annotations:
        m_start = int(cam_start * n_samples / n_cam)
        m_end   = min(int(cam_end * n_samples / n_cam) + 1, n_samples)

        print(f"    [Path-plan input @ model frames {m_start}~{m_end}] \"{human_text[:55]}...\"")
        result     = generate_symbolic_rules_from_human(human_text, seq_name)
        risk_delta = float(result['risk_delta'])
        reason     = result['reason']

        # Gap-based boost: threshold까지 거리에 비례해서 delta 적용
        # 이미 0.65 이상인 프레임은 건드리지 않음 (FP 방지)
        gap = np.maximum(0.0, 0.65 - deepsee_pred)
        applied = np.minimum(gap, risk_delta)
        seg_applied = np.zeros(n_samples)
        seg_applied[m_start:m_end] = applied[m_start:m_end]
        delta_symbolic += seg_applied

        n_boosted = int((seg_applied[m_start:m_end] > 0).sum())
        avg_boost = float(seg_applied[m_start:m_end].mean())
        print(f"      → risk_delta={risk_delta:.3f} | boosted {n_boosted}/{m_end-m_start} frames (avg +{avg_boost:.3f}) | {reason}")
        segment_info.append((m_start, m_end, human_text))

    delta_hw = W_HW * G
    r_tilde  = np.clip(deepsee_pred + delta_symbolic + delta_hw, 0, 1)
    return r_tilde, segment_info

# ── 5. Lead Time 계산 ─────────────────────────────────────────────────────
def compute_lead_time(gt, hds, deepsee, seq_name, n_samples, threshold=0.6):
    """
    GT 고위험 이벤트 시작 시점 대비 HDS/DeepSEE의 경보 선행 시간 측정.
    Lead time > 0: GT 스파이크 전에 경보 (proactive)
    Lead time < 0: GT 스파이크 후에 경보 (reactive)
    """
    n_cam      = CAM_FRAME_COUNT[seq_name]
    sec_per_mf = (n_cam / CAM_HZ) / n_samples  # 모델 프레임당 초

    gt_binary = (gt > threshold).astype(int)

    # GT 이벤트 시작점 탐지 (low→high 전환)
    events = []
    for i in range(1, len(gt_binary)):
        if gt_binary[i] == 1 and gt_binary[i-1] == 0:
            events.append(i)

    results = []
    for ev_start in events:
        lookback = max(0, ev_start - 30)

        # HDS가 threshold 처음 넘는 시점
        hds_cross  = next((i for i in range(lookback, ev_start + 1) if hds[i] > threshold),     None)
        deep_cross = next((i for i in range(lookback, ev_start + 1) if deepsee[i] > threshold), None)

        lead_hds  = (ev_start - hds_cross)  * sec_per_mf if hds_cross  is not None else None
        lead_deep = (ev_start - deep_cross) * sec_per_mf if deep_cross is not None else None

        results.append({
            "event_start":  ev_start,
            "lead_hds_sec":  lead_hds,
            "lead_deep_sec": lead_deep,
        })

    return results

# ── 5-2. Early Warning Rate 계산 ─────────────────────────────────────────
def compute_early_warning_rate(lead_results, thresholds=(1.0, 3.0, 5.0)):
    """
    GT 고위험 이벤트 중 X초 전에 경보한 비율 (Early Warning Rate)
    HDS vs DeepSEE 비교
    """
    total = len(lead_results)
    if total == 0:
        return {}

    ewr = {}
    for thr in thresholds:
        hds_warned  = sum(1 for r in lead_results if r['lead_hds_sec']  is not None and r['lead_hds_sec']  >= thr)
        deep_warned = sum(1 for r in lead_results if r['lead_deep_sec'] is not None and r['lead_deep_sec'] >= thr)
        ewr[thr] = {
            'hds':     hds_warned  / total,
            'deepsee': deep_warned / total,
            'n_events': total,
        }
    return ewr

# ── 6. 평가 ──────────────────────────────────────────────────────────────
def evaluate(gt, baseline, deepsee, hds, risk_threshold=0.6):
    from sklearn.metrics import precision_score, recall_score, f1_score

    gt_bin   = (gt       > risk_threshold).astype(int)
    base_bin = (baseline > risk_threshold).astype(int)
    deep_bin = (deepsee  > risk_threshold).astype(int)
    hds_bin  = (hds      > risk_threshold).astype(int)

    def safe_metrics(pred):
        if pred.sum() == 0:
            return 0.0, 0.0, 0.0
        return (precision_score(gt_bin, pred, zero_division=0),
                recall_score(gt_bin, pred, zero_division=0),
                f1_score(gt_bin, pred, zero_division=0))

    bp, br, bf = safe_metrics(base_bin)
    dp, dr, df = safe_metrics(deep_bin)
    hp, hr, hf = safe_metrics(hds_bin)

    return {
        "baseline_f1": bf, "baseline_prec": bp, "baseline_rec": br,
        "deepsee_f1":  df, "deepsee_prec":  dp, "deepsee_rec":  dr,
        "hds_f1":      hf, "hds_prec":      hp, "hds_rec":      hr,
        "gt_positive_rate": gt_bin.mean(),
    }

# ── 7. 시각화 ─────────────────────────────────────────────────────────────
def plot_results(seq_name, gt, baseline, deepsee, hds, G, metrics, segment_info, lead_results):
    fig = plt.figure(figsize=(18, 13))
    gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(gt,       color='black', alpha=0.4, lw=1.5, label='Ground Truth')
    ax1.plot(baseline, color='green', alpha=0.7, lw=1.5, label='Baseline (RF)')
    ax1.plot(deepsee,  color='blue',  alpha=0.7, lw=1.5, label='DeepSEE')
    ax1.plot(hds,      color='red',   alpha=0.9, lw=2.0, label='HDS (Ours)')
    ax1.axhline(0.6, color='gray', ls='--', alpha=0.5, label='Risk Threshold')
    for i, (m_start, m_end, _) in enumerate(segment_info):
        ax1.axvspan(m_start, m_end, color='yellow', alpha=0.2,
                    label='Path-plan input zone' if i == 0 else '')
    # Lead time 화살표 표시
    for r in lead_results:
        ev = r['event_start']
        if r['lead_hds_sec'] is not None and r['lead_hds_sec'] > 0:
            hds_frame = ev - int(r['lead_hds_sec'] * len(gt) / (CAM_FRAME_COUNT.get(seq_name, 2000) / CAM_HZ))
            ax1.annotate('', xy=(ev, 0.62), xytext=(max(0, hds_frame), 0.62),
                         arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax1.set_title(f'Drift Risk Prediction — {seq_name}', fontsize=13)
    ax1.set_ylabel('Normalized Drift Risk')
    ax1.legend(fontsize=8, ncol=5)
    ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(range(len(G)), 0, G, color='orange', alpha=0.5,
                     label='Image Quality Guardrail (Spearman weighted)')
    for m_start, m_end, _ in segment_info:
        ax2.axvspan(m_start, m_end, color='yellow', alpha=0.2)
    ax2.set_ylabel('Guardrail Score'); ax2.set_ylim(0, 1.2)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(gs[2, 0])
    labels = ['Baseline\n(RF)', 'DeepSEE', 'HDS\n(Ours)']
    f1s    = [metrics['baseline_f1'], metrics['deepsee_f1'], metrics['hds_f1']]
    bars   = ax3.bar(labels, f1s, color=['green','blue','red'], alpha=0.7, width=0.4)
    for bar, val in zip(bars, f1s):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_ylabel('F1 Score'); ax3.set_title('High-Risk Event Detection F1')
    ax3.set_ylim(0, 1.1); ax3.grid(True, alpha=0.2, axis='y')

    ax4 = fig.add_subplot(gs[2, 1])
    diff_events = [
        r for r in lead_results
        if r['lead_hds_sec'] is not None and (
            r['lead_deep_sec'] is None or
            r['lead_hds_sec'] - r['lead_deep_sec'] > 0.1
        )
    ]
    if diff_events:
        x = np.arange(len(diff_events))
        w = 0.35
        ax4.bar(x - w/2, [r['lead_hds_sec'] for r in diff_events], w,
                label='HDS', color='red', alpha=0.8)
        ax4.bar(x + w/2, [r['lead_deep_sec'] or 0 for r in diff_events], w,
                label='DeepSEE', color='blue', alpha=0.8)
        ax4.axhline(0, color='black', lw=0.8)
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"E{r['event_start']}" for r in diff_events], fontsize=7)
        ax4.set_xlabel('Drift Event (frame)'); ax4.set_ylabel('Lead Time (sec)')
        ax4.set_title('Early Warning Lead Time\n(events where HDS > DeepSEE)')
        ax4.legend(fontsize=9); ax4.grid(True, alpha=0.2, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No lead-time advantage\n(easy or DeepSEE already covers)',
                 ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        ax4.set_title('Lead Time')

    plt.savefig(f"{OUT_DIR}/HDS_{seq_name}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{OUT_DIR}/HDS_{seq_name}.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUT_DIR}/HDS_{seq_name}.png")

# ── 메인 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("HDS (Hybrid DeepSEE) — Proactive Path-Plan Based Evaluation")
    print("=" * 65)

    predictions = load_deepsee_predictions()
    all_metrics    = {}
    all_lead_hds   = []
    all_lead_deep  = []

    for fold_idx, seq in enumerate(SEQS):
        print(f"\n[Fold {fold_idx}] {seq}")
        data = predictions[seq]
        n    = len(data['gt'])

        G, bright_norm, entropy_norm = compute_hardware_guardrail(seq, n)

        has_annot = seq in HUMAN_ANNOTATIONS
        if not has_annot:
            print(f"  (easy — no path-plan annotations)")

        hds, segment_info = compute_hds_score(data['deepsee'], G, seq, n)
        metrics           = evaluate(data['gt'], data['baseline'], data['deepsee'], hds)
        lead_results      = compute_lead_time(data['gt'], hds, data['deepsee'], seq, n)
        ewr               = compute_early_warning_rate(lead_results)
        all_metrics[seq]  = metrics

        print(f"  GT high-risk rate : {metrics['gt_positive_rate']*100:.1f}%")
        print(f"  Baseline F1: {metrics['baseline_f1']:.3f}  (P={metrics['baseline_prec']:.3f} R={metrics['baseline_rec']:.3f})")
        print(f"  DeepSEE  F1: {metrics['deepsee_f1']:.3f}  (P={metrics['deepsee_prec']:.3f} R={metrics['deepsee_rec']:.3f})")
        print(f"  HDS      F1: {metrics['hds_f1']:.3f}  (P={metrics['hds_prec']:.3f} R={metrics['hds_rec']:.3f})")
        imp = (metrics['hds_f1'] - metrics['deepsee_f1']) / (metrics['deepsee_f1'] + 1e-8) * 100
        print(f"  HDS vs DeepSEE F1: {imp:+.1f}%")

        # Early Warning Rate
        if ewr and has_annot:
            print(f"  Early Warning Rate (n_events={list(ewr.values())[0]['n_events']}):")
            for thr, v in ewr.items():
                print(f"    ≥{thr:.0f}s:  HDS={v['hds']*100:.1f}%  DeepSEE={v['deepsee']*100:.1f}%  (+{(v['hds']-v['deepsee'])*100:.1f}pp)")

        if lead_results and has_annot:
            # HDS vs DeepSEE 차이가 있는 이벤트만 필터링
            diff_events = [
                r for r in lead_results
                if r['lead_hds_sec'] is not None and (
                    r['lead_deep_sec'] is None or
                    r['lead_hds_sec'] - r['lead_deep_sec'] > 0.1
                )
            ]
            print(f"  Lead Time (HDS > DeepSEE 이벤트만, {len(diff_events)}/{len(lead_results)}):")
            for i, r in enumerate(diff_events):
                lh = f"{r['lead_hds_sec']:.2f}s"
                ld = f"{r['lead_deep_sec']:.2f}s" if r['lead_deep_sec'] is not None else "MISSED"
                adv = r['lead_hds_sec'] - (r['lead_deep_sec'] or 0)
                print(f"    Event (frame {r['event_start']:3d}): HDS={lh}, DeepSEE={ld}  (+{adv:.2f}s)")

            if diff_events:
                hds_adv = [
                    r['lead_hds_sec'] - (r['lead_deep_sec'] or 0)
                    for r in diff_events
                ]
                print(f"  HDS advantage (avg): +{np.mean(hds_adv):.2f}s earlier warning")
                all_lead_hds.extend([r['lead_hds_sec'] for r in diff_events])
                all_lead_deep.extend([r['lead_deep_sec'] or 0 for r in diff_events])

        plot_results(seq, data['gt'], data['baseline'], data['deepsee'],
                     hds, G, metrics, segment_info, lead_results)

    print("\n" + "=" * 65)
    print("전체 평균 결과")
    print("=" * 65)
    avg_base = np.mean([m['baseline_f1'] for m in all_metrics.values()])
    avg_deep = np.mean([m['deepsee_f1']  for m in all_metrics.values()])
    avg_hds  = np.mean([m['hds_f1']      for m in all_metrics.values()])
    print(f"  Baseline avg F1 : {avg_base:.3f}")
    print(f"  DeepSEE  avg F1 : {avg_deep:.3f}")
    print(f"  HDS      avg F1 : {avg_hds:.3f}")
    print(f"  HDS vs DeepSEE  : {(avg_hds-avg_deep)/(avg_deep+1e-8)*100:+.1f}%")
    if all_lead_hds:
        advantages = [h - d for h, d in zip(all_lead_hds, all_lead_deep)]
        print(f"\n  [Lead Time — HDS > DeepSEE 이벤트 기준]")
        print(f"  HDS  avg lead  : {np.mean(all_lead_hds):.2f}s")
        print(f"  Deep avg lead  : {np.mean(all_lead_deep):.2f}s")
        print(f"  HDS advantage  : +{np.mean(advantages):.2f}s earlier warning")

    # ── 전체 Early Warning Rate (annotated 시퀀스만) ──────────────────────
    all_lead_flat = []
    all_lead_deep_flat = []
    for fold_idx, seq in enumerate(SEQS):
        if seq not in HUMAN_ANNOTATIONS:
            continue
        data = predictions[seq]
        n    = len(data['gt'])
        G, _, _ = compute_hardware_guardrail(seq, n)
        hds_s, _ = compute_hds_score(data['deepsee'], G, seq, n)
        lr = compute_lead_time(data['gt'], hds_s, data['deepsee'], seq, n)
        for r in lr:
            all_lead_flat.append(r['lead_hds_sec'])
            all_lead_deep_flat.append(r['lead_deep_sec'])

    if all_lead_flat:
        total = len(all_lead_flat)
        print(f"\n  [Early Warning Rate — 전체 GT 이벤트 기준, n={total}]")
        print(f"  {'Threshold':<10} {'HDS':>10} {'DeepSEE':>10} {'Gap':>10}")
        print(f"  {'-'*42}")
        for thr in [1.0, 3.0, 5.0]:
            h = sum(1 for v in all_lead_flat      if v is not None and v >= thr) / total
            d = sum(1 for v in all_lead_deep_flat if v is not None and v >= thr) / total
            print(f"  ≥{thr:.0f}s{'':<7} {h*100:>9.1f}% {d*100:>9.1f}% {(h-d)*100:>+9.1f}pp")

    # ── LLM 응답 로그 저장 ────────────────────────────────────────────────
    llm_log_path = f"{OUT_DIR}/llm_responses.json"
    with open(llm_log_path, "w", encoding="utf-8") as f:
        json.dump(LLM_LOG, f, ensure_ascii=False, indent=2)
    print(f"\n  LLM 응답 로그 저장: {llm_log_path}  ({len(LLM_LOG)} 건)")

    # ── 결과 요약 텍스트 저장 ─────────────────────────────────────────────
    summary_path = f"{OUT_DIR}/results_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"HDS v1 Rerun — {_ts}\n")
        f.write(f"RUN_DIR: {RUN_DIR}\n")
        f.write(f"Guardrail: Image Quality (Spearman) — W_ENTROPY={W_ENTROPY:.3f}, W_BRIGHT={W_BRIGHT:.3f}, W_LAP={W_LAP:.3f}\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'Seq':<20} {'Base F1':>8} {'Deep F1':>8} {'HDS F1':>8}\n")
        f.write("-" * 50 + "\n")
        for seq, m in all_metrics.items():
            f.write(f"{seq:<20} {m['baseline_f1']:>8.3f} {m['deepsee_f1']:>8.3f} {m['hds_f1']:>8.3f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'AVG':<20} {avg_base:>8.3f} {avg_deep:>8.3f} {avg_hds:>8.3f}\n")
        f.write("=" * 65 + "\n\n")
        if all_lead_flat:
            f.write(f"Early Warning Rate (n={len(all_lead_flat)} GT events)\n")
            f.write(f"{'Threshold':<12} {'HDS':>10} {'DeepSEE':>10} {'Gap':>10}\n")
            f.write("-" * 45 + "\n")
            for thr in [1.0, 3.0, 5.0]:
                h = sum(1 for v in all_lead_flat      if v is not None and v >= thr) / len(all_lead_flat)
                d = sum(1 for v in all_lead_deep_flat if v is not None and v >= thr) / len(all_lead_flat)
                f.write(f"≥{thr:.0f}s{'':<9} {h*100:>9.1f}% {d*100:>9.1f}% {(h-d)*100:>+9.1f}pp\n")
        f.write(f"\nLLM 응답 로그: llm_responses.json ({len(LLM_LOG)} 건)\n")
    print(f"  결과 요약 저장: {summary_path}")
    print(f"\n  출력 디렉토리: {OUT_DIR}")
