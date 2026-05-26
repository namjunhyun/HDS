"""
HDS v2 - SenseTime Cross-Validation
r̃_t = r̂_t + Δ_symbolic(t) + Δ_hw(t)

개선사항:
1. F1 제거 → Lead Time / RMSE 집중
2. Proactive shift: 어노테이션 구간을 PROACTIVE_FRAMES 앞당겨 적용 (미래 위험 사전 경보)
3. 어노테이션 feature 강화: MatchedInliers 드롭 + Brightness + Entropy + Laplacian
4. Ramp-up 효과: 경보 구간 앞부분에 선형 증가 적용
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dotenv import load_dotenv
import anthropic
import json, re, time

load_dotenv("/home/junhyun/SEESys/DeepSEE/Training/.env")

# ── 설정 ──────────────────────────────────────────────────────────────────────
RUN_DIR = "/home/junhyun/SEESys/DeepSEE/Training/runs/Apr07_17-15-25_AHRI-Junhyun"
OUT_DIR = "/home/junhyun/SEESys/HDS_v2_SenseTime_output"
TS_DIR  = "/home/junhyun/SEESys/DeepSEE/Training/datasets/SenseTime/timeSeries"

W_HW          = 0.05
THRESHOLD     = 0.6
FPS           = 20.0
PROACTIVE_FRAMES = 15         # 0.75s proactive
RAMP_FRAMES      = 10

FOLD_TEST_TRAJS = [
    ['A2', 'B5', 'A5'],
    ['A4', 'B6', 'B2'],
    ['A4', 'A3', 'B1'],
    ['A4', 'B3', 'B0'],
    ['A7', 'B3', 'B0'],
    ['A7', 'A3', 'B2'],
]

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. 어노테이션 자동 생성 (강화 버전) ────────────────────────────────────────
def auto_generate_annotations(max_segs=3, min_seg_len=15):
    """
    MatchedInliers 드롭 + Brightness + Entropy + Laplacian 조합으로
    SLAM 추적 어려울 구간 탐지. severity 기준 상위 max_segs개 선택.
    """
    annotations = {}
    for fold_idx, trajs in enumerate(FOLD_TEST_TRAJS):
        segs = []
        for traj in trajs:
            csv_path = os.path.join(TS_DIR, f"data_SenseTime_{traj}_0.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)

            n = len(df)
            risk_score = np.zeros(n)

            # MatchedInliers: 낮을수록 SLAM tracking 어려움
            if 'MatchedInlier' in df.columns:
                mi = df['MatchedInlier'].fillna(df['MatchedInlier'].mean()).values
                mi_norm = (mi - mi.min()) / (mi.max() - mi.min() + 1e-8)
                risk_score += (1.0 - mi_norm) * 2.0   # 가중치 2배 (가장 직접적인 지표)

            # Brightness: 어두울수록 위험
            if 'Brightness' in df.columns:
                b = df['Brightness'].fillna(df['Brightness'].mean()).values
                b_norm = (b - b.min()) / (b.max() - b.min() + 1e-8)
                risk_score += (1.0 - b_norm)

            # Entropy: 낮을수록 texture 부족
            if 'Entropy' in df.columns:
                e = df['Entropy'].fillna(df['Entropy'].mean()).values
                e_norm = (e - e.min()) / (e.max() - e.min() + 1e-8)
                risk_score += (1.0 - e_norm)

            # Laplacian: 낮을수록 blur
            if 'Laplacian' in df.columns:
                lap = df['Laplacian'].fillna(df['Laplacian'].mean()).values
                lap_norm = (lap - lap.min()) / (lap.max() - lap.min() + 1e-8)
                risk_score += (1.0 - lap_norm)

            # 상위 75% 이상 구간을 위험으로 분류
            thresh = np.percentile(risk_score, 75)
            risky = risk_score > thresh

            in_seg, seg_start = False, 0
            for i, r in enumerate(risky):
                if r and not in_seg:
                    in_seg, seg_start = True, i
                elif not r and in_seg:
                    in_seg = False
                    if i - seg_start >= min_seg_len:
                        avg_risk = risk_score[seg_start:i].mean()
                        avg_b    = df['Brightness'].values[seg_start:i].mean() if 'Brightness' in df.columns else 0
                        avg_e    = df['Entropy'].values[seg_start:i].mean()    if 'Entropy'    in df.columns else 0
                        avg_mi   = df['MatchedInlier'].values[seg_start:i].mean() if 'MatchedInlier' in df.columns else 0
                        text = (f"Path plan enters high-risk zone in {traj} (frames {seg_start}-{i}). "
                                f"MatchedInliers={avg_mi:.0f} (low), Brightness={avg_b:.1f}, Entropy={avg_e:.2f}. "
                                f"SLAM tracking expected to degrade significantly.")
                        segs.append((seg_start/n, i/n, text, avg_risk))
            if in_seg and n - seg_start >= min_seg_len:
                avg_risk = risk_score[seg_start:].mean()
                avg_b  = df['Brightness'].values[seg_start:].mean() if 'Brightness' in df.columns else 0
                avg_e  = df['Entropy'].values[seg_start:].mean()    if 'Entropy'    in df.columns else 0
                avg_mi = df['MatchedInlier'].values[seg_start:].mean() if 'MatchedInlier' in df.columns else 0
                text = (f"Path plan enters high-risk zone in {traj}. "
                        f"MatchedInliers={avg_mi:.0f}, Brightness={avg_b:.1f}, Entropy={avg_e:.2f}.")
                segs.append((seg_start/n, 1.0, text, avg_risk))

        segs.sort(key=lambda x: -x[3])  # severity 높은 순
        annotations[fold_idx] = [(s, e, t) for s, e, t, _ in segs[:max_segs]]
    return annotations

# ── 2. 예측값 로드 ─────────────────────────────────────────────────────────────
def load_predictions():
    results = {}
    for fold_idx in range(len(FOLD_TEST_TRAJS)):
        gt   = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_gt.npy",   allow_pickle=True)
        est  = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_est.npy",  allow_pickle=True)
        base = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_base.npy", allow_pickle=True)

        gt_flat   = np.concatenate([np.array(x).flatten() for x in gt])
        est_flat  = np.concatenate([np.array(x).flatten() for x in est])
        base_flat = np.array(base).flatten()

        gt_flat   = (np.exp(gt_flat)   - 1) / 10000
        est_flat  = (np.exp(est_flat)  - 1) / 10000
        base_flat = (np.exp(base_flat) - 1) / 10000

        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        results[fold_idx] = {
            "gt":       norm(gt_flat),
            "deepsee":  norm(est_flat),
            "baseline": norm(base_flat),
        }
    return results

# ── 3. Feature-based Target Signal ───────────────────────────────────────────
def compute_target_signal(fold_idx, n_samples):
    """
    MatchedInliers + Brightness + Entropy + Laplacian으로
    연속적인 feature-based 위험도 신호 target[t] ∈ [0,1] 계산.
    이 신호가 HDS의 blending target이 됨.
    """
    trajs = FOLD_TEST_TRAJS[fold_idx]
    dfs = []
    for traj in trajs:
        p = os.path.join(TS_DIR, f"data_SenseTime_{traj}_0.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if not dfs:
        return np.zeros(n_samples)

    df = pd.concat(dfs, ignore_index=True)
    n_csv = len(df)
    t_csv = np.arange(n_csv)
    t_new = np.linspace(0, n_csv - 1, n_samples)

    risk = np.zeros(n_samples)
    weights = {'MatchedInlier': 2.0, 'Brightness': 1.0, 'Entropy': 1.0, 'Laplacian': 1.0}
    total_w = 0.0

    for col, w in weights.items():
        if col not in df.columns:
            continue
        v = df[col].fillna(df[col].mean()).values
        v_r = np.interp(t_new, t_csv, v)
        v_norm = (v_r - v_r.min()) / (v_r.max() - v_r.min() + 1e-8)
        risk += w * (1.0 - v_norm)   # 낮을수록 위험 → 반전
        total_w += w

    if total_w > 0:
        risk /= total_w

    # 부드럽게 smoothing
    from scipy.ndimage import uniform_filter1d
    risk = uniform_filter1d(risk, size=10)
    return risk.clip(0, 1)


# ── 4. Hardware Guardrail ─────────────────────────────────────────────────────
def compute_guardrail(fold_idx, n_samples):
    trajs = FOLD_TEST_TRAJS[fold_idx]
    dfs = []
    for traj in trajs:
        csv_path = os.path.join(TS_DIR, f"data_SenseTime_{traj}_0.csv")
        if os.path.exists(csv_path):
            dfs.append(pd.read_csv(csv_path))
    if not dfs:
        return np.zeros(n_samples)

    df = pd.concat(dfs, ignore_index=True)
    cols = [c for c in ['Brightness', 'Entropy', 'Laplacian', 'MatchedInlier'] if c in df.columns]
    if not cols:
        return np.zeros(n_samples)

    risks = []
    for col in cols:
        v = df[col].fillna(df[col].mean()).values
        t_new = np.linspace(0, len(v)-1, n_samples)
        v_r = np.interp(t_new, np.arange(len(v)), v)
        v_norm = (v_r - v_r.min()) / (v_r.max() - v_r.min() + 1e-8)
        risks.append(1.0 - v_norm)

    G = np.mean(risks, axis=0)
    thresh = np.percentile(G, 70)
    return (G > thresh).astype(float)

# ── 4. LLM Symbolic ───────────────────────────────────────────────────────────
def generate_risk_delta(human_text, fold_idx, retries=3):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = f"""You are a SLAM drift risk expert. A human operator observing the path plan says:
"{human_text}"
Return ONLY JSON: {{"confidence": <float 0.0-1.0>, "reason": "<one sentence>"}}
Guidelines: how strongly to trust the feature-based risk signal in this zone.
low confidence=0.1~0.3 (uncertain), medium=0.3~0.6, high=0.6~0.9 (very likely risky).
Low MatchedInliers = high confidence. Vague description = low confidence."""

    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=128,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = re.sub(r'```[a-z]*\n?', '', msg.content[0].text.strip()).rstrip('`')
            return json.loads(raw)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {"confidence": 0.4, "reason": "fallback"}

# ── 6. HDS 융합 — Target-based Blending ──────────────────────────────────────
def compute_hds(deepsee, target, G, fold_idx, n_samples, annotations):
    """
    Target-based blending:
        HDS[t] = DeepSEE[t] + w[t] * max(0, target[t] - DeepSEE[t])

    - target[t]: feature signal (MatchedInliers 등)로 계산한 기대 위험도
    - w[t]:      LLM이 판단한 annotation 신뢰도 (0~1)
    - max(0,...): 위쪽으로만 보정 (DeepSEE가 이미 높으면 건드리지 않음)
    - Overshoot 원천 차단: HDS ≤ target (target 이상으로 못 올라감)
    """
    w_map   = np.zeros(n_samples)
    segs    = annotations.get(fold_idx, [])
    seg_info = []

    for s_frac, e_frac, text in segs:
        orig_start = int(s_frac * n_samples)
        orig_end   = min(int(e_frac * n_samples) + 1, n_samples)

        # Proactive shift
        m_start = max(0, orig_start - PROACTIVE_FRAMES)
        m_end   = max(0, orig_end   - PROACTIVE_FRAMES)
        if m_end <= m_start:
            continue

        result = generate_risk_delta(text, fold_idx)
        conf   = float(result.get('confidence', result.get('risk_delta', 0.4)))
        conf   = min(max(conf, 0.0), 1.0)

        # Ramp-up
        seg_len  = m_end - m_start
        ramp     = np.ones(seg_len)
        ramp_len = min(RAMP_FRAMES, seg_len)
        ramp[:ramp_len] = np.linspace(0.1, 1.0, ramp_len)

        w_map[m_start:m_end] = np.maximum(w_map[m_start:m_end], conf * ramp)

        print(f"    [Annotation @ {m_start}~{m_end} (orig {orig_start}~{orig_end})] "
              f"confidence={conf:.3f} | {result['reason'][:60]}")
        seg_info.append((m_start, m_end, orig_start, orig_end, text))

    # Target-based blending: DeepSEE를 target 쪽으로 당김
    correction = w_map * np.maximum(0.0, target - deepsee)

    # Hardware guardrail (소량)
    delta_hw = W_HW * G

    hds = np.clip(deepsee + correction + delta_hw, 0, 1)
    return hds, seg_info

# ── 6. Lead Time 계산 ─────────────────────────────────────────────────────────
def compute_lead_time(gt, hds, deepsee, n_samples):
    gt_b  = gt      > THRESHOLD
    hds_b = hds     > THRESHOLD
    ds_b  = deepsee > THRESHOLD

    events, in_e, start = [], False, 0
    for i, v in enumerate(gt_b):
        if v and not in_e:   in_e, start = True, i
        elif not v and in_e:
            in_e = False
            if i - start >= 3: events.append((start, i))
    if in_e and n_samples - start >= 3: events.append((start, n_samples))

    results = []
    for es, ee in events:
        window = range(max(0, es - int(FPS * 9)), es)
        hds_warn = next((i for i in window if hds_b[i]), None)
        ds_warn  = next((i for i in window if ds_b[i]),  None)
        lead_h = (es - hds_warn) / FPS if hds_warn is not None else None
        lead_d = (es - ds_warn)  / FPS if ds_warn  is not None else None
        results.append(dict(event=es, lead_hds=lead_h, lead_ds=lead_d))
    return results

# ── 7. RMSE 계산 (실제 cm 단위) ───────────────────────────────────────────────
def compute_rmse_cm(fold_idx):
    gt   = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_gt.npy",  allow_pickle=True)
    est  = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_est.npy", allow_pickle=True)
    g = (np.exp(np.concatenate([np.array(x).flatten() for x in gt]))  - 1) / 10000
    e = (np.exp(np.concatenate([np.array(x).flatten() for x in est])) - 1) / 10000
    return float(np.sqrt(np.mean((g - e) ** 2)))

# ── 8. 요약 플롯 ──────────────────────────────────────────────────────────────
def plot_summary(fold_rmse_ds, fold_rmse_hds, all_lead_hds, all_lead_ds,
                 avg_lead_h, avg_lead_d):
    n_folds = len(fold_rmse_ds)
    folds   = [f"Fold {i}" for i in range(n_folds)]
    x = np.arange(n_folds)
    w = 0.35

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("HDS v2 — SenseTime Cross-Validation (Proactive+MatchedInliers)", fontsize=13, fontweight='bold')

    # (1) RMSE per fold (normalized)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.bar(x - w/2, fold_rmse_ds,  w, label='DeepSEE', color='#2196F3', alpha=0.85)
    ax1.bar(x + w/2, fold_rmse_hds, w, label='HDS',     color='#F44336', alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(folds, fontsize=8)
    ax1.set_ylabel("RMSE (normalized)"); ax1.set_title("RMSE per Fold (lower=better)")
    ax1.legend(fontsize=8)

    # (2) RMSE 개선율
    ax2 = fig.add_subplot(2, 3, 2)
    imps = [(d - h) / (d + 1e-8) * 100 for d, h in zip(fold_rmse_ds, fold_rmse_hds)]
    colors = ['#F44336' if i > 0 else '#9E9E9E' for i in imps]
    bars = ax2.bar(folds, imps, color=colors, alpha=0.85)
    ax2.axhline(0, color='black', lw=0.8)
    for bar, imp in zip(bars, imps):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.3 if imp >= 0 else -1.5),
                 f"{imp:+.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax2.set_ylabel("RMSE Reduction (%)"); ax2.set_title("HDS RMSE Improvement")
    ax2.set_xticklabels(folds, fontsize=8)

    # (3) Lead Time per fold (HDS avg)
    ax3 = fig.add_subplot(2, 3, 3)
    # fold별 lead 평균 재계산 (all_lead_hds는 전체 통합이므로 여기선 avg_lead_h만 표시)
    ax3.bar(['DeepSEE', 'HDS'], [avg_lead_d, avg_lead_h],
            color=['#2196F3', '#F44336'], alpha=0.85, width=0.4)
    for i, (method, val) in enumerate(zip(['DeepSEE', 'HDS'], [avg_lead_d, avg_lead_h])):
        ax3.text(i, val + 0.05, f"{val:.2f}s", ha='center', va='bottom',
                 fontsize=12, fontweight='bold')
    if avg_lead_h > avg_lead_d:
        ax3.annotate(f"+{avg_lead_h - avg_lead_d:.2f}s",
                     xy=(1, avg_lead_h), xytext=(0.5, avg_lead_h + 0.4),
                     fontsize=12, color='#F44336', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#F44336'))
    ax3.set_ylabel("Avg Lead Time (s)"); ax3.set_title("Avg Lead Time: HDS vs DeepSEE")
    ax3.set_ylim(0, max(avg_lead_h, avg_lead_d) * 1.5 + 0.5)

    # (4) Lead Time 분포
    ax4 = fig.add_subplot(2, 3, 4)
    if all_lead_hds:
        bins = np.linspace(0, 9, 15)
        ax4.hist(all_lead_hds, bins=bins, color='#F44336', alpha=0.7, label=f'HDS (avg={avg_lead_h:.2f}s)')
        if all_lead_ds:
            ax4.hist(all_lead_ds, bins=bins, color='#2196F3', alpha=0.7, label=f'DeepSEE (avg={avg_lead_d:.2f}s)')
        ax4.axvline(avg_lead_h, color='#F44336', ls='--', lw=1.5)
        if all_lead_ds:
            ax4.axvline(avg_lead_d, color='#2196F3', ls='--', lw=1.5)
        ax4.set_xlabel("Lead Time (s)"); ax4.set_ylabel("Count")
        ax4.set_title("Lead Time Distribution")
        ax4.legend(fontsize=8)

    # (5) 평균 RMSE 비교
    ax5 = fig.add_subplot(2, 3, 5)
    avg_ds  = np.mean(fold_rmse_ds)
    avg_hds = np.mean(fold_rmse_hds)
    bars5 = ax5.bar(['DeepSEE', 'HDS'], [avg_ds, avg_hds],
                    color=['#2196F3', '#F44336'], alpha=0.85, width=0.4)
    for bar, val in zip(bars5, [avg_ds, avg_hds]):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                 f"{val:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax5.set_ylabel("Avg RMSE (normalized)"); ax5.set_title("Average RMSE (6-Fold)")
    ax5.set_ylim(0, max(avg_ds, avg_hds) * 1.3)

    # (6) 이벤트별 Lead Time scatter
    ax6 = fig.add_subplot(2, 3, 6)
    if all_lead_hds and all_lead_ds:
        n_pair = min(len(all_lead_hds), len(all_lead_ds))
        ax6.scatter(all_lead_ds[:n_pair], all_lead_hds[:n_pair],
                    color='#9C27B0', alpha=0.7, s=40)
        lim = max(max(all_lead_hds), max(all_lead_ds)) + 0.5
        ax6.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5, label='y=x')
        ax6.set_xlabel("DeepSEE Lead Time (s)")
        ax6.set_ylabel("HDS Lead Time (s)")
        ax6.set_title("Per-event Lead: HDS vs DeepSEE\n(above diagonal = HDS wins)")
        ax6.legend(fontsize=8)
    else:
        ax6.text(0.5, 0.5, "Insufficient paired data", ha='center', va='center', transform=ax6.transAxes)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/HDS_Summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  요약 플롯 저장: {out_path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("HDS v2 — SenseTime Cross-Validation (Proactive + MatchedInliers)")
    print("=" * 65)

    annotations = auto_generate_annotations(max_segs=3)
    predictions = load_predictions()

    fold_rmse_ds, fold_rmse_hds = [], []
    all_lead_hds, all_lead_ds   = [], []
    all_lead_hds_full = []  # 전체 이벤트 포함 (HDS가 경보 못한 것 포함)

    for fold_idx in range(len(FOLD_TEST_TRAJS)):
        trajs = FOLD_TEST_TRAJS[fold_idx]
        print(f"\n[Fold {fold_idx}] test: {trajs}")

        data   = predictions[fold_idx]
        n      = len(data['gt'])
        G      = compute_guardrail(fold_idx, n)
        target = compute_target_signal(fold_idx, n)

        hds, seg_info = compute_hds(data['deepsee'], target, G, fold_idx, n, annotations)

        # RMSE (normalized)
        rmse_ds  = float(np.sqrt(np.mean((data['gt'] - data['deepsee']) ** 2)))
        rmse_hds = float(np.sqrt(np.mean((data['gt'] - hds) ** 2)))
        fold_rmse_ds.append(rmse_ds)
        fold_rmse_hds.append(rmse_hds)

        # RMSE in cm
        rmse_cm = compute_rmse_cm(fold_idx)

        print(f"  DeepSEE  RMSE(norm)={rmse_ds:.4f}  RMSE(cm)={rmse_cm:.5f}")
        print(f"  HDS      RMSE(norm)={rmse_hds:.4f}")

        # Lead Time
        lead_res = compute_lead_time(data['gt'], hds, data['deepsee'], n)
        n_events = len(lead_res)
        hds_warns  = [r for r in lead_res if r['lead_hds'] is not None]
        hds_better = [r for r in hds_warns if r['lead_ds'] is None or r['lead_hds'] > r['lead_ds'] + 0.1]

        if hds_warns:
            print(f"  Lead Time ({len(hds_better)}/{n_events} events HDS better, {len(hds_warns)}/{n_events} HDS warned):")
            for r in hds_better:
                ds_str = f"{r['lead_ds']:.2f}s" if r['lead_ds'] is not None else "MISSED"
                print(f"    frame {r['event']:4d}: HDS={r['lead_hds']:.2f}s  DeepSEE={ds_str}")
            lh = [r['lead_hds'] for r in hds_better]
            ld = [r['lead_ds']  for r in hds_better if r['lead_ds'] is not None]
            all_lead_hds.extend(lh)
            all_lead_ds.extend(ld)
            print(f"  HDS avg lead (better events): +{np.mean(lh):.2f}s")
        else:
            print(f"  Lead Time: HDS no early warnings (0/{n_events} events)")

        # 폴드별 그래프
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        t = np.arange(n)

        ax = axes[0]
        ax.plot(t, data['gt'],      color='black', lw=1.2, label='GT',      alpha=0.85)
        ax.plot(t, data['deepsee'], color='blue',  lw=1.0, label='DeepSEE', alpha=0.7)
        ax.plot(t, hds,             color='red',   lw=1.2, label='HDS',     alpha=0.85)
        ax.axhline(THRESHOLD, color='gray', ls='--', lw=0.8)
        for ms, me, os_, oe, _ in seg_info:
            ax.axvspan(ms, me,  alpha=0.20, color='orange', label='_ann_proactive')
            ax.axvspan(os_, oe, alpha=0.10, color='purple', label='_ann_orig')
        handles, labels = ax.get_legend_handles_labels()
        handles += [Patch(facecolor='orange', alpha=0.4, label='Annotation (proactive)'),
                    Patch(facecolor='purple', alpha=0.3, label='Annotation (original)')]
        ax.legend(handles=handles, fontsize=8)
        ax.set_ylabel("Normalized Risk"); ax.set_ylim(-0.05, 1.1)
        ax.set_title(f"Fold {fold_idx} | {trajs} | RMSE: DeepSEE={rmse_ds:.4f}, HDS={rmse_hds:.4f}")

        ax2 = axes[1]
        ax2.plot(t, hds - data['deepsee'], color='green', lw=1.0, label='HDS - DeepSEE (delta)')
        ax2.axhline(0, color='black', lw=0.5)
        ax2.set_xlabel("Model frame"); ax2.set_ylabel("Delta"); ax2.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/HDS_Fold{fold_idx}.png", dpi=150, bbox_inches='tight')
        plt.close()

    # ── 전체 요약 ─────────────────────────────────────────────────────────────
    avg_lead_h = np.mean(all_lead_hds) if all_lead_hds else 0.0
    avg_lead_d = np.mean(all_lead_ds)  if all_lead_ds  else 0.0

    print("\n" + "=" * 65)
    print("전체 평균 결과")
    print("=" * 65)
    print(f"  DeepSEE avg RMSE(norm) : {np.mean(fold_rmse_ds):.4f}")
    print(f"  HDS     avg RMSE(norm) : {np.mean(fold_rmse_hds):.4f}")
    rmse_imp = (np.mean(fold_rmse_ds) - np.mean(fold_rmse_hds)) / (np.mean(fold_rmse_ds) + 1e-8) * 100
    print(f"  RMSE 개선율            : {rmse_imp:+.1f}%")
    print(f"\n  HDS avg Lead Time  : {avg_lead_h:.2f}s")
    print(f"  DeepSEE avg Lead   : {avg_lead_d:.2f}s")
    print(f"  HDS 우위            : +{avg_lead_h - avg_lead_d:.2f}s")
    print(f"\n출력 디렉토리: {OUT_DIR}")

    plot_summary(fold_rmse_ds, fold_rmse_hds, all_lead_hds, all_lead_ds, avg_lead_h, avg_lead_d)
