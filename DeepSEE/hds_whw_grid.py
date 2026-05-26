"""
W_HW 그리드 서치 — LLM 재호출 없이 캐시된 risk_delta 사용
"""
import os, json
import numpy as np
import pandas as pd

RUN_DIR  = "/home/junhyun/SEESys/DeepSEE/Training/runs/Apr04_00-06-40_AHRI-Junhyun"
LLM_LOG  = "/home/junhyun/SEESys/HDS_output/rerun_20260408_143737/llm_responses.json"
CSV_DIR  = "/home/junhyun/SEESys/DeepSEE/data"
SEQS     = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
CAM_HZ   = 20.0

CAM_FRAME_COUNT = {
    "MH_01_easy": 2912, "MH_02_easy": 3014,
    "MH_03_medium": 2700, "MH_04_difficult": 2033, "MH_05_difficult": 2273,
}

HUMAN_ANNOTATIONS = {
    "MH_01_easy":      [(2219, 2357, "Path plan indicates a dark machinery room with dense metallic pipes ahead. Low illumination combined with specular reflections will significantly degrade visual feature tracking.")],
    "MH_03_medium":    [(47, 176, "Path plan enters a poorly lit industrial zone with only lateral lighting. Reduced illumination and dark shadows will significantly degrade feature detection."),
                        (500, 870, "Path plan shows a rapid maneuver zone ahead. Expect heavy camera shake and feature tracking instability."),
                        (1150, 1320, "Upcoming low-light corridor in path plan. Poor illumination will reduce visual features."),
                        (2070, 2240, "Path plan indicates unstable motion zone ahead. Camera blur expected.")],
    "MH_04_difficult": [(77, 231, "Path plan shows downward-looking viewpoint over dark industrial machinery at mission start. Low contrast scene with limited visual features anticipated."),
                        (990, 1443, "Path plan enters extremely dark zone ahead. Near-zero ambient lighting — almost silhouette-only visibility. Severe drift risk."),
                        (1358, 1589, "Path plan navigates through dense metallic tank and pipe cluster. Highly reflective surfaces and complex geometry cause severe feature confusion and mismatching."),
                        (1630, 1826, "Path plan shows another dark low-visibility zone. Poor ambient lighting will cause repeated visual feature degradation.")],
    "MH_05_difficult": [(0, 420, "Unstable initial flight phase in path plan. High vibration expected."),
                        (1070, 1530, "Dark low-feature zone detected ahead in path plan. Severe drift risk anticipated."),
                        (1920, 2060, "Sudden trajectory change in path plan. Rapid motion change expected.")],
}

# ── LLM 캐시 로드 (seq + human_text → risk_delta) ──────────────────────────
with open(LLM_LOG, encoding="utf-8") as f:
    llm_cache_raw = json.load(f)

llm_cache = {}
for entry in llm_cache_raw:
    key = (entry["seq"], entry["human_text"][:40])
    llm_cache[key] = float(entry["response"]["risk_delta"])

def get_risk_delta(seq_name, human_text):
    key = (seq_name, human_text[:40])
    if key in llm_cache:
        return llm_cache[key]
    raise ValueError(f"Cache miss: {key}")

# ── 데이터 로드 ────────────────────────────────────────────────────────────
def load_predictions():
    results = {}
    for fold_idx, seq in enumerate(SEQS):
        gt   = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_gt.npy",   allow_pickle=True)
        est  = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_est.npy",  allow_pickle=True)
        base = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_base.npy", allow_pickle=True)
        gt_flat   = np.concatenate([np.array(x).flatten() for x in gt])
        est_flat  = np.concatenate([np.array(x).flatten() for x in est])
        base_flat = np.array(base).flatten()
        def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
        results[seq] = {"gt": norm(gt_flat), "deepsee": norm(est_flat), "baseline": norm(base_flat)}
    return results

# ── Image Quality Guardrail ────────────────────────────────────────────────
_W_ENTROPY, _W_BRIGHT, _W_LAP = 0.335, 0.305, 0.262
_WT = _W_ENTROPY + _W_BRIGHT + _W_LAP
W_ENTROPY, W_BRIGHT, W_LAP = _W_ENTROPY/_WT, _W_BRIGHT/_WT, _W_LAP/_WT

def compute_guardrail(seq_name, n_samples):
    df = pd.read_csv(f"{CSV_DIR}/data_EuRoC_{seq_name}_0.csv")
    df = df.dropna(subset=['Brightness', 'Entropy', 'Laplacian'])
    t = np.arange(len(df))
    t_new = np.linspace(0, len(df)-1, n_samples)
    def interp_norm(col):
        v = np.interp(t_new, t, df[col].values)
        return (v - v.min()) / (v.max() - v.min() + 1e-8)
    bright_n  = interp_norm('Brightness')
    entropy_n = interp_norm('Entropy')
    lap_n     = interp_norm('Laplacian')
    return W_ENTROPY*(1-entropy_n) + W_BRIGHT*(1-bright_n) + W_LAP*lap_n

# ── HDS 계산 ──────────────────────────────────────────────────────────────
def compute_hds(deepsee, G, seq_name, n_samples, W_HW, gap_thresh=0.65):
    delta_sy = np.zeros(n_samples)
    n_cam = CAM_FRAME_COUNT[seq_name]
    for cam_s, cam_e, htxt in HUMAN_ANNOTATIONS.get(seq_name, []):
        ms = int(cam_s * n_samples / n_cam)
        me = min(int(cam_e * n_samples / n_cam) + 1, n_samples)
        rd = get_risk_delta(seq_name, htxt)
        gap = np.maximum(0.0, gap_thresh - deepsee)
        applied = np.minimum(gap, rd)
        delta_sy[ms:me] += applied[ms:me]
    return np.clip(deepsee + delta_sy + W_HW * G, 0, 1)

# ── EWR 계산 ──────────────────────────────────────────────────────────────
def compute_ewr_all(preds, W_HW, gap_thresh=0.65, risk_thr=0.6):
    all_hds, all_deep = [], []
    for seq in SEQS:
        if seq not in HUMAN_ANNOTATIONS:
            continue
        dp = preds[seq]
        n  = len(dp['gt'])
        G  = compute_guardrail(seq, n)
        hds = compute_hds(dp['deepsee'], G, seq, n, W_HW, gap_thresh)
        n_cam = CAM_FRAME_COUNT[seq]
        sec_per_mf = (n_cam / CAM_HZ) / n
        gt_bin = (dp['gt'] > risk_thr).astype(int)
        events = [i for i in range(1, n) if gt_bin[i]==1 and gt_bin[i-1]==0]
        for ev in events:
            lb = max(0, ev - 30)
            hc  = next((i for i in range(lb, ev+1) if hds[i]         > risk_thr), None)
            dc  = next((i for i in range(lb, ev+1) if dp['deepsee'][i] > risk_thr), None)
            all_hds.append( (ev - hc)  * sec_per_mf if hc  is not None else None)
            all_deep.append((ev - dc) * sec_per_mf if dc is not None else None)
    return all_hds, all_deep

def ewr_rate(leads, thr):
    total = len(leads)
    return sum(1 for v in leads if v is not None and v >= thr) / total * 100

# ── F1 (전체 시퀀스) ──────────────────────────────────────────────────────
def compute_avg_f1(preds, W_HW, gap_thresh=0.65, risk_thr=0.6):
    from sklearn.metrics import f1_score
    f1s = []
    for seq in SEQS:
        dp = preds[seq]
        n  = len(dp['gt'])
        G  = compute_guardrail(seq, n)
        hds = compute_hds(dp['deepsee'], G, seq, n, W_HW, gap_thresh)
        gt_b  = (dp['gt']  > risk_thr).astype(int)
        hds_b = (hds       > risk_thr).astype(int)
        f1s.append(f1_score(gt_b, hds_b, zero_division=0))
    return np.mean(f1s)

def compute_rmse(preds, W_HW, gap_thresh=0.65):
    rmses = []
    for seq in SEQS:
        dp = preds[seq]
        n  = len(dp['gt'])
        G  = compute_guardrail(seq, n)
        hds = compute_hds(dp['deepsee'], G, seq, n, W_HW, gap_thresh)
        rmses.append(np.sqrt(np.mean((hds - dp['gt'])**2)))
    return np.mean(rmses)

# ── 그리드 서치 ───────────────────────────────────────────────────────────
W_HW_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
GAP_GRID  = [0.60, 0.65, 0.70]

preds = load_predictions()

# DeepSEE baseline EWR
deep_leads = []
for seq in SEQS:
    if seq not in HUMAN_ANNOTATIONS:
        continue
    dp = preds[seq]
    n  = len(dp['gt'])
    n_cam = CAM_FRAME_COUNT[seq]
    sec_per_mf = (n_cam / CAM_HZ) / n
    gt_bin = (dp['gt'] > 0.6).astype(int)
    events = [i for i in range(1, n) if gt_bin[i]==1 and gt_bin[i-1]==0]
    for ev in events:
        lb = max(0, ev-30)
        dc = next((i for i in range(lb, ev+1) if dp['deepsee'][i] > 0.6), None)
        deep_leads.append((ev - dc) * (n_cam/CAM_HZ)/n if dc is not None else None)

print("=" * 72)
print("W_HW Grid Search — Image Quality Guardrail (Spearman weighted)")
print(f"  DeepSEE baseline : EWR ≥1s={ewr_rate(deep_leads,1):.1f}%  ≥3s={ewr_rate(deep_leads,3):.1f}%  ≥5s={ewr_rate(deep_leads,5):.1f}%")
print("=" * 72)

print(f"\n{'W_HW':>6} {'gap':>5} | {'EWR≥1s':>7} {'EWR≥3s':>7} {'EWR≥5s':>7} | {'F1':>6} {'RMSE':>7} | {'Δ≥1s':>6} {'Δ≥3s':>6} {'Δ≥5s':>6}")
print("-" * 72)

for gap_thr in GAP_GRID:
    for whw in W_HW_GRID:
        hds_l, _ = compute_ewr_all(preds, whw, gap_thr)
        e1 = ewr_rate(hds_l, 1.0)
        e3 = ewr_rate(hds_l, 3.0)
        e5 = ewr_rate(hds_l, 5.0)
        f1 = compute_avg_f1(preds, whw, gap_thr)
        rm = compute_rmse(preds, whw, gap_thr)
        d1 = e1 - ewr_rate(deep_leads, 1)
        d3 = e3 - ewr_rate(deep_leads, 3)
        d5 = e5 - ewr_rate(deep_leads, 5)
        marker = " ◀" if (e1 >= 91.5 and e5 >= 85.4 and f1 >= 0.68) else ""
        print(f"{whw:>6.2f} {gap_thr:>5.2f} | {e1:>6.1f}% {e3:>6.1f}% {e5:>6.1f}% | {f1:>6.3f} {rm:>7.4f} | {d1:>+5.1f}pp {d3:>+5.1f}pp {d5:>+5.1f}pp{marker}")
    print()

print(f"\n현재 설정(W_HW=0.10, gap=0.65) 기준 DeepSEE F1: ", end="")
from sklearn.metrics import f1_score
f1s_deep = []
for seq in SEQS:
    dp = preds[seq]
    gt_b = (dp['gt'] > 0.6).astype(int)
    d_b  = (dp['deepsee'] > 0.6).astype(int)
    f1s_deep.append(f1_score(gt_b, d_b, zero_division=0))
print(f"{np.mean(f1s_deep):.3f}")
