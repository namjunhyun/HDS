"""
HDS on SenseTime — guardrail + symbolic (LLM)
- Hardware guardrail: low Brightness + low Laplacian = visual degradation
- Symbolic rules: 저품질 구간 자동 탐지 → Claude API로 risk_delta 생성
"""
import os, glob, datetime, json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from dotenv import load_dotenv
import anthropic

load_dotenv("/home/junhyun/SEESys/DeepSEE/Training/.env")

RUN_DIR  = "/home/junhyun/SEESys/DeepSEE/Training/runs/Apr08_11-02-44_AHRI-Junhyun"
DATA_DIR = "/home/junhyun/SEESys/DeepSEE/Training/datasets/SenseTime/timeSeries"
_ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR  = f"/home/junhyun/SEESys/HDS_output/hds_sensetime_symbolic_{_ts}"
os.makedirs(OUT_DIR, exist_ok=True)

W_HW      = 0.1
W_SY      = 0.3
THRESHOLD = 0.6
LLM_LOG   = []

# SenseTime 6-fold test sequence mapping
FOLD_TEST_SEQS = [
    ['A2', 'B5', 'A5'],
    ['A4', 'B6', 'B2'],
    ['A4', 'A3', 'B1'],
    ['A4', 'B3', 'B0'],
    ['A7', 'B3', 'B0'],
    ['A7', 'A3', 'B2'],
]

def load_sensetime_df(seq_names):
    dfs = []
    for seq in seq_names:
        files = sorted(glob.glob(f"{DATA_DIR}/data_SenseTime_{seq}_*.csv"))
        dfs.extend([pd.read_csv(f) for f in files])
    return pd.concat(dfs, ignore_index=True)

def detect_bad_zones(df, n_samples, bright_pct=25, lap_pct=25, min_gap=20):
    """Brightness/Laplacian 하위 percentile 구간을 위험 후보로 탐지, 인접 구간 병합"""
    bright = df['Brightness'].values
    lap    = df['Laplacian'].values
    t      = np.arange(len(bright))
    t_new  = np.linspace(0, len(bright)-1, n_samples)
    bright_r = np.interp(t_new, t, bright)
    lap_r    = np.interp(t_new, t, lap)

    bad = (bright_r < np.percentile(bright_r, bright_pct)) | \
          (lap_r    < np.percentile(lap_r,    lap_pct))

    # 연속 구간 병합
    zones, in_zone, start = [], False, 0
    for i, b in enumerate(bad):
        if b and not in_zone:
            in_zone, start = True, i
        elif not b and in_zone:
            in_zone = False
            zones.append((start, i))
    if in_zone:
        zones.append((start, len(bad)))

    # 너무 짧은 구간 제거 (< min_gap)
    zones = [(s, e) for s, e in zones if e - s >= min_gap]

    # 특성 기술 생성
    annotated = []
    for s, e in zones:
        avg_b = bright_r[s:e].mean()
        avg_l = lap_r[s:e].mean()
        if avg_b < np.percentile(bright_r, bright_pct) and avg_l < np.percentile(lap_r, lap_pct):
            text = f"Detected dark and blurry zone (brightness={avg_b:.1f}, sharpness={avg_l:.1f}). Low illumination and motion blur will severely degrade visual feature tracking."
        elif avg_b < np.percentile(bright_r, bright_pct):
            text = f"Detected dark zone (brightness={avg_b:.1f}). Low illumination will reduce feature detection quality."
        else:
            text = f"Detected blurry/low-texture zone (sharpness={avg_l:.1f}). Motion blur or texture-poor surface will degrade SLAM tracking."
        annotated.append((s, e, text))

    # guardrail
    bright_norm = 1.0 - (bright_r - bright_r.min()) / (bright_r.max() - bright_r.min() + 1e-8)
    lap_norm    = 1.0 - (lap_r    - lap_r.min())    / (lap_r.max()    - lap_r.min()    + 1e-8)
    G = 0.5 * bright_norm + 0.5 * lap_norm

    return annotated, G

def generate_symbolic_rule(human_text, seq_label):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = f"""You are a SLAM drift risk expert. Sensor data analysis says:

"{human_text}"

Sequence: {seq_label}

Decide how much to increase the drift risk score (0~1 scale).
Return ONLY JSON: {{"risk_delta": <float 0.0-0.5>, "reason": "<one sentence>"}}

Guidelines:
- Minor degradation: 0.05~0.15
- Moderate (low light or blur): 0.15~0.30
- Severe (dark + blur): 0.30~0.50"""

    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=128,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = re.sub(r'```[a-z]*\n?', '', msg.content[0].text.strip()).strip().rstrip('`')
    result = json.loads(raw)
    LLM_LOG.append({"seq": seq_label, "text": human_text, "response": result,
                    "ts": datetime.datetime.now().isoformat()})
    return result

def compute_hds(deep_n, G, annotations, n_samples, seq_label):
    delta_sy = np.zeros(n_samples)
    for s, e, text in annotations:
        print(f"    [zone {s}~{e}] {text[:60]}...")
        result = generate_symbolic_rule(text, seq_label)
        rd = float(result['risk_delta'])
        gap     = np.maximum(0.0, 0.65 - deep_n)
        applied = np.minimum(gap, rd)
        seg     = np.zeros(n_samples)
        seg[s:e] = applied[s:e]
        delta_sy += seg
        print(f"      → risk_delta={rd:.3f} | {result['reason']}")
    return np.clip(deep_n + delta_sy + W_HW * G, 0, 1)

def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
def get_rmse(gt, p): return np.sqrt(np.mean((gt - p) ** 2))
def get_mape(gt, p):
    m = np.abs(gt) > 1e-6
    return np.mean(np.abs((gt[m] - p[m]) / gt[m])) * 100

def compute_ewr(gt, hds, deep, n_samples, fps=30.0):
    spm = 1.0 / fps
    gt_bin = (gt > THRESHOLD).astype(int)
    events = [i for i in range(1, len(gt_bin)) if gt_bin[i]==1 and gt_bin[i-1]==0]
    if not events: return {}, []
    results = []
    for ev in events:
        lb = max(0, ev-int(5/spm))
        hc = next((i for i in range(lb, ev+1) if hds[i]  > THRESHOLD), None)
        dc = next((i for i in range(lb, ev+1) if deep[i] > THRESHOLD), None)
        results.append({'event_start': ev,
                        'lead_hds_sec':  (ev-hc)*spm if hc is not None else None,
                        'lead_deep_sec': (ev-dc)*spm if dc is not None else None})
    ewr = {}
    total = len(results)
    for thr in [1.0, 3.0, 5.0]:
        h = sum(1 for r in results if r['lead_hds_sec']  is not None and r['lead_hds_sec']  >= thr)
        d = sum(1 for r in results if r['lead_deep_sec'] is not None and r['lead_deep_sec'] >= thr)
        ewr[thr] = {'hds': h/total, 'deepsee': d/total, 'n': total}
    return ewr, results

# ── 메인 ─────────────────────────────────────────────────────────────────
print("="*60)
print("HDS on SenseTime (guardrail + symbolic)")
print("="*60)

all_metrics  = []
all_ewr_h    = {1.0:[], 3.0:[], 5.0:[]}
all_ewr_d    = {1.0:[], 3.0:[], 5.0:[]}
total_events = 0

for fold_idx in range(1, 6):  # fold 0 파일 손상
    seqs      = FOLD_TEST_SEQS[fold_idx]
    seq_label = f"SenseTime_fold{fold_idx}"

    gt_raw   = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_gt.npy",   allow_pickle=True)
    est_raw  = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_est.npy",  allow_pickle=True)
    base_raw = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_base.npy", allow_pickle=True)

    gt_flat   = (np.exp(np.concatenate([np.array(x).flatten() for x in gt_raw]).astype(float))  - 1) / 10000
    est_flat  = (np.exp(np.concatenate([np.array(x).flatten() for x in est_raw]).astype(float)) - 1) / 10000
    base_flat = (np.exp(np.array(base_raw).flatten().astype(float)) - 1) / 10000

    n      = len(gt_flat)
    gt_n   = norm(gt_flat)
    deep_n = norm(est_flat)
    base_n = norm(base_flat)

    df_seq             = load_sensetime_df(seqs)
    annotations, G     = detect_bad_zones(df_seq, n)

    print(f"\n[Fold {fold_idx}] test={seqs}  →  {len(annotations)} bad zones detected")
    hds_n = compute_hds(deep_n, G, annotations, n, seq_label)

    gt_bin   = (gt_n   > THRESHOLD).astype(int)
    bf = f1_score(gt_bin, (base_n > THRESHOLD).astype(int), zero_division=0)
    df_ = f1_score(gt_bin, (deep_n > THRESHOLD).astype(int), zero_division=0)
    hf = f1_score(gt_bin, (hds_n  > THRESHOLD).astype(int), zero_division=0)

    b_rmse = get_rmse(gt_flat, base_flat); d_rmse = get_rmse(gt_flat, est_flat)
    b_mape = get_mape(gt_flat, base_flat); d_mape = get_mape(gt_flat, est_flat)

    # HDS RMSE (normalized → original scale 역변환)
    gt_min, gt_max = gt_flat.min(), gt_flat.max()
    hds_orig = hds_n * (gt_max - gt_min) + gt_min
    h_rmse = get_rmse(gt_flat, hds_orig)
    h_mape = get_mape(gt_flat, hds_orig)

    ewr, lead_results = compute_ewr(gt_n, hds_n, deep_n, n)
    total_events += len(lead_results)
    for thr in [1.0, 3.0, 5.0]:
        if thr in ewr:
            all_ewr_h[thr].append(ewr[thr]['hds']     * ewr[thr]['n'])
            all_ewr_d[thr].append(ewr[thr]['deepsee'] * ewr[thr]['n'])

    all_metrics.append({'base_f1':bf,'deep_f1':df_,'hds_f1':hf,
                        'b_rmse':b_rmse,'d_rmse':d_rmse,'h_rmse':h_rmse,
                        'b_mape':b_mape,'d_mape':d_mape,'h_mape':h_mape})

    # ── 시각화 ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios':[3,1]})
    ax = axes[0]
    ax.plot(gt_n,   color='black', alpha=0.4, lw=1.5, label='GT')
    ax.plot(base_n, color='green', alpha=0.7, lw=1.2, label=f'RF Baseline (F1={bf:.3f})')
    ax.plot(deep_n, color='blue',  alpha=0.7, lw=1.2, label=f'DeepSEE (F1={df_:.3f})')
    ax.plot(hds_n,  color='red',   alpha=0.9, lw=2.0, label=f'HDS (F1={hf:.3f})')
    ax.axhline(THRESHOLD, color='gray', ls='--', alpha=0.5)
    for s, e, _ in annotations:
        ax.axvspan(s, e, color='yellow', alpha=0.25,
                   label='Auto-detected zone' if s == annotations[0][0] else '')
    ax.set_title(f'HDS on SenseTime — Fold {fold_idx} {seqs}', fontsize=12)
    ax.set_ylabel('Normalized Drift Risk')
    ax.legend(fontsize=8, ncol=5)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-0.05, 1.1)

    ax2 = axes[1]
    bright_vals = df_seq['Brightness'].values
    lap_vals    = df_seq['Laplacian'].values
    t_raw = np.linspace(0, n-1, len(bright_vals))
    ax2.plot(t_raw, (bright_vals - bright_vals.min())/(bright_vals.max()-bright_vals.min()+1e-8),
             color='orange', alpha=0.7, label='Brightness (norm)')
    ax2.plot(t_raw, (lap_vals - lap_vals.min())/(lap_vals.max()-lap_vals.min()+1e-8),
             color='purple', alpha=0.7, label='Sharpness (norm)')
    for s, e, _ in annotations:
        ax2.axvspan(s, e, color='yellow', alpha=0.3)
    ax2.set_ylabel('Image Quality'); ax2.set_xlabel('Model Frame')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/HDS_SenseTime_fold{fold_idx}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{OUT_DIR}/HDS_SenseTime_fold{fold_idx}.pdf", bbox_inches='tight')
    plt.close()

    print(f"  F1   — Base:{bf:.3f}  DeepSEE:{df_:.3f}  HDS:{hf:.3f}")
    print(f"  RMSE — Base:{b_rmse:.5f}  DeepSEE:{d_rmse:.5f}  HDS:{h_rmse:.5f}")
    print(f"  MAPE — Base:{b_mape:.1f}%  DeepSEE:{d_mape:.1f}%  HDS:{h_mape:.1f}%")
    if ewr:
        for thr, v in ewr.items():
            print(f"  EWR≥{thr:.0f}s — HDS:{v['hds']*100:.1f}%  DeepSEE:{v['deepsee']*100:.1f}%  (n={v['n']})")

print("\n" + "="*60)
print("전체 평균")
print("="*60)
avg_bf = np.mean([m['base_f1']  for m in all_metrics])
avg_df = np.mean([m['deep_f1']  for m in all_metrics])
avg_hf = np.mean([m['hds_f1']   for m in all_metrics])
avg_br = np.mean([m['b_rmse']   for m in all_metrics])
avg_dr = np.mean([m['d_rmse']   for m in all_metrics])
avg_hr = np.mean([m['h_rmse']   for m in all_metrics])
avg_bm = np.mean([m['b_mape']   for m in all_metrics])
avg_dm = np.mean([m['d_mape']   for m in all_metrics])
avg_hm = np.mean([m['h_mape']   for m in all_metrics])
print(f"  F1   — Base:{avg_bf:.3f}  DeepSEE:{avg_df:.3f}  HDS:{avg_hf:.3f}")
print(f"  RMSE — Base:{avg_br:.5f}  DeepSEE:{avg_dr:.5f}  HDS:{avg_hr:.5f}")
print(f"  MAPE — Base:{avg_bm:.1f}%  DeepSEE:{avg_dm:.1f}%  HDS:{avg_hm:.1f}%")
if total_events > 0:
    print(f"\n  EWR (n={total_events} events)")
    for thr in [1.0, 3.0, 5.0]:
        h = sum(all_ewr_h[thr]) / total_events if all_ewr_h[thr] else 0
        d = sum(all_ewr_d[thr]) / total_events if all_ewr_d[thr] else 0
        print(f"  ≥{thr:.0f}s — HDS:{h*100:.1f}%  DeepSEE:{d*100:.1f}%  ({(h-d)*100:+.1f}pp)")

# 저장
with open(f"{OUT_DIR}/llm_responses.json","w") as f:
    json.dump(LLM_LOG, f, ensure_ascii=False, indent=2)
with open(f"{OUT_DIR}/results_summary.txt","w") as f:
    f.write(f"HDS SenseTime (symbolic+guardrail) — {_ts}\n\n")
    f.write(f"{'Fold':<25} {'Base F1':>8} {'Deep F1':>8} {'HDS F1':>8}\n")
    for i, m in enumerate(all_metrics):
        fi = i + 1
        f.write(f"Fold {fi} {str(FOLD_TEST_SEQS[fi]):<18} {m['base_f1']:>8.3f} {m['deep_f1']:>8.3f} {m['hds_f1']:>8.3f}\n")
    f.write(f"{'AVG':<25} {avg_bf:>8.3f} {avg_df:>8.3f} {avg_hf:>8.3f}\n")
print(f"\n저장: {OUT_DIR}")
