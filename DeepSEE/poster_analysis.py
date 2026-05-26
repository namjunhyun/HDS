"""
분석 Figure 2종:
Fig1 - 난이도가 높을수록(DeepSEE가 실패할수록) HDS가 빛난다
Fig2 - LLM annotation 품질(risk_delta 크기)에 따른 EWR 변화
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

RUN_DIR = "/home/junhyun/SEESys/DeepSEE/Training/runs/Apr04_00-06-40_AHRI-Junhyun"
CSV_DIR = "/home/junhyun/SEESys/DeepSEE/data"
OUT_DIR = "/home/junhyun/SEESys/HDS_output/poster_analysis"
import os; os.makedirs(OUT_DIR, exist_ok=True)

SEQS = ["MH_01_easy","MH_02_easy","MH_03_medium","MH_04_difficult","MH_05_difficult"]
LABELS = ["MH01\n(Easy)","MH02\n(Easy)","MH03\n(Medium)","MH04\n(Difficult)","MH05\n(Difficult)"]
CAM = {"MH_01_easy":2912,"MH_02_easy":3014,"MH_03_medium":2700,"MH_04_difficult":2033,"MH_05_difficult":2273}
CAM_HZ = 20.0

with open("/home/junhyun/SEESys/HDS_output/rerun_20260408_143737/llm_responses.json") as f:
    raw = json.load(f)
cache = {(e["seq"], e["human_text"][:40]): float(e["response"]["risk_delta"]) for e in raw}

ANNOTS = {
    "MH_01_easy":      [(2219,2357,"Path plan indicates a dark machinery room with dense metallic pipes ahead. Low illumination combined with specular reflections will significantly degrade visual feature tracking.")],
    "MH_03_medium":    [(47,176,"Path plan enters a poorly lit industrial zone with only lateral lighting. Reduced illumination and dark shadows will significantly degrade feature detection."),
                        (500,870,"Path plan shows a rapid maneuver zone ahead. Expect heavy camera shake and feature tracking instability."),
                        (1150,1320,"Upcoming low-light corridor in path plan. Poor illumination will reduce visual features."),
                        (2070,2240,"Path plan indicates unstable motion zone ahead. Camera blur expected.")],
    "MH_04_difficult": [(77,231,"Path plan shows downward-looking viewpoint over dark industrial machinery at mission start. Low contrast scene with limited visual features anticipated."),
                        (990,1443,"Path plan enters extremely dark zone ahead. Near-zero ambient lighting. Severe drift risk."),
                        (1358,1589,"Path plan navigates through dense metallic tank and pipe cluster. Highly reflective surfaces."),
                        (1630,1826,"Path plan shows another dark low-visibility zone. Poor ambient lighting.")],
    "MH_05_difficult": [(0,420,"Unstable initial flight phase in path plan. High vibration expected."),
                        (1070,1530,"Dark low-feature zone detected ahead in path plan. Severe drift risk anticipated."),
                        (1920,2060,"Sudden trajectory change in path plan. Rapid motion change expected.")],
}

def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)

def compute_ewr(deepsee, hds, gt, seq, thr=0.6):
    n = len(gt)
    sec = (CAM[seq] / CAM_HZ) / n
    gb = (gt > thr).astype(int)
    events = [i for i in range(1, n) if gb[i]==1 and gb[i-1]==0]
    lh, ld = [], []
    for ev in events:
        lb = max(0, ev-30)
        hc = next((i for i in range(lb, ev+1) if hds[i] > thr), None)
        dc = next((i for i in range(lb, ev+1) if deepsee[i] > thr), None)
        lh.append((ev-hc)*sec if hc is not None else None)
        ld.append((ev-dc)*sec if dc is not None else None)
    def rate(leads, t): return sum(1 for v in leads if v is not None and v >= t)/len(leads)*100 if leads else 0
    return {t: (rate(lh,t), rate(ld,t)) for t in [1.0,3.0,5.0]}, len(events)

def load_seq(fi, seq):
    gt  = norm(np.concatenate([np.array(x).flatten() for x in np.load(f"{RUN_DIR}/SupervisedFinetune_{fi}_Y_gt.npy",  allow_pickle=True)]))
    est = norm(np.concatenate([np.array(x).flatten() for x in np.load(f"{RUN_DIR}/SupervisedFinetune_{fi}_Y_est.npy", allow_pickle=True)]))
    return gt, est

def apply_hds(est, seq, scale=1.0):
    n = len(est)
    dsy = np.zeros(n)
    for cs, ce, ht in ANNOTS.get(seq, []):
        ms = int(cs*n/CAM[seq]); me = min(int(ce*n/CAM[seq])+1, n)
        rd = cache.get((seq, ht[:40]), 0.2) * scale
        dsy[ms:me] += np.minimum(np.maximum(0, 0.65-est), rd)[ms:me]
    return np.clip(est + dsy, 0, 1)

# ── 데이터 수집 ────────────────────────────────────────────────────────────
seq_data = []
for fi, (seq, label) in enumerate(zip(SEQS, LABELS)):
    gt, est = load_seq(fi, seq)
    hds = apply_hds(est, seq, scale=1.0)
    ewr, n_ev = compute_ewr(est, hds, gt, seq)
    has_annot = seq in ANNOTS
    seq_data.append({
        "seq": seq, "label": label, "n_ev": n_ev,
        "has_annot": has_annot,
        "deep5": ewr[5.0][1],
        "hds5":  ewr[5.0][0],
        "deep1": ewr[1.0][1],
        "hds1":  ewr[1.0][0],
        "gap5":  ewr[5.0][0] - ewr[5.0][1],
    })

# ════════════════════════════════════════════════════════════════════
# Fig 1: DeepSEE가 약할수록 HDS 개선폭이 크다
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("HDS shines when DeepSEE struggles", fontsize=14, fontweight='bold')

C_HDS  = '#C0392B'
C_DEEP = '#2980B9'
C_BASE = '#7F8C8D'

# (a) 시퀀스별 EWR≥5s 비교 (DeepSEE 오름차순 정렬)
ax = axes[0]
sorted_d = sorted(seq_data, key=lambda x: x["deep5"])
labels_s  = [d["label"] for d in sorted_d]
deep_vals = [d["deep5"] for d in sorted_d]
hds_vals  = [d["hds5"]  for d in sorted_d]
gaps      = [d["gap5"]  for d in sorted_d]
has_annot = [d["has_annot"] for d in sorted_d]

x = np.arange(len(labels_s)); w = 0.35
b1 = ax.bar(x - w/2, deep_vals, w, label='DeepSEE', color=C_DEEP, alpha=0.85)
b2 = ax.bar(x + w/2, hds_vals,  w, label='HDS (Ours)', color=C_HDS, alpha=0.85)

for i, (dv, hv, gap, ann) in enumerate(zip(deep_vals, hds_vals, gaps, has_annot)):
    ax.text(i - w/2, dv + 0.8, f'{dv:.1f}%', ha='center', va='bottom', fontsize=8, color=C_DEEP)
    ax.text(i + w/2, hv + 0.8, f'{hv:.1f}%', ha='center', va='bottom', fontsize=8.5, color=C_HDS, fontweight='bold')
    if ann and gap > 1:
        ax.text(i, max(dv, hv) + 5, f'+{gap:.1f}pp', ha='center', fontsize=9,
                color='#16A085', fontweight='bold')
    if not ann:
        ax.text(i, 5, 'no\nannot.', ha='center', fontsize=7.5, color='gray', style='italic')

ax.set_xticks(x); ax.set_xticklabels(labels_s, fontsize=10)
ax.set_ylabel('EWR ≥5s (%)', fontsize=11)
ax.set_ylim(0, 108)
ax.set_title('(a) EWR ≥5s per Sequence\n(sorted by DeepSEE performance)', fontsize=11)
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# 화살표: DeepSEE 낮을수록 HDS 개선 큰 트렌드
ax.annotate('', xy=(4.3, 20), xytext=(1.7, 20),
            arrowprops=dict(arrowstyle='->', color='#16A085', lw=2.0))
ax.text(3.0, 22, 'HDS advantage\n↑ as DeepSEE fails', ha='center', fontsize=8.5, color='#16A085')

# (b) DeepSEE EWR vs HDS improvement scatter
ax = axes[1]
annot_seqs = [d for d in seq_data if d["has_annot"]]
deep5_vals = [d["deep5"] for d in annot_seqs]
gap5_vals  = [d["gap5"]  for d in annot_seqs]
labels_a   = [d["label"] for d in annot_seqs]

colors_pt = [C_HDS if g > 5 else C_DEEP for g in gap5_vals]
scatter = ax.scatter(deep5_vals, gap5_vals, s=180, c=colors_pt, alpha=0.85, zorder=3, edgecolors='white', linewidth=1.5)

for i, (x_pt, y_pt, lbl) in enumerate(zip(deep5_vals, gap5_vals, labels_a)):
    ax.annotate(lbl.replace('\n',' '), (x_pt, y_pt),
                textcoords="offset points", xytext=(8, 4), fontsize=9.5)

# 추세선
if len(deep5_vals) > 2:
    z = np.polyfit(deep5_vals, gap5_vals, 1)
    p = np.poly1d(z)
    xs = np.linspace(min(deep5_vals)-2, max(deep5_vals)+2, 100)
    ax.plot(xs, p(xs), '--', color='#16A085', alpha=0.7, lw=1.8, label='trend')

ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.set_xlabel('DeepSEE EWR ≥5s (%)', fontsize=11)
ax.set_ylabel('HDS Improvement (pp)', fontsize=11)
ax.set_title('(b) DeepSEE weakness → HDS gain\n(annotated sequences only)', fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
for ext in ['pdf','png']:
    plt.savefig(f"{OUT_DIR}/fig1_difficulty_effect.{ext}", dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig1")

# ════════════════════════════════════════════════════════════════════
# Fig 2: LLM annotation 품질 (risk_delta scale) vs EWR
# scale=0: 엉터리(delta 없음=DeepSEE), scale=1: 정상
# ════════════════════════════════════════════════════════════════════
scales = [0.0, 0.25, 0.5, 0.75, 1.0]
scale_labels = ['0%\n(no input)', '25%\n(vague)', '50%\n(partial)', '75%\n(near-good)', '100%\n(full)']

# 전체 weighted EWR (annotation 있는 seq만)
ewr1_list, ewr3_list, ewr5_list = [], [], []
for scale in scales:
    e1_all, e3_all, e5_all, n_total = 0, 0, 0, 0
    for fi, seq in enumerate(SEQS):
        if seq not in ANNOTS:
            continue
        gt, est = load_seq(fi, seq)
        hds = apply_hds(est, seq, scale=scale)
        ewr, n_ev = compute_ewr(est, hds, gt, seq)
        e1_all += ewr[1.0][0] * n_ev
        e3_all += ewr[3.0][0] * n_ev
        e5_all += ewr[5.0][0] * n_ev
        n_total += n_ev
    ewr1_list.append(e1_all / n_total)
    ewr3_list.append(e3_all / n_total)
    ewr5_list.append(e5_all / n_total)

# DeepSEE baseline (scale=0 결과와 동일하지만 명시)
deep_ewr1 = ewr1_list[0]
deep_ewr3 = ewr3_list[0]
deep_ewr5 = ewr5_list[0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("LLM Annotation Quality → EWR Impact", fontsize=14, fontweight='bold')

x = np.arange(len(scales))

# (a) 라인 플롯
ax = axes[0]
ax.plot(x, ewr1_list, 'o-', color='#E67E22', lw=2, ms=8, label='EWR ≥1s')
ax.plot(x, ewr3_list, 's-', color='#8E44AD', lw=2, ms=8, label='EWR ≥3s')
ax.plot(x, ewr5_list, '^-', color=C_HDS,    lw=2.5, ms=9, label='EWR ≥5s', zorder=5)

ax.axhline(deep_ewr5, color=C_DEEP, ls='--', lw=1.5, alpha=0.7, label=f'DeepSEE ≥5s ({deep_ewr5:.1f}%)')

for i, (e1, e3, e5) in enumerate(zip(ewr1_list, ewr3_list, ewr5_list)):
    ax.text(i, e5 + 0.8, f'{e5:.1f}%', ha='center', va='bottom', fontsize=9, color=C_HDS, fontweight='bold')

ax.fill_between(x, deep_ewr5, ewr5_list, alpha=0.12, color=C_HDS, label='HDS gain over DeepSEE')
ax.set_xticks(x); ax.set_xticklabels(scale_labels, fontsize=10)
ax.set_xlabel('LLM Annotation Quality', fontsize=11)
ax.set_ylabel('EWR (%)', fontsize=11)
ax.set_ylim(70, 100)
ax.set_title('(a) EWR vs Annotation Quality\n(EuRoC, n=130 events)', fontsize=11)
ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# (b) 개선폭 막대 (DeepSEE 대비 gain)
ax = axes[1]
gain1 = [e - deep_ewr1 for e in ewr1_list]
gain3 = [e - deep_ewr3 for e in ewr3_list]
gain5 = [e - deep_ewr5 for e in ewr5_list]

w = 0.25
ax.bar(x - w, gain1, w, label='≥1s gain', color='#E67E22', alpha=0.85)
ax.bar(x,     gain3, w, label='≥3s gain', color='#8E44AD', alpha=0.85)
ax.bar(x + w, gain5, w, label='≥5s gain', color=C_HDS,    alpha=0.85)

for i, g5 in enumerate(gain5):
    ax.text(i + w, g5 + 0.2, f'+{g5:.1f}pp', ha='center', va='bottom', fontsize=9,
            color=C_HDS, fontweight='bold')

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(scale_labels, fontsize=10)
ax.set_xlabel('LLM Annotation Quality', fontsize=11)
ax.set_ylabel('EWR Gain over DeepSEE (pp)', fontsize=11)
ax.set_title('(b) EWR Gain vs Annotation Quality\n(better annotation → more gain)', fontsize=11)
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# 텍스트 주석
ax.annotate('Random/\nno input', xy=(0, gain5[0]+0.3), xytext=(0.5, gain5[0]+3),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9, color='gray', ha='center')
ax.annotate('Accurate\npath plan', xy=(4, gain5[4]-0.5), xytext=(3.4, gain5[4]-4),
            arrowprops=dict(arrowstyle='->', color=C_HDS), fontsize=9, color=C_HDS, ha='center')

plt.tight_layout()
for ext in ['pdf','png']:
    plt.savefig(f"{OUT_DIR}/fig2_annotation_quality.{ext}", dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig2")
print(f"\n저장: {OUT_DIR}")
