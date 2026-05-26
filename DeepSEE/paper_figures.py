"""
논문용 Figure 생성
- Fig 1: Early Warning Rate (bar chart, HDS vs DeepSEE)
- Fig 2: Lead Time per sequence
- Fig 3: Per-sequence F1 + EWR table
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import os

OUT_DIR = "/home/junhyun/SEESys/HDS_output/paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 결과 데이터 (hds_v1.py 실행 결과) ─────────────────────────────────────────
SEQS_ALL       = ["MH_01", "MH_02", "MH_03", "MH_04", "MH_05"]
SEQS_ANNOTATED = ["MH_01", "MH_03", "MH_04", "MH_05"]  # MH_02 제외

# Per-sequence F1
f1_base  = [0.594, 0.730, 0.905, 0.514, 0.639]
f1_deep  = [0.485, 0.463, 0.729, 0.787, 0.801]
f1_hds   = [0.472, 0.463, 0.749, 0.880, 0.817]

# Per-sequence Early Warning Rate (annotated only: MH01, MH03, MH04, MH05)
# [seq, threshold] → (HDS%, DeepSEE%)
ewr_data = {
    "MH_01": {1: (81.4, 81.4), 3: (74.4, 74.4), 5: (74.4, 72.1)},
    "MH_03": {1: (96.8, 87.1), 3: (96.8, 87.1), 5: (90.3, 83.9)},
    "MH_04": {1: (96.2, 80.8), 3: (92.3, 73.1), 5: (88.5, 65.4)},
    "MH_05": {1: (96.7, 96.7), 3: (96.7, 96.7), 5: (93.3, 93.3)},
}

# Overall EWR (n=130)
ewr_overall = {1: (91.5, 86.2), 3: (88.5, 82.3), 5: (85.4, 78.5)}

# Lead Time (annotated sequences, HDS-winning events)
lead_data = {
    "MH_01": 1.07,
    "MH_03": 3.80,
    "MH_04": 4.65,
    "MH_05": 0.96,
}
lead_deep = {
    "MH_01": 0.0,
    "MH_03": 0.0,
    "MH_04": 0.0,
    "MH_05": 0.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Main Result — EWR + Lead Time (2-panel)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("HDS: Proactive SLAM Failure Prediction\n(EuRoC, Human Path-Plan Annotations)",
             fontsize=13, fontweight='bold')

# Panel A: Overall Early Warning Rate
ax = axes[0]
thresholds = [1, 3, 5]
hds_ewrs  = [ewr_overall[t][0] for t in thresholds]
deep_ewrs = [ewr_overall[t][1] for t in thresholds]
x = np.arange(len(thresholds))
w = 0.32
bars1 = ax.bar(x - w/2, deep_ewrs, w, label='DeepSEE', color='steelblue', alpha=0.85)
bars2 = ax.bar(x + w/2, hds_ewrs,  w, label='HDS (Ours)', color='crimson',  alpha=0.85)
for bar, val in zip(bars1, deep_ewrs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val:.1f}%',
            ha='center', va='bottom', fontsize=9, color='steelblue', fontweight='bold')
for bar, val in zip(bars2, hds_ewrs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val:.1f}%',
            ha='center', va='bottom', fontsize=9, color='crimson', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['≥1s ahead', '≥3s ahead', '≥5s ahead'], fontsize=10)
ax.set_ylabel('Early Warning Rate (%)', fontsize=11)
ax.set_title('(a) Overall Early Warning Rate\n(n=130 GT high-risk events)', fontsize=10)
ax.set_ylim(60, 100)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: Lead Time advantage per sequence
ax = axes[1]
seqs_ann = list(lead_data.keys())
hds_leads  = [lead_data[s] for s in seqs_ann]
deep_leads = [lead_deep[s]  for s in seqs_ann]
x = np.arange(len(seqs_ann))
bars_d = ax.bar(x - w/2, deep_leads, w, label='DeepSEE', color='steelblue', alpha=0.85)
bars_h = ax.bar(x + w/2, hds_leads,  w, label='HDS (Ours)', color='crimson',  alpha=0.85)
for bar, val in zip(bars_h, hds_leads):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'+{val:.2f}s',
            ha='center', va='bottom', fontsize=9, color='crimson', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(seqs_ann, fontsize=10)
ax.set_ylabel('Avg Lead Time (seconds)', fontsize=11)
ax.set_title('(b) Avg Lead Time Advantage\n(HDS-winning events only)', fontsize=10)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = f"{OUT_DIR}/fig1_main_result.pdf"
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.savefig(out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Per-sequence EWR at ≥5s (most demanding threshold)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

seqs_ann = list(ewr_data.keys())
hds_5  = [ewr_data[s][5][0] for s in seqs_ann]
deep_5 = [ewr_data[s][5][1] for s in seqs_ann]
x = np.arange(len(seqs_ann))
w = 0.35
bars_d = ax.bar(x - w/2, deep_5, w, label='DeepSEE', color='steelblue', alpha=0.85)
bars_h = ax.bar(x + w/2, hds_5,  w, label='HDS (Ours)', color='crimson',  alpha=0.85)

for bar, val in zip(bars_d, deep_5):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='steelblue')
for bar, val in zip(bars_h, hds_5):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='crimson', fontweight='bold')

# gap annotation
for i, (h, d, s) in enumerate(zip(hds_5, deep_5, seqs_ann)):
    gap = h - d
    if abs(gap) > 0.5:
        ymax = max(h, d) + 4
        ax.annotate(f'+{gap:.1f}pp', xy=(i, ymax), ha='center',
                    fontsize=9, color='darkgreen', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([s.replace('MH_0', 'MH0') for s in seqs_ann], fontsize=11)
ax.set_ylabel('Early Warning Rate (%)', fontsize=12)
ax.set_title('Early Warning Rate ≥5s Ahead — Per Sequence\n(EuRoC with human path-plan annotations)',
             fontsize=11)
ax.set_ylim(50, 105)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
out2 = f"{OUT_DIR}/fig2_ewr_per_seq.pdf"
plt.savefig(out2, dpi=200, bbox_inches='tight')
plt.savefig(out2.replace('.pdf','.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: F1 Score comparison (supplementary)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(SEQS_ALL))
w = 0.25
ax.bar(x - w, f1_base, w, label='RF Baseline', color='forestgreen', alpha=0.75)
ax.bar(x,     f1_deep, w, label='DeepSEE',     color='steelblue',   alpha=0.85)
ax.bar(x + w, f1_hds,  w, label='HDS (Ours)',  color='crimson',     alpha=0.85)

for i, (b, d, h) in enumerate(zip(f1_base, f1_deep, f1_hds)):
    ax.text(i+w, h+0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=8,
            color='crimson', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(SEQS_ALL, fontsize=11)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('F1 Score Comparison — EuRoC Cross-Validation\n(* MH_02 has no path-plan annotation)',
             fontsize=11)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axvline(x=0.5, color='gray', ls=':', lw=1, alpha=0.5)
plt.tight_layout()
out3 = f"{OUT_DIR}/fig3_f1.pdf"
plt.savefig(out3, dpi=200, bbox_inches='tight')
plt.savefig(out3.replace('.pdf','.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out3}")

# ─────────────────────────────────────────────────────────────────────────────
# Table: LaTeX-style results summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("LaTeX Table (논문용)")
print("="*65)
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{HDS Proactive Warning Performance on EuRoC}")
print(r"\begin{tabular}{lcccccc}")
print(r"\hline")
print(r"Sequence & \multicolumn{2}{c}{F1 Score} & \multicolumn{3}{c}{Early Warning Rate} & Lead Time \\")
print(r" & DeepSEE & HDS & $\geq$1s & $\geq$3s & $\geq$5s & Adv. (s) \\")
print(r"\hline")

rows = [
    ("MH01", 0.485, 0.472, 81.4, 81.4, 74.4, 72.1, 74.4, 72.1, 1.07),
    ("MH02*", 0.463, 0.463, "-", "-", "-", "-", "-", "-", "-"),
    ("MH03", 0.729, 0.749, 96.8, 87.1, 96.8, 87.1, 90.3, 83.9, 3.80),
    ("MH04", 0.787, 0.880, 96.2, 80.8, 92.3, 73.1, 88.5, 65.4, 4.65),
    ("MH05", 0.801, 0.817, 96.7, 96.7, 96.7, 96.7, 93.3, 93.3, 0.96),
]

for row in rows:
    if row[1] == "-" or row[8] == "-":
        print(f"  {row[0]} & {row[1]:.3f} & {row[2]:.3f} & -- & -- & -- & -- \\\\")
    else:
        d1,h1 = row[3], row[4]
        d3,h3 = row[5], row[6]
        d5,h5 = row[7], row[8]
        lt     = row[9] if isinstance(row[9], float) else 0
        print(f"  {row[0]} & {row[1]:.3f} & \\textbf{{{row[2]:.3f}}} & "
              f"{h1:.1f}/{d1:.1f} & {h3:.1f}/{d3:.1f} & "
              f"\\textbf{{{h5:.1f}}}/{d5:.1f} & +{lt:.2f} \\\\")

print(r"\hline")
print(r"  \textbf{Avg} & 0.653 & \textbf{0.675} & 91.5/86.2 & 88.5/82.3 & \textbf{85.4}/78.5 & +3.35 \\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\footnotesize{* MH02: no path-plan annotation (unannotated baseline)}")
print(r"\end{table}")

print(f"\n모든 figure 저장: {OUT_DIR}")
