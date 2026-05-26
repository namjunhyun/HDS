"""
포스터용 Figure v2
- Fig A: EWR 비교 (EuRoC + SenseTime 나란히)
- Fig B: Lead Time per sequence (EuRoC)
- Fig C: EWR ≥5s per sequence (EuRoC)
- Fig D: 시스템 요약 텍스트 박스
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import os

OUT_DIR = "/home/junhyun/SEESys/HDS_output/poster_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 색상 ──────────────────────────────────────────────────────────────────
C_HDS  = '#C0392B'   # crimson
C_DEEP = '#2980B9'   # steel blue
C_BASE = '#27AE60'   # green

FONT = {'family': 'sans-serif', 'size': 11}
matplotlib.rc('font', **FONT)

# ── 데이터 ────────────────────────────────────────────────────────────────
# EuRoC EWR
euroc_hds  = [91.5, 88.5, 85.4]
euroc_deep = [86.2, 82.3, 78.5]

# SenseTime EWR
st_hds  = [85.4, 63.4, 22.0]
st_deep = [75.6, 51.2,  9.8]

thresholds = ['≥1s', '≥3s', '≥5s']

# EuRoC per-sequence EWR ≥5s
seq_names = ['MH01', 'MH03', 'MH04', 'MH05']
seq_hds5  = [74.4, 90.3, 88.5, 93.3]
seq_deep5 = [72.1, 83.9, 65.4, 93.3]

# Lead Time (EuRoC)
lead_hds  = [1.07, 3.80, 4.65, 0.96]
lead_deep = [0.0,  0.0,  0.0,  0.0]

# ── Fig A: EWR 두 데이터셋 나란히 ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.suptitle('Early Warning Rate: HDS vs DeepSEE', fontsize=14, fontweight='bold', y=1.01)

x = np.arange(len(thresholds))
w = 0.35

for ax, hds_vals, deep_vals, title, ylim, n_label in zip(
    axes,
    [euroc_hds, st_hds],
    [euroc_deep, st_deep],
    ['(a) EuRoC MAV Dataset\n(n=130 GT events)', '(b) SenseTime Dataset\n(n=41 GT events)'],
    [(60, 100), (0, 100)],
    ['EuRoC', 'SenseTime']
):
    bars_d = ax.bar(x - w/2, deep_vals, w, label='DeepSEE', color=C_DEEP, alpha=0.85, zorder=3)
    bars_h = ax.bar(x + w/2, hds_vals,  w, label='HDS (Ours)', color=C_HDS, alpha=0.85, zorder=3)

    for bar, val in zip(bars_d, deep_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color=C_DEEP, fontweight='bold')
    for bar, val, dv in zip(bars_h, hds_vals, deep_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color=C_HDS, fontweight='bold')
        gap = val - dv
        if gap > 0.5:
            ax.text(x[list(hds_vals).index(val)], max(val, dv) + 5,
                    f'+{gap:.1f}pp', ha='center', fontsize=8.5, color='#16A085', fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(thresholds, fontsize=11)
    ax.set_ylabel('Early Warning Rate (%)', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(ylim)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
for ext in ['pdf', 'png']:
    plt.savefig(f"{OUT_DIR}/figA_ewr_both.{ext}", dpi=200, bbox_inches='tight')
plt.close()
print("Saved figA")

# ── Fig B: Lead Time per sequence ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(seq_names))
w = 0.35
ax.bar(x - w/2, lead_deep, w, label='DeepSEE', color=C_DEEP, alpha=0.85)
ax.bar(x + w/2, lead_hds,  w, label='HDS (Ours)', color=C_HDS, alpha=0.85)
for i, val in enumerate(lead_hds):
    ax.text(i + w/2, val + 0.05, f'+{val:.2f}s', ha='center', va='bottom',
            fontsize=9.5, color=C_HDS, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(seq_names, fontsize=11)
ax.set_ylabel('Avg Lead Time (seconds)', fontsize=11)
ax.set_title('(b) Lead Time Advantage per Sequence\n(EuRoC, HDS-winning events)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
for ext in ['pdf', 'png']:
    plt.savefig(f"{OUT_DIR}/figB_leadtime.{ext}", dpi=200, bbox_inches='tight')
plt.close()
print("Saved figB")

# ── Fig C: EWR ≥5s per sequence (EuRoC) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(seq_names))
w = 0.35
bars_d = ax.bar(x - w/2, seq_deep5, w, label='DeepSEE', color=C_DEEP, alpha=0.85)
bars_h = ax.bar(x + w/2, seq_hds5,  w, label='HDS (Ours)', color=C_HDS, alpha=0.85)
for bar, val in zip(bars_d, seq_deep5):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color=C_DEEP)
for bar, val, dv in zip(bars_h, seq_hds5, seq_deep5):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color=C_HDS, fontweight='bold')
    gap = val - dv
    if abs(gap) > 1:
        i = list(seq_hds5).index(val)
        ax.annotate(f'+{gap:.1f}pp', xy=(i, max(val,dv)+5), ha='center',
                    fontsize=8.5, color='#16A085', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(seq_names, fontsize=11)
ax.set_ylabel('Early Warning Rate (%)', fontsize=11)
ax.set_title('(c) EWR ≥5s Ahead — Per Sequence (EuRoC)', fontsize=11)
ax.set_ylim(50, 108)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
for ext in ['pdf', 'png']:
    plt.savefig(f"{OUT_DIR}/figC_ewr_per_seq.{ext}", dpi=200, bbox_inches='tight')
plt.close()
print("Saved figC")

# ── Fig D: 전체 합본 (포스터 1장) ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

# D-1: EuRoC EWR
ax1 = fig.add_subplot(gs[0, :2])
x = np.arange(3); w = 0.35
b1 = ax1.bar(x - w/2, euroc_deep, w, label='DeepSEE', color=C_DEEP, alpha=0.85)
b2 = ax1.bar(x + w/2, euroc_hds,  w, label='HDS (Ours)', color=C_HDS, alpha=0.85)
for bar, val in zip(b1, euroc_deep):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color=C_DEEP, fontweight='bold')
for bar, val, dv in zip(b2, euroc_hds, euroc_deep):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9.5, color=C_HDS, fontweight='bold')
    ax1.text(x[list(euroc_hds).index(val)], max(val,dv)+3.5,
             f'+{val-dv:.1f}pp', ha='center', fontsize=8.5, color='#16A085', fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(['≥1s ahead','≥3s ahead','≥5s ahead'], fontsize=11)
ax1.set_ylabel('Early Warning Rate (%)'); ax1.set_ylim(60,100)
ax1.set_title('(a) EuRoC: Overall EWR  (n=130 GT events)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=10); ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# D-2: SenseTime EWR
ax2 = fig.add_subplot(gs[0, 2])
b3 = ax2.bar(x - w/2, st_deep, w, label='DeepSEE', color=C_DEEP, alpha=0.85)
b4 = ax2.bar(x + w/2, st_hds,  w, label='HDS (Ours)', color=C_HDS, alpha=0.85)
for bar, val in zip(b3, st_deep):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8, color=C_DEEP)
for bar, val, dv in zip(b4, st_hds, st_deep):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8, color=C_HDS, fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(['≥1s','≥3s','≥5s'], fontsize=10)
ax2.set_ylabel('Early Warning Rate (%)'); ax2.set_ylim(0, 105)
ax2.set_title('(b) SenseTime EWR\n(n=41 GT events)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

# D-3: Lead Time
ax3 = fig.add_subplot(gs[1, :2])
x4 = np.arange(4); w2 = 0.35
ax3.bar(x4 - w2/2, lead_deep, w2, label='DeepSEE', color=C_DEEP, alpha=0.85)
ax3.bar(x4 + w2/2, lead_hds,  w2, label='HDS (Ours)', color=C_HDS, alpha=0.85)
for i, val in enumerate(lead_hds):
    ax3.text(i + w2/2, val + 0.05, f'+{val:.2f}s', ha='center', va='bottom',
             fontsize=10, color=C_HDS, fontweight='bold')
ax3.set_xticks(x4); ax3.set_xticklabels(seq_names, fontsize=11)
ax3.set_ylabel('Avg Lead Time (seconds)')
ax3.set_title('(c) Lead Time Advantage — EuRoC (HDS-winning events)', fontsize=11, fontweight='bold')
ax3.legend(fontsize=10); ax3.grid(axis='y', alpha=0.3)
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

# D-4: 요약 텍스트
ax4 = fig.add_subplot(gs[1, 2])
ax4.axis('off')
summary = (
    "HDS: Hybrid DeepSEE System\n"
    "─────────────────────────\n"
    "r̃ = r̂ + Δ_symbolic + Δ_hw\n\n"
    "• r̂  : DeepSEE prediction\n"
    "• Δsy: LLM symbolic rules\n"
    "       from path-plan input\n"
    "• Δhw: IMU guardrail\n\n"
    "Key Results (EuRoC)\n"
    "─────────────────────────\n"
    "EWR ≥5s: 85.4% vs 78.5%\n"
    "Lead Time: +3.35s avg\n"
    "Training-free (no GT used)"
)
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
         fontsize=10, va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#BDC3C7', alpha=0.9))
ax4.set_title('(d) System Overview', fontsize=11, fontweight='bold')

fig.suptitle('HDS: Proactive SLAM Failure Warning via Path-Plan Aware Hybrid Prediction',
             fontsize=13, fontweight='bold', y=1.01)

for ext in ['pdf', 'png']:
    plt.savefig(f"{OUT_DIR}/figD_poster_full.{ext}", dpi=200, bbox_inches='tight')
plt.close()
print("Saved figD (full poster)")
print(f"\n모든 figure: {OUT_DIR}")
