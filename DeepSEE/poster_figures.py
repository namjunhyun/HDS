"""
포스터용 Figure 생성 (크고 깔끔하게)
- Fig A: Early Warning Rate (HDS vs DeepSEE)
- Fig B: Lead Time per sequence
- Fig C: MH04 타임라인 예시
- Fig D: 결과 테이블
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyArrowPatch
import matplotlib.patches as mpatches
import os

OUT_DIR = "/home/junhyun/SEESys/HDS_output/poster_figures"
os.makedirs(OUT_DIR, exist_ok=True)

FONT_TITLE  = 20
FONT_LABEL  = 16
FONT_TICK   = 14
FONT_ANNOT  = 13

COLOR_HDS   = '#E8392A'   # red
COLOR_DEEP  = '#2E6DB4'   # blue
COLOR_BASE  = '#2E9E4F'   # green
COLOR_GT    = '#222222'

# ── 데이터 ─────────────────────────────────────────────────────────────────
ewr_overall = {1: (91.5, 86.2), 3: (88.5, 82.3), 5: (85.4, 78.5)}

ewr_seq = {
    "MH01": {1: (81.4, 81.4), 3: (74.4, 74.4), 5: (74.4, 72.1)},
    "MH03": {1: (96.8, 87.1), 3: (96.8, 87.1), 5: (90.3, 83.9)},
    "MH04": {1: (96.2, 80.8), 3: (92.3, 73.1), 5: (88.5, 65.4)},
    "MH05": {1: (96.7, 96.7), 3: (96.7, 96.7), 5: (93.3, 93.3)},
}

lead_hds  = {"MH01": 1.07, "MH03": 3.80, "MH04": 4.65, "MH05": 0.96}
lead_deep = {"MH01": 0.0,  "MH03": 0.0,  "MH04": 0.0,  "MH05": 0.0}

# ═══════════════════════════════════════════════════════════════════════════
# Fig A: Overall EWR (3-threshold bar chart)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

thrs   = [1, 3, 5]
labels = ['≥ 1s ahead', '≥ 3s ahead', '≥ 5s ahead']
hds_v  = [ewr_overall[t][0] for t in thrs]
deep_v = [ewr_overall[t][1] for t in thrs]
x = np.arange(3)
w = 0.36

b1 = ax.bar(x - w/2, deep_v, w, color=COLOR_DEEP, label='DeepSEE', alpha=0.88, zorder=3)
b2 = ax.bar(x + w/2, hds_v,  w, color=COLOR_HDS,  label='HDS (Ours)', alpha=0.88, zorder=3)

for bar, val in zip(b1, deep_v):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=FONT_ANNOT,
            color=COLOR_DEEP, fontweight='bold')
for bar, val, hval in zip(b2, hds_v, deep_v):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=FONT_ANNOT,
            color=COLOR_HDS, fontweight='bold')
    gap = val - hval
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3.0,
            f'+{gap:.1f}pp', ha='center', va='bottom', fontsize=11,
            color='darkgreen', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=FONT_TICK)
ax.set_ylabel('Early Warning Rate (%)', fontsize=FONT_LABEL)
ax.set_title('Proactive Warning Coverage\n(n = 130 GT high-risk events, EuRoC)',
             fontsize=FONT_TITLE, pad=12)
ax.set_ylim(60, 100)
ax.legend(fontsize=FONT_LABEL, loc='lower left')
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=FONT_TICK)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figA_ewr_overall.png", dpi=200, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/figA_ewr_overall.pdf", bbox_inches='tight')
plt.close()
print("Saved: figA_ewr_overall")

# ═══════════════════════════════════════════════════════════════════════════
# Fig B: Lead Time advantage per sequence
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

seqs = ["MH01", "MH03", "MH04", "MH05"]
lh = [lead_hds[s]  for s in seqs]
ld = [lead_deep[s] for s in seqs]
x  = np.arange(len(seqs))
w  = 0.36

b1 = ax.bar(x - w/2, ld, w, color=COLOR_DEEP, label='DeepSEE',    alpha=0.88, zorder=3)
b2 = ax.bar(x + w/2, lh, w, color=COLOR_HDS,  label='HDS (Ours)', alpha=0.88, zorder=3)

for bar, val in zip(b2, lh):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f'+{val:.2f}s', ha='center', va='bottom', fontsize=FONT_ANNOT,
            color=COLOR_HDS, fontweight='bold')

# avg line
avg = np.mean(lh)
ax.axhline(avg, color=COLOR_HDS, ls='--', lw=1.8, alpha=0.6, zorder=2)
ax.text(len(seqs)-0.1, avg+0.08, f'avg +{avg:.2f}s',
        ha='right', fontsize=12, color=COLOR_HDS, fontstyle='italic')

ax.set_xticks(x)
ax.set_xticklabels(seqs, fontsize=FONT_TICK)
ax.set_ylabel('Avg Lead Time (seconds)', fontsize=FONT_LABEL)
ax.set_title('Lead Time Advantage per Sequence\n(HDS-winning events only)',
             fontsize=FONT_TITLE, pad=12)
ax.set_ylim(0, 6.5)
ax.legend(fontsize=FONT_LABEL)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=FONT_TICK)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figB_leadtime.png", dpi=200, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/figB_leadtime.pdf", bbox_inches='tight')
plt.close()
print("Saved: figB_leadtime")

# ═══════════════════════════════════════════════════════════════════════════
# Fig C: MH04 타임라인 예시 (실제 npy 데이터 로드)
# ═══════════════════════════════════════════════════════════════════════════
RUN_DIR = "/home/junhyun/SEESys/DeepSEE/Training/runs/Apr04_00-06-40_AHRI-Junhyun"

try:
    import sys
    sys.path.insert(0, "/home/junhyun/SEESys/DeepSEE/Training")

    fold_idx = 3  # MH04_difficult
    gt_raw   = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_gt.npy",   allow_pickle=True)
    est_raw  = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_est.npy",  allow_pickle=True)
    base_raw = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_base.npy", allow_pickle=True)

    gt_flat   = np.concatenate([np.array(x).flatten() for x in gt_raw]).astype(float)
    est_flat  = np.concatenate([np.array(x).flatten() for x in est_raw]).astype(float)
    base_flat = base_raw.flatten().astype(float)

    gt_flat   = (np.exp(gt_flat)   - 1) / 10000
    est_flat  = (np.exp(est_flat)  - 1) / 10000
    base_flat = (np.exp(base_flat) - 1) / 10000

    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
    gt   = norm(gt_flat)
    deep = norm(est_flat)

    # HDS 재계산 (annotation 구간만)
    hds = deep.copy()
    n = len(deep)
    # annotation: model frames 12-38, 162-238, 223-262, 267-300 (with proactive shift)
    ann_zones = [(12, 38, 0.280), (162, 238, 0.380), (223, 262, 0.420), (267, 300, 0.280)]
    PROACTIVE = 15
    for s, e, delta in ann_zones:
        ms = max(0, s - PROACTIVE)
        me = max(0, e - PROACTIVE)
        seg = me - ms
        if seg <= 0: continue
        ramp = np.ones(seg)
        ramp[:min(10, seg)] = np.linspace(0.1, 1.0, min(10, seg))
        gap = np.maximum(0, 0.65 - deep[ms:me])
        applied = np.minimum(gap, delta * 0.88 * ramp)
        hds[ms:me] = np.clip(hds[ms:me] + applied, 0, 1)

    # 타임라인 (초 단위)
    n_cam = 2228
    cam_hz = 20.0
    t_max = n_cam / cam_hz
    t = np.linspace(0, t_max, n)

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.fill_between(t, 0, gt, alpha=0.15, color=COLOR_GT, label='_nolegend_')
    ax.plot(t, gt,   color=COLOR_GT,   lw=2.0, alpha=0.6, label='Ground Truth', zorder=3)
    ax.plot(t, deep, color=COLOR_DEEP, lw=2.0, alpha=0.85, label='DeepSEE', zorder=4)
    ax.plot(t, hds,  color=COLOR_HDS,  lw=2.5, alpha=0.95, label='HDS (Ours)', zorder=5)
    ax.axhline(0.6, color='gray', ls='--', lw=1.5, alpha=0.6, label='Risk threshold (0.6)')

    # 어노테이션 구간 음영
    cam_hz_m = cam_hz
    ann_zones_cam = [(12, 38), (162, 238), (223, 262), (267, 300)]
    for i, (ms, me) in enumerate(ann_zones_cam):
        ts = ms / n * t_max
        te = me / n * t_max
        ts_pro = max(0, ts - PROACTIVE / n * t_max)
        ax.axvspan(ts_pro, te, alpha=0.10, color='orange', zorder=1)
        if i == 0:
            ax.axvspan(ts_pro, te, alpha=0.10, color='orange', label='Human annotation zone')

    # Lead time 화살표 예시 (frame 49: HDS=9.13s, DeepSEE=0.91s)
    # event at frame ~49 → t_event
    ev_frame = 49
    t_event = ev_frame / n * t_max
    t_hds_warn  = t_event - 5.78
    t_deep_warn = t_event - 0.91
    y_arrow = 0.67

    ax.annotate('', xy=(t_event, y_arrow), xytext=(t_hds_warn, y_arrow),
                arrowprops=dict(arrowstyle='->', color=COLOR_HDS, lw=2.0))
    ax.annotate('', xy=(t_event, y_arrow+0.06), xytext=(t_deep_warn, y_arrow+0.06),
                arrowprops=dict(arrowstyle='->', color=COLOR_DEEP, lw=2.0))
    ax.text((t_hds_warn + t_event)/2, y_arrow - 0.04,
            'HDS: 5.78s ahead', ha='center', fontsize=11, color=COLOR_HDS, fontweight='bold')
    ax.text((t_deep_warn + t_event)/2, y_arrow + 0.10,
            'DeepSEE: 0.91s', ha='center', fontsize=11, color=COLOR_DEEP)
    ax.axvline(t_event, color='black', ls=':', lw=1.5, alpha=0.5)
    ax.text(t_event + 0.3, 0.05, 'GT event\nonset', fontsize=10, alpha=0.7)

    ax.set_xlabel('Time (seconds)', fontsize=FONT_LABEL)
    ax.set_ylabel('Normalized Risk Score', fontsize=FONT_LABEL)
    ax.set_title('MH04_difficult — Proactive vs Reactive Warning Example',
                 fontsize=FONT_TITLE, pad=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, t_max)
    ax.legend(fontsize=FONT_LABEL-1, loc='upper right')
    ax.grid(alpha=0.2, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/figC_timeline_MH04.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{OUT_DIR}/figC_timeline_MH04.pdf", bbox_inches='tight')
    plt.close()
    print("Saved: figC_timeline_MH04")

except Exception as e:
    print(f"Fig C 실패: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# Fig D: Per-sequence EWR ≥5s (가장 강한 threshold)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

seqs  = list(ewr_seq.keys())
hds_5  = [ewr_seq[s][5][0] for s in seqs]
deep_5 = [ewr_seq[s][5][1] for s in seqs]
x = np.arange(len(seqs))
w = 0.36

b1 = ax.bar(x - w/2, deep_5, w, color=COLOR_DEEP, label='DeepSEE',    alpha=0.88, zorder=3)
b2 = ax.bar(x + w/2, hds_5,  w, color=COLOR_HDS,  label='HDS (Ours)', alpha=0.88, zorder=3)

for bar, val in zip(b1, deep_5):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=FONT_ANNOT-1, color=COLOR_DEEP)
for bar, val, dval in zip(b2, hds_5, deep_5):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=FONT_ANNOT-1,
            color=COLOR_HDS, fontweight='bold')
    gap = val - dval
    if abs(gap) > 1.0:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4.5,
                f'+{gap:.1f}pp', ha='center', fontsize=11, color='darkgreen', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(seqs, fontsize=FONT_TICK)
ax.set_ylabel('Early Warning Rate (%)', fontsize=FONT_LABEL)
ax.set_title('Early Warning Rate ≥ 5s Ahead — Per Sequence',
             fontsize=FONT_TITLE, pad=12)
ax.set_ylim(50, 106)
ax.legend(fontsize=FONT_LABEL)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=FONT_TICK)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figD_ewr_per_seq.png", dpi=200, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/figD_ewr_per_seq.pdf", bbox_inches='tight')
plt.close()
print("Saved: figD_ewr_per_seq")

print(f"\n모든 포스터 figure 저장: {OUT_DIR}")
print("파일 목록:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
