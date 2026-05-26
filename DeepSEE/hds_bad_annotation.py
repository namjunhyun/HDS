"""
Good vs Bad annotation 비교 실험
- Good: 실제 위험 구간을 정확히 묘사한 경로 계획 입력
- Bad:  같은 구간을 완전히 반대로 묘사 (밝고 안전한 구간이라고 거짓 입력)
→ LLM이 bad input에서 낮은 risk_delta를 부여하면 EWR 개선 없음을 증명
"""
import os, json, re, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import anthropic
from dotenv import load_dotenv

load_dotenv("/home/junhyun/SEESys/DeepSEE/Training/.env")

RUN_DIR = "/home/junhyun/SEESys/DeepSEE/Training/runs/Apr04_00-06-40_AHRI-Junhyun"
OUT_DIR = "/home/junhyun/SEESys/HDS_output/poster_analysis"
SEQS    = ["MH_01_easy","MH_02_easy","MH_03_medium","MH_04_difficult","MH_05_difficult"]
CAM     = {"MH_01_easy":2912,"MH_02_easy":3014,"MH_03_medium":2700,"MH_04_difficult":2033,"MH_05_difficult":2273}
CAM_HZ  = 20.0

os.makedirs(OUT_DIR, exist_ok=True)

# ── Good annotations (실제 상황 정확히 묘사) ───────────────────────────────
GOOD_ANNOTS = {
    "MH_01_easy": [
        (2219, 2357, "Path plan indicates a dark machinery room with dense metallic pipes ahead. Low illumination combined with specular reflections will significantly degrade visual feature tracking."),
    ],
    "MH_03_medium": [
        (47,   176,  "Path plan enters a poorly lit industrial zone with only lateral lighting. Reduced illumination and dark shadows will significantly degrade feature detection."),
        (500,  870,  "Path plan shows a rapid maneuver zone ahead. Expect heavy camera shake and feature tracking instability."),
        (1150, 1320, "Upcoming low-light corridor in path plan. Poor illumination will reduce visual features."),
        (2070, 2240, "Path plan indicates unstable motion zone ahead. Camera blur expected."),
    ],
    "MH_04_difficult": [
        (77,   231,  "Path plan shows downward-looking viewpoint over dark industrial machinery at mission start. Low contrast scene with limited visual features anticipated."),
        (990,  1443, "Path plan enters extremely dark zone ahead. Near-zero ambient lighting — almost silhouette-only visibility. Severe drift risk."),
        (1358, 1589, "Path plan navigates through dense metallic tank and pipe cluster. Highly reflective surfaces and complex geometry cause severe feature confusion and mismatching."),
        (1630, 1826, "Path plan shows another dark low-visibility zone. Poor ambient lighting will cause repeated visual feature degradation."),
    ],
    "MH_05_difficult": [
        (0,    420,  "Unstable initial flight phase in path plan. High vibration expected."),
        (1070, 1530, "Dark low-feature zone detected ahead in path plan. Severe drift risk anticipated."),
        (1920, 2060, "Sudden trajectory change in path plan. Rapid motion change expected."),
    ],
}

# ── Bad annotations (실제와 정반대 묘사 — 안전하다고 잘못 기입) ──────────────
BAD_ANNOTS = {
    "MH_01_easy": [
        (2219, 2357, "Path plan shows a bright, well-lit open corridor ahead. Excellent ambient lighting and clean walls provide abundant visual features for reliable tracking."),
    ],
    "MH_03_medium": [
        (47,   176,  "Path plan enters a brightly lit industrial workspace with uniform overhead lighting. High contrast environment with rich textures expected — ideal for feature tracking."),
        (500,  870,  "Path plan shows a slow, stable traversal zone. Smooth motion with minimal vibration expected. Feature tracking should remain stable throughout."),
        (1150, 1320, "Upcoming well-lit corridor in path plan. Bright illumination and clear wall textures will provide strong visual features."),
        (2070, 2240, "Path plan indicates a stable, slow-motion zone ahead. No camera blur expected. Clean and featureful environment."),
    ],
    "MH_04_difficult": [
        (77,   231,  "Path plan shows an upward-looking viewpoint over a brightly lit workspace at mission start. High contrast scene with abundant visual features anticipated."),
        (990,  1443, "Path plan enters a well-lit zone ahead. Strong ambient lighting with clear textures — excellent conditions for visual SLAM tracking."),
        (1358, 1589, "Path plan navigates through an open, texture-rich environment. Smooth surfaces with clear geometric patterns provide excellent feature matching."),
        (1630, 1826, "Path plan shows a bright high-visibility zone. Strong lighting will ensure consistent visual feature availability throughout this segment."),
    ],
    "MH_05_difficult": [
        (0,    420,  "Stable initial flight phase in path plan. Low vibration and smooth motion expected. Feature tracking will be highly reliable."),
        (1070, 1530, "Bright, feature-rich zone detected ahead in path plan. Excellent visual conditions with strong ambient lighting. Minimal drift risk."),
        (1920, 2060, "Gradual, smooth trajectory in path plan. No sudden motion changes expected. Camera remains stable throughout."),
    ],
}

# ── LLM 호출 ───────────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def get_risk_delta(human_text, seq_name, label=""):
    prompt = f"""You are a SLAM drift risk expert. A human operator reviewing the robot's path plan says:

"{human_text}"

Sequence: {seq_name}

Based on this proactive path-plan observation, decide how much to adjust the drift risk score (0~1 scale) for the upcoming zone.
Positive delta means increased risk, negative delta means reduced risk (safe conditions).

Return ONLY a JSON dict:
{{
  "risk_delta": <float -0.3 to 0.5>,
  "reason": "<one sentence>"
}}

Guidelines:
- Safe, bright, stable conditions: -0.30 ~ -0.05
- Neutral / uncertain: 0.0
- Minor degradation: 0.05 ~ 0.15
- Moderate (low light or fast motion): 0.15 ~ 0.30
- Severe (dark + no features + rapid motion): 0.30 ~ 0.50"""

    msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=128,
                                  messages=[{"role":"user","content":prompt}])
    raw = re.sub(r'```[a-z]*\n?','',msg.content[0].text.strip()).strip().rstrip('`')
    result = json.loads(raw)
    print(f"  [{label}] '{human_text[:60]}...' → risk_delta={result['risk_delta']:.3f}")
    return float(result['risk_delta'])

# ── Good/Bad risk_delta 수집 ───────────────────────────────────────────────
print("="*60)
print("Good annotations → LLM risk_delta")
print("="*60)
good_deltas = {}
for seq, annots in GOOD_ANNOTS.items():
    good_deltas[seq] = []
    for cs, ce, ht in annots:
        rd = get_risk_delta(ht, seq, "GOOD")
        good_deltas[seq].append((cs, ce, ht, rd))

print("\n" + "="*60)
print("Bad annotations → LLM risk_delta")
print("="*60)
bad_deltas = {}
for seq, annots in BAD_ANNOTS.items():
    bad_deltas[seq] = []
    for cs, ce, ht in annots:
        rd = get_risk_delta(ht, seq, "BAD ")
        bad_deltas[seq].append((cs, ce, ht, rd))

# ── HDS 계산 ──────────────────────────────────────────────────────────────
def norm(x): return (x-x.min())/(x.max()-x.min()+1e-8)

def apply_hds(est, seq, delta_info):
    n = len(est)
    dsy = np.zeros(n)
    for cs, ce, ht, rd in delta_info:
        ms = int(cs*n/CAM[seq]); me = min(int(ce*n/CAM[seq])+1, n)
        if rd >= 0:
            # 양수: gap-based boost (이미 높으면 건드리지 않음)
            gap = np.maximum(0, 0.65-est)
            dsy[ms:me] += np.minimum(gap, rd)[ms:me]
        else:
            # 음수: 직접 감소 (안전하다고 판단 → risk 낮춤)
            dsy[ms:me] += rd
    return np.clip(est+dsy, 0, 1)

def compute_ewr(est, hds, gt, seq, thr=0.6):
    n = len(gt); sec = (CAM[seq]/CAM_HZ)/n
    gb = (gt>thr).astype(int)
    events = [i for i in range(1,n) if gb[i]==1 and gb[i-1]==0]
    lh = []
    for ev in events:
        lb = max(0,ev-30)
        hc = next((i for i in range(lb,ev+1) if hds[i]>thr), None)
        lh.append((ev-hc)*sec if hc is not None else None)
    def rate(t): return sum(1 for v in lh if v is not None and v>=t)/len(lh)*100 if lh else 0
    return {1:rate(1),3:rate(3),5:rate(5)}, len(events)

# ── 결과 수집 ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EWR 결과")
print("="*60)

results = []
for fi, seq in enumerate(SEQS):
    if seq not in GOOD_ANNOTS:
        continue
    gt_raw  = np.load(f"{RUN_DIR}/SupervisedFinetune_{fi}_Y_gt.npy",  allow_pickle=True)
    est_raw = np.load(f"{RUN_DIR}/SupervisedFinetune_{fi}_Y_est.npy", allow_pickle=True)
    gt  = norm(np.concatenate([np.array(x).flatten() for x in gt_raw]))
    est = norm(np.concatenate([np.array(x).flatten() for x in est_raw]))

    hds_good = apply_hds(est, seq, good_deltas[seq])
    hds_bad  = apply_hds(est, seq, bad_deltas[seq])

    ewr_good, n_ev = compute_ewr(est, hds_good, gt, seq)
    ewr_bad,  _    = compute_ewr(est, hds_bad,  gt, seq)
    ewr_base, _    = compute_ewr(est, est,       gt, seq)

    avg_good_rd = np.mean([d[3] for d in good_deltas[seq]])
    avg_bad_rd  = np.mean([d[3] for d in bad_deltas[seq]])

    print(f"\n{seq} (n_events={n_ev})")
    print(f"  avg risk_delta — Good:{avg_good_rd:.3f}  Bad:{avg_bad_rd:.3f}")
    print(f"  EWR≥5s — DeepSEE:{ewr_base[5]:.1f}%  HDS(bad):{ewr_bad[5]:.1f}%  HDS(good):{ewr_good[5]:.1f}%")

    results.append({"seq":seq, "n_ev":n_ev,
        "rd_good":avg_good_rd, "rd_bad":avg_bad_rd,
        "base1":ewr_base[1],"base3":ewr_base[3],"base5":ewr_base[5],
        "bad1": ewr_bad[1], "bad3": ewr_bad[3],  "bad5": ewr_bad[5],
        "good1":ewr_good[1],"good3":ewr_good[3],"good5":ewr_good[5],
    })

# JSON 저장
with open(f"{OUT_DIR}/good_bad_annotation_results.json","w") as f:
    json.dump({"good_deltas":{s:[(a,b,c,d) for a,b,c,d in v] for s,v in good_deltas.items()},
               "bad_deltas": {s:[(a,b,c,d) for a,b,c,d in v] for s,v in bad_deltas.items()},
               "results": results}, f, indent=2)

# ── Figure ────────────────────────────────────────────────────────────────
C_DEEP = '#2980B9'
C_GOOD = '#C0392B'
C_BAD  = '#E67E22'

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("LLM Annotation Quality: Good vs Bad vs No Annotation", fontsize=13, fontweight='bold')

seq_labels = [r["seq"].replace("MH_0","MH").replace("_easy","(E)").replace("_medium","(M)").replace("_difficult","(D)") for r in results]
n_seqs = len(results)
x = np.arange(n_seqs); w = 0.25

# (a) EWR≥5s per sequence
ax = axes[0]
b1 = ax.bar(x - w, [r["base5"] for r in results], w, label='DeepSEE\n(no annot)', color=C_DEEP, alpha=0.85)
b2 = ax.bar(x,     [r["bad5"]  for r in results], w, label='HDS (bad annot)',      color=C_BAD,  alpha=0.85)
b3 = ax.bar(x + w, [r["good5"] for r in results], w, label='HDS (good annot)',     color=C_GOOD, alpha=0.85)
for i, r in enumerate(results):
    ax.text(i-w,   r["base5"]+0.8, f'{r["base5"]:.0f}%', ha='center', fontsize=8,   color=C_DEEP)
    ax.text(i,     r["bad5"] +0.8, f'{r["bad5"]:.0f}%',  ha='center', fontsize=8,   color=C_BAD)
    ax.text(i+w,   r["good5"]+0.8, f'{r["good5"]:.0f}%', ha='center', fontsize=8.5, color=C_GOOD, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(seq_labels, fontsize=10)
ax.set_ylabel('EWR ≥5s (%)'); ax.set_ylim(0, 110)
ax.set_title('(a) EWR ≥5s per Sequence', fontsize=11)
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# (b) 전체 weighted EWR (1s/3s/5s)
ax = axes[1]
n_tot = sum(r["n_ev"] for r in results)
def wavg(key): return sum(r[key]*r["n_ev"] for r in results)/n_tot

thresholds = ['≥1s', '≥3s', '≥5s']
base_ewrs = [wavg("base1"), wavg("base3"), wavg("base5")]
bad_ewrs  = [wavg("bad1"),  wavg("bad3"),  wavg("bad5")]
good_ewrs = [wavg("good1"), wavg("good3"), wavg("good5")]

xi = np.arange(3)
ax.bar(xi - w, base_ewrs, w, label='DeepSEE (no annot)', color=C_DEEP, alpha=0.85)
ax.bar(xi,     bad_ewrs,  w, label='HDS (bad annot)',     color=C_BAD,  alpha=0.85)
ax.bar(xi + w, good_ewrs, w, label='HDS (good annot)',    color=C_GOOD, alpha=0.85)
for i, (bv, bav, gv) in enumerate(zip(base_ewrs, bad_ewrs, good_ewrs)):
    ax.text(i-w, bv+0.5, f'{bv:.1f}%', ha='center', fontsize=9, color=C_DEEP)
    ax.text(i,   bav+0.5,f'{bav:.1f}%',ha='center', fontsize=9, color=C_BAD)
    ax.text(i+w, gv+0.5, f'{gv:.1f}%', ha='center', fontsize=9.5, color=C_GOOD, fontweight='bold')
ax.set_xticks(xi); ax.set_xticklabels(thresholds, fontsize=11)
ax.set_ylabel('EWR (%)'); ax.set_ylim(60, 105)
ax.set_title(f'(b) Overall EWR\n(n={n_tot} GT events, annotated seqs)', fontsize=11)
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# (c) LLM risk_delta 비교 (핵심: bad가 낮게 나왔음을 보여줌)
ax = axes[2]
rd_good = [r["rd_good"] for r in results]
rd_bad  = [r["rd_bad"]  for r in results]
bg = ax.bar(x - w/2, rd_good, w, label='Good annotation', color=C_GOOD, alpha=0.85)
bb = ax.bar(x + w/2, rd_bad,  w, label='Bad annotation',  color=C_BAD,  alpha=0.85)
for i, (g, b) in enumerate(zip(rd_good, rd_bad)):
    ax.text(i-w/2, g+0.005, f'{g:.3f}', ha='center', fontsize=8.5, color=C_GOOD, fontweight='bold')
    ax.text(i+w/2, b+0.005, f'{b:.3f}', ha='center', fontsize=8.5, color=C_BAD)
ax.set_xticks(x); ax.set_xticklabels(seq_labels, fontsize=10)
ax.set_ylabel('Avg LLM risk_delta'); ax.set_ylim(0, 0.55)
ax.set_title('(c) LLM-assigned risk_delta\n(Good input → high delta / Bad input → low delta)', fontsize=11)
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.axhline(0.15, color='gray', ls='--', lw=1, alpha=0.6, label='min effective threshold')

plt.tight_layout()
for ext in ['pdf','png']:
    plt.savefig(f"{OUT_DIR}/fig3_good_vs_bad_annotation.{ext}", dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved fig3 → {OUT_DIR}/fig3_good_vs_bad_annotation.png")
