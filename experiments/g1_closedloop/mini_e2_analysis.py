"""
실증2 — G1 클로즈드루프 ATE 분석 파이프라인.
조건 C0(완화없음)/C1(DeepSEE-only)/C2(HDS full) × 반복 R 궤적 → ATE 비교 + Wilcoxon.

입력 디렉토리 규칙 (TUM 포맷, `ts tx ty tz qx qy qz qw`):
  est_C0_run1.tum, est_C1_run1.tum, est_C2_run1.tum, ref_run1.tum, ...
  (ref_run{r}.tum = 그 반복의 reference/pseudo-GT 궤적)

판정: C2 ATE < C1 ATE (HDS가 DeepSEE-only를 이김) — 특히 sensor-blind 경로에서.

사용:
  python3 mini_e2_analysis.py <traj_dir>
  python3 mini_e2_analysis.py --demo      # 합성 데이터로 파이프라인 검증
"""
import os, sys, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj_eval import load_tum, associate, ate

CONDS = ["C0", "C1", "C2"]


def wilcoxon_signed(a, b):
    """짝지은 a,b의 Wilcoxon signed-rank (scipy 있으면 사용, 없으면 부호검정 근사)."""
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b
    d = d[d != 0]
    if len(d) == 0: return float("nan"), float("nan")
    try:
        from scipy.stats import wilcoxon
        s, p = wilcoxon(a, b)
        return float(s), float(p)
    except Exception:
        # 부호검정: d>0 개수 ~ Binom(n,0.5)
        n = len(d); k = int((d > 0).sum())
        import math
        p = 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) * 0.5 ** n
        return float(k), float(min(1.0, p))


def eval_dir(traj_dir):
    refs = sorted(glob.glob(os.path.join(traj_dir, "ref_run*.tum")))
    runs = [re.search(r"ref_run(\d+)\.tum", os.path.basename(r)).group(1) for r in refs]
    ate_by_cond = {c: [] for c in CONDS}
    print(f"{'run':>4} " + " ".join(f"{c+'_ATE':>9}" for c in CONDS))
    for r in runs:
        _, rxyz, _ = load_tum(os.path.join(traj_dir, f"ref_run{r}.tum"))
        rt, _, _ = load_tum(os.path.join(traj_dir, f"ref_run{r}.tum"))
        row = {}
        for c in CONDS:
            p = os.path.join(traj_dir, f"est_{c}_run{r}.tum")
            if not os.path.exists(p): row[c] = None; continue
            et, exyz, _ = load_tum(p)
            pairs = associate(et, rt, max_dt=0.05)
            if len(pairs) < 5: row[c] = None; continue
            a, _ = ate(exyz[[i for i, _ in pairs]], rxyz[[j for _, j in pairs]], align="sim3")
            row[c] = a["rmse"]; ate_by_cond[c].append(a["rmse"])
        print(f"{r:>4} " + " ".join((f"{row[c]*100:>8.2f}cm" if row.get(c) is not None else f"{'--':>9}") for c in CONDS))
    return ate_by_cond


def report(ate_by_cond):
    print("\n" + "=" * 56)
    for c in CONDS:
        v = ate_by_cond[c]
        if v: print(f"  {c}: ATE = {np.mean(v)*100:.2f} ± {np.std(v)*100:.2f} cm  (n={len(v)})")
    print("=" * 56)
    if ate_by_cond["C1"] and ate_by_cond["C2"] and len(ate_by_cond["C1"]) == len(ate_by_cond["C2"]):
        s, p = wilcoxon_signed(ate_by_cond["C1"], ate_by_cond["C2"])
        red = (np.mean(ate_by_cond["C1"]) - np.mean(ate_by_cond["C2"])) / np.mean(ate_by_cond["C1"]) * 100
        print(f"  C2 vs C1: ATE {red:+.1f}% 변화, Wilcoxon p={p:.4f}")
        print("  판정:", "✅ HDS(C2)가 DeepSEE-only(C1) 이김" if red > 0 and p < 0.1
              else "❌ C2가 C1 못 이김 — 사람 레이어 가치 재검토")
    else:
        print("  (C1/C2 짝 반복 수 불일치 — Wilcoxon 생략)")


def make_demo(d):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0); n = 400; t = np.linspace(0, 40, n)
    ref = np.c_[np.cos(t), np.sin(t), 0.05 * t]
    drift = {"C0": 0.12, "C1": 0.07, "C2": 0.03}  # C2가 최소 드리프트(HDS 완화 효과)
    for r in range(1, 6):
        ts = 1000 + t
        np.savetxt(os.path.join(d, f"ref_run{r}.tum"),
                   np.c_[ts, ref, np.zeros((n, 3)), np.ones(n)], fmt="%.6f")
        for c in CONDS:
            cum = np.cumsum(rng.normal(0, drift[c], (n, 3)), axis=0) * 0.1
            est = ref + cum + rng.normal(0, 0.01, (n, 3))
            np.savetxt(os.path.join(d, f"est_{c}_run{r}.tum"),
                       np.c_[ts, est, np.zeros((n, 3)), np.ones(n)], fmt="%.6f")
    print(f"[demo] 합성 궤적 생성: {d} (C0>C1>C2 드리프트)\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        dd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_demo_traj")
        make_demo(dd); report(eval_dir(dd))
    elif len(sys.argv) > 1:
        report(eval_dir(sys.argv[1]))
    else:
        print(__doc__)
