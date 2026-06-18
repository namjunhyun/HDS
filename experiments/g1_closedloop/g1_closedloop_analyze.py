"""
미니 E2 분석 — LiDAR reference로 시각 SLAM 드리프트 측정 + 사람 효과(C1 vs C2) + 예측 품질.

로봇이 녹화할 것 (run 디렉토리에, cond∈{C1,C2}, r=1..R):
  orb_<cond>_<r>.tum     : ORB-SLAM3 궤적 (시각, TUM)
  lidar_<cond>_<r>.tum   : LiDAR-SLAM 궤적 (reference, TUM)
  hds_<cond>_<r>.csv     : HDS 로그  (헤더 t,hds)
  segments_<cond>_<r>.csv: 구간  (헤더 t0,t1,label  label∈{normal,blind})

산출:
  · drift(t) = 시작구간 정렬 후 |ORB − LiDAR|  (cm 단위 진짜 드리프트)
  · 사람 효과: blind 구간 C2 drift < C1 drift ?  /  normal 구간 C1≈C2 ? (특이성)
  · 예측 품질: HDS 점수가 drift를 예측하나 (AUC, drift>5cm 기준)

사용:
  python3 g1_closedloop_analyze.py <run_dir>
  python3 g1_closedloop_analyze.py --demo
"""
import os, sys, glob, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj_eval import load_tum, associate, umeyama

DRIFT_EVENT_M = 0.05   # 5cm 이상 = 드리프트 이벤트
ALIGN_FRAC = 0.2


def auc_roc(label, score):
    label = np.asarray(label); score = np.asarray(score, float)
    P, N = label.sum(), (1 - label).sum()
    if P == 0 or N == 0: return float("nan")
    o = np.argsort(-score, kind="mergesort"); tp = fp = a = ptp = pfp = 0.0; prev = None
    for i in o:
        if score[i] != prev: a += (fp - pfp) * (tp + ptp) / 2; ptp, pfp, prev = tp, fp, score[i]
        tp += label[i] == 1; fp += label[i] == 0
    a += (fp - pfp) * (tp + ptp) / 2; return a / (P * N)


def assoc_traj(orb_p, lidar_p):
    """ORB↔LiDAR 시간정렬 → T, O(raw), L(raw), drift_abs(시작정렬 절대 드리프트)."""
    to, xo, _ = load_tum(orb_p); tl, xl, _ = load_tum(lidar_p)
    pairs = associate(to, tl, max_dt=0.05)
    if len(pairs) < 10: return None, None, None, None
    io = [p[0] for p in pairs]; il = [p[1] for p in pairs]
    O = xo[io]; L = xl[il]; T = to[io]
    k = max(5, int(ALIGN_FRAC * len(O)))
    R, t, _ = umeyama(O[:k], L[:k], with_scale=False)
    drift_abs = np.linalg.norm((R @ O.T).T + t - L, axis=1)
    return T, O, L, drift_abs


def rpe_segment(T, O, L, t0, t1, delta_s=1.0):
    """구간 [t0,t1]의 RPE = mean |ΔO − ΔL| (delta_s 윈도우). 정렬 불변 + 누적 무관 = 구간 국소 드리프트율."""
    m = np.where((T >= t0) & (T <= t1))[0]
    if len(m) < 3: return None
    Ts, Os, Ls = T[m], O[m], L[m]
    errs = []
    for i in range(len(Ts)):
        j = np.searchsorted(Ts, Ts[i] + delta_s)
        if j < len(Ts):
            errs.append(np.linalg.norm((Os[j]-Os[i]) - (Ls[j]-Ls[i])))
    return float(np.mean(errs)) if errs else None


def load_segments(p):
    segs = []
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            segs.append((float(r["t0"]), float(r["t1"]), r["label"].strip()))
    return segs


def analyze(run_dir):
    by = {"C1": {"normal": [], "blind": []}, "C2": {"normal": [], "blind": []}}
    pool_hds, pool_evt = [], []
    runs = 0
    for orb in sorted(glob.glob(os.path.join(run_dir, "orb_*.tum"))):
        base = os.path.basename(orb)[4:-4]                 # <cond>_<r>
        cond = base.split("_")[0]
        lidar = os.path.join(run_dir, f"lidar_{base}.tum")
        if not os.path.exists(lidar): continue
        T, O, L, drift = assoc_traj(orb, lidar)
        if T is None: continue
        runs += 1
        segs = load_segments(os.path.join(run_dir, f"segments_{base}.csv"))
        # 구간별 RPE(국소 드리프트율) — 누적 무관
        for t0, t1, lab in segs:
            rpe = rpe_segment(T, O, L, t0, t1)
            if rpe is not None and cond in by and lab in by[cond]:
                by[cond][lab].append(rpe)
        # 예측: HDS(t) interp → drift 이벤트
        hp = os.path.join(run_dir, f"hds_{base}.csv")
        if os.path.exists(hp):
            hd = list(csv.DictReader(open(hp)))
            ht = np.array([float(r["t"]) for r in hd]); hv = np.array([float(r["hds"]) for r in hd])
            hi = np.interp(T, ht, hv)
            pool_hds += list(hi); pool_evt += list((drift > DRIFT_EVENT_M).astype(int))

    print(f"분석 런 수: {runs}\n")
    print("=== 사람 효과 (구간별 RPE = 국소 드리프트율, cm/s) ===")
    print(f"{'구간':<10}{'C1':>10}{'C2':>10}{'개선':>10}")
    for lab in ["blind", "normal"]:
        c1 = by["C1"][lab]; c2 = by["C2"][lab]
        if c1 and c2:
            m1, m2 = np.mean(c1) * 100, np.mean(c2) * 100
            imp = (m1 - m2) / m1 * 100 if m1 else 0
            print(f"{lab:<10}{m1:>9.1f}{m2:>10.1f}{imp:>9.0f}%")
    print("\n해석: blind에서 C2≪C1(사람 효과) + normal에서 C1≈C2(특이성) → 사람이 *센서 맹점에서만* 도움.")
    if pool_evt and 0 < sum(pool_evt) < len(pool_evt):
        print(f"\n=== 예측 품질 (HDS vs 진짜 드리프트>5cm) ===")
        print(f"  AUC = {auc_roc(pool_evt, pool_hds):.3f}  (n={len(pool_evt)}, 드리프트이벤트 {sum(pool_evt)})")


def make_demo(d):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0); n = 600; t = np.linspace(0, 60, n)
    lidar = np.c_[np.cos(t), np.sin(t), 0.04 * t]        # reference(정확)
    # 구간: 0-20s normal, 20-40s BLIND(시각 열화), 40-60s normal
    segs = [(0, 20, "normal"), (20, 40, "blind"), (40, 60, "normal")]
    for cond in ["C1", "C2"]:
        for r in range(1, 4):
            ts = 1000 + t
            np.savetxt(os.path.join(d, f"lidar_{cond}_{r}.tum"),
                       np.c_[ts, lidar, np.zeros((n, 3)), np.ones(n)], fmt="%.6f")
            # blind 구간에서 시각 드리프트 누적: C1 많이, C2(완화) 적게
            rate = np.where((t >= 20) & (t <= 40), (0.020 if cond == "C1" else 0.006), 0.001)
            dxy = np.cumsum(rng.normal(0, 1, (n, 3)) * rate[:, None], axis=0)
            orb = lidar + dxy + rng.normal(0, 0.003, (n, 3))
            np.savetxt(os.path.join(d, f"orb_{cond}_{r}.tum"),
                       np.c_[ts, orb, np.zeros((n, 3)), np.ones(n)], fmt="%.6f")
            # HDS: blind 진입 직전부터 상승(C2는 완화로 실제 드리프트 작아도 경고는 같이 올림)
            hds = np.clip(np.where((t >= 18) & (t <= 40), 0.8, 0.1) + rng.normal(0, 0.05, n), 0, 1)
            with open(os.path.join(d, f"hds_{cond}_{r}.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["t", "hds"])
                for i in range(n): w.writerow([f"{ts[i]:.3f}", f"{hds[i]:.3f}"])
            with open(os.path.join(d, f"segments_{cond}_{r}.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["t0", "t1", "label"])
                for a, b, l in segs: w.writerow([1000 + a, 1000 + b, l])
    print(f"[demo] 합성 데이터: {d}  (blind에서 C1 드리프트↑, C2 완화↓)\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        dd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_demo_g1")
        make_demo(dd); analyze(dd)
    elif len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        print(__doc__)
