"""
궤적 평가 — 추정 vs reference 정렬 후 ATE/RPE (E2 클로즈드루프 메인 지표).

G1엔 모캡이 없으므로 reference(pseudo-GT)는 LiDAR-SLAM(FAST-LIO 등) 또는 loop-closure 보정 궤적.
둘 다 TUM 포맷:  `timestamp tx ty tz qx qy qz qw`  (공백 구분, 주석 #).

지원: 시간 association(최근접), Umeyama 정렬(SE3 또는 Sim3), ATE(RMSE/mean/median), RPE.
의존성: numpy 만.

사용:
  python3 traj_eval.py est.tum ref.tum --align sim3
  python3 traj_eval.py --selftest
"""
import sys, argparse
import numpy as np


def load_tum(path):
    ts, xyz, quat = [], [], []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        v = line.replace(",", " ").split()
        if len(v) < 8:
            continue
        ts.append(float(v[0])); xyz.append([float(v[1]), float(v[2]), float(v[3])])
        quat.append([float(v[4]), float(v[5]), float(v[6]), float(v[7])])
    return np.array(ts), np.array(xyz), np.array(quat)


def associate(t_est, t_ref, max_dt=0.02):
    """추정 타임스탬프마다 최근접 reference를 매칭 (|Δt|≤max_dt)."""
    pairs = []
    j = 0
    for i, t in enumerate(t_est):
        k = int(np.argmin(np.abs(t_ref - t)))
        if abs(t_ref[k] - t) <= max_dt:
            pairs.append((i, k))
    return pairs


def umeyama(src, dst, with_scale=True):
    """src(Nx3) → dst(Nx3) 최적 R,t,s (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = (D.T @ S) / len(src)
    U, d, Vt = np.linalg.svd(cov)
    Sgn = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sgn[2, 2] = -1
    R = U @ Sgn @ Vt
    s = (np.trace(np.diag(d) @ Sgn) / (S.var(0).sum())) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def ate(est_xyz, ref_xyz, align="sim3"):
    R, t, s = umeyama(est_xyz, ref_xyz, with_scale=(align == "sim3"))
    aligned = (s * (R @ est_xyz.T).T) + t
    err = np.linalg.norm(aligned - ref_xyz, axis=1)
    return {"rmse": float(np.sqrt((err**2).mean())), "mean": float(err.mean()),
            "median": float(np.median(err)), "std": float(err.std()),
            "max": float(err.max()), "n": len(err), "scale": float(s)}, aligned


def rpe(est_xyz, ref_xyz, delta=1):
    """프레임 간격 delta에 대한 상대 위치 오차 (정렬 불변)."""
    n = len(est_xyz)
    errs = []
    for i in range(n - delta):
        de = est_xyz[i + delta] - est_xyz[i]
        dr = ref_xyz[i + delta] - ref_xyz[i]
        errs.append(np.linalg.norm(de - dr))
    errs = np.array(errs)
    return {"rmse": float(np.sqrt((errs**2).mean())), "mean": float(errs.mean()), "n": len(errs)}


def evaluate(est_path, ref_path, align="sim3", max_dt=0.02):
    te, xe, _ = load_tum(est_path)
    tr, xr, _ = load_tum(ref_path)
    pairs = associate(te, tr, max_dt)
    if len(pairs) < 3:
        raise SystemExit(f"매칭점 부족({len(pairs)}). max_dt 또는 타임스탬프 확인.")
    ie = [p[0] for p in pairs]; ir = [p[1] for p in pairs]
    est, ref = xe[ie], xr[ir]
    a, _ = ate(est, ref, align)
    r = rpe(est, ref)
    print(f"매칭점 {len(pairs)} / 추정 {len(te)} · reference {len(tr)}  (정렬={align})")
    print(f"ATE  RMSE={a['rmse']*100:.2f}cm  mean={a['mean']*100:.2f}cm  median={a['median']*100:.2f}cm  max={a['max']*100:.2f}cm")
    print(f"RPE  RMSE={r['rmse']*100:.2f}cm  (Δ=1)" + (f"  scale={a['scale']:.4f}" if align == 'sim3' else ""))
    return {"ate": a, "rpe": r, "n_matched": len(pairs)}


def _selftest():
    rng = np.random.default_rng(0)
    n = 500
    t = np.linspace(0, 50, n)
    ref = np.c_[np.cos(t), np.sin(t), 0.1 * t]
    # 알려진 Sim3 변환 + 노이즈 2cm 적용 → ATE가 ~2cm 회복하는지
    th = 0.7; R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    s = 1.5; tr = np.array([3, -2, 1])
    est = (s * (R @ ref.T).T + tr) + rng.normal(0, 0.02, ref.shape)
    a, _ = ate(est, ref, "sim3")
    # est = 1.5·R·ref + t 이므로 est→ref 정렬은 역 scale 1/1.5≈0.667 을 복원해야 함
    print(f"[selftest] 주입 노이즈 2cm → ATE RMSE={a['rmse']*100:.2f}cm, 복원 scale={a['scale']:.3f} (기대 0.667)")
    assert abs(a['rmse'] - 0.02) < 0.006 and abs(a['scale'] - 1/1.5) < 0.05, "정렬 오류"
    print("[selftest] PASS — Umeyama 정렬·ATE 정상")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("est", nargs="?"); ap.add_argument("ref", nargs="?")
    ap.add_argument("--align", choices=["se3", "sim3"], default="sim3")
    ap.add_argument("--max_dt", type=float, default=0.02)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.est and args.ref:
        evaluate(args.est, args.ref, args.align, args.max_dt)
    else:
        ap.error("est ref 경로 또는 --selftest 필요")
