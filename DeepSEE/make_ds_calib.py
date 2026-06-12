"""
DeepSEE 출력 → [0,1] 고정 캘리브레이터 생성 (결정론적, 상태 없음).

배포 모델(SupervisedFinetune_1, May18 run)의 학습-시점 예측 분포(Y_est)로부터
고정 (lo, hi) 퍼센타일을 구해 runs/ds_calib.npz 에 저장한다.
추론 시 norm_running(rolling min-max + EMA) 대신 이 affine map을 쓰면
실행 이력에 무관하게 동일 raw_pred → 동일 점수가 보장된다.

사용:  python3 make_ds_calib.py [RUN_DIR]
"""
import sys, os, glob
import numpy as np

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/junhyun/SEESys/DeepSEE/Training/runs/May18_14-44-57_AHRI-Junhyun"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "ds_calib.npz")

LO_PCT, HI_PCT = 1.0, 99.0   # 이상치에 강인한 robust 범위


def _flatten(path):
    a = np.load(path, allow_pickle=True)
    try:
        return np.concatenate([np.asarray(x).flatten() for x in a])
    except Exception:
        return np.asarray(a, dtype=float).flatten()


def main():
    est_files = sorted(glob.glob(os.path.join(RUN_DIR, "SupervisedFinetune_*_Y_est.npy")))
    if not est_files:
        sys.exit(f"Y_est 파일 없음: {RUN_DIR}")

    pooled = np.concatenate([_flatten(p) for p in est_files])
    lo = float(np.percentile(pooled, LO_PCT))
    hi = float(np.percentile(pooled, HI_PCT))
    assert hi > lo, "hi <= lo : 캘리브레이션 실패"

    np.savez(OUT, lo=lo, hi=hi,
             lo_pct=LO_PCT, hi_pct=HI_PCT,
             n=pooled.size, run_dir=RUN_DIR)
    print(f"pooled est: n={pooled.size} min={pooled.min():.4f} max={pooled.max():.4f} "
          f"mean={pooled.mean():.4f}")
    print(f"calib: lo(p{LO_PCT})={lo:.4f}  hi(p{HI_PCT})={hi:.4f}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
