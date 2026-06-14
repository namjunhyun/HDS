"""
VLM-비전 게이트 확인 — 낙관 편향 제거: 균형(균등 랜덤) 표본 + 다시퀀스(MH_03/04/05).
+ DeepSEE의 음의 상관이 정렬 버그인지 진짜인지 세그먼트 단위 sanity check.

사용: python3 vlm_vision_confirm.py [n_per_seq]   기본 20
"""
import os, sys, glob, random
import numpy as np
import pandas as pd
from vlm_vision_gate import vlm_risk, auc_roc, load_cache, EA, IMG, RUN, SEQS, TAU_ABS, CACHE
import json

SEQ_SET = ["MH_03_medium", "MH_04_difficult", "MH_05_difficult"]   # 실드리프트 있는 시퀀스


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rng = random.Random(0)

    cache = load_cache()
    client = None
    try:
        from dotenv import load_dotenv
        import anthropic
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
        if os.environ.get("ANTHROPIC_API_KEY"):
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except Exception as e:
        print(f"[anthropic 실패] {e}")

    print("=== 세그먼트 단위 sanity check: corr(DeepSEE est, Y_gt) — 프레임 매핑 없이 ===")
    for seq in SEQ_SET:
        fold = SEQS.index(seq)
        est = np.concatenate([np.asarray(x).flatten() for x in
              np.load(f"{RUN}/EuRoC_ZeroShot_{fold}_Y_est.npy", allow_pickle=True)])
        gt  = np.concatenate([np.asarray(x).flatten() for x in
              np.load(f"{RUN}/EuRoC_ZeroShot_{fold}_Y_gt.npy", allow_pickle=True)])
        print(f"  {seq}: corr(est,gt)={np.corrcoef(est,gt)[0,1]:+.3f}  "
              f"est[{est.min():.2f},{est.max():.2f}] gt[{gt.min():.2f},{gt.max():.2f}]")

    print("\n=== 균형 랜덤 표본: VLM vs DeepSEE (프레임 단위, GT=RelErr>%.2f) ===" % TAU_ABS)
    allrows = []
    for seq in SEQ_SET:
        fold = SEQS.index(seq)
        df = pd.read_csv(f"{EA}/{seq}/features.csv").reset_index(drop=True)
        frames = sorted(glob.glob(IMG.format(seq=seq) + "/*.png"))
        fr_ts = np.array([int(os.path.basename(f).split('.')[0]) / 1e9 for f in frames])
        est = np.concatenate([np.asarray(x).flatten() for x in
              np.load(f"{RUN}/EuRoC_ZeroShot_{fold}_Y_est.npy", allow_pickle=True)])
        n_seg = len(est)
        rel = df["RelativeError"].to_numpy()
        valid = [i for i in range(len(df)) if np.isfinite(rel[i])]
        pick = rng.sample(valid, min(n, len(valid)))

        rows = []
        for ridx in pick:
            t = df["TimeStamp"].iloc[ridx]
            fi = int(np.argmin(np.abs(fr_ts - t)))
            seg = min(int(ridx * n_seg / len(df)), n_seg - 1)
            try:
                vr = vlm_risk(client, cache, frames[fi])
            except Exception as e:
                print(f"  [VLM 오류] {e}"); vr = None
            rows.append({"seq": seq, "event": int(rel[ridx] > TAU_ABS), "vlm": vr,
                         "deepsee": float(est[seg])})
        json.dump(cache, open(CACHE, "w"))
        have = [r for r in rows if r["vlm"] is not None]
        lab = [r["event"] for r in have]
        if 0 < sum(lab) < len(lab):
            a_v = auc_roc(lab, [r["vlm"] for r in have])
            a_d = auc_roc(lab, [r["deepsee"] for r in have])
            print(f"  {seq}: n={len(have)} events={sum(lab)}  VLM_AUC={a_v:.3f}  DeepSEE_AUC={a_d:.3f}")
        else:
            print(f"  {seq}: n={len(have)} events={sum(lab)} (한쪽만 → 시퀀스별 AUC 생략)")
        allrows += have

    lab = [r["event"] for r in allrows]
    print("\n" + "=" * 60)
    print(f"전체 풀: n={len(allrows)} events={sum(lab)}")
    if 0 < sum(lab) < len(lab):
        a_v = auc_roc(lab, [r["vlm"] for r in allrows])
        a_d = auc_roc(lab, [r["deepsee"] for r in allrows])
        print(f"  VLM-자율 AUC = {a_v:.3f}")
        print(f"  DeepSEE  AUC = {a_d:.3f}")
        print("=" * 60)
        print("판정:", "VLM > DeepSEE (확인됨)" if a_v > a_d + 0.05 else
                       "DeepSEE >= VLM" if a_d > a_v + 0.05 else "비등")
    json.dump(allrows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              "vlm_vision_confirm_results.json"), "w"))


if __name__ == "__main__":
    main()
