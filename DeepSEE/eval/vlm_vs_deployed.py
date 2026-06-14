"""
#1 VLM vs 배포-DeepSEE 재비교 (올바른 모델). 같은 프레임/타임스탬프에서 공정 비교.
DeepSEE est = 배포 모델 시계열(deployed_euroc_temporal.npz). VLM = 캐시 재사용.

사용: python3 vlm_vs_deployed.py
"""
import os, glob
import numpy as np
from vlm_vision_gate import vlm_risk, auc_roc, load_cache, IMG, CACHE
import json

TEMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "deployed_euroc_temporal.npz")
TAU = 4.6
SEQS = ["MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
K = 12  # 클래스당 표본 (gt 상/하위)


def main():
    Z = np.load(TEMP)
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

    allrows = []
    for seq in SEQS:
        ts = Z[seq + "_ts"]; est = Z[seq + "_est"]; gt = Z[seq + "_gt"]
        frames = sorted(glob.glob(IMG.format(seq=seq) + "/*.png"))
        fr_ts = np.array([int(os.path.basename(f).split('.')[0]) / 1e9 for f in frames])
        order = np.argsort(gt)
        pick = np.concatenate([order[:K], order[-K:]])
        rows = []
        for i in pick:
            fi = int(np.argmin(np.abs(fr_ts - ts[i])))
            try:
                vr = vlm_risk(client, cache, frames[fi])
            except Exception as e:
                print(f"  [VLM 오류] {e}"); vr = None
            rows.append({"event": int(gt[i] > TAU), "vlm": vr, "deepsee": float(est[i])})
        json.dump(cache, open(CACHE, "w"))
        have = [r for r in rows if r["vlm"] is not None]
        lab = [r["event"] for r in have]
        if 0 < sum(lab) < len(lab):
            print(f"  {seq}: n={len(have)} ev={sum(lab)} "
                  f"VLM_AUC={auc_roc(lab,[r['vlm'] for r in have]):.3f} "
                  f"DeepSEE_AUC={auc_roc(lab,[r['deepsee'] for r in have]):.3f}")
        allrows += have

    lab = [r["event"] for r in allrows]
    av = auc_roc(lab, [r["vlm"] for r in allrows])
    ad = auc_roc(lab, [r["deepsee"] for r in allrows])
    print("\n" + "=" * 56)
    print(f"전체 풀: n={len(allrows)} events={sum(lab)}")
    print(f"  VLM-자율          AUC = {av:.3f}")
    print(f"  DeepSEE(배포 May18) AUC = {ad:.3f}")
    print("=" * 56)
    print("판정:", "VLM > DeepSEE" if av > ad + 0.05 else "DeepSEE >= VLM" if ad > av + 0.05 else "비등")
    json.dump({"n": len(allrows), "events": int(sum(lab)),
               "VLM_AUC": round(av, 3), "DeepSEE_deployed_AUC": round(ad, 3)},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlm_vs_deployed_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
