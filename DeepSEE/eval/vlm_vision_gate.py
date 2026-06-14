"""
VLM-비전 게이트 — 자율 VLM이 카메라 프레임만으로 드리프트를 예측하나? DeepSEE를 이기나?

기존 게이트(gate_llm_vs_rules)는 사람 *텍스트* → LLM이었다. 여기선 사람 없이
**VLM이 EuRoC 프레임을 직접 보고** 드리프트 위험을 자율 판정 → GT 대비 DeepSEE와 AUC 비교.

이게 답하는 것:
  · VLM-자율이 DeepSEE보다 드리프트를 잘 예측하나? (이기면 DeepSEE-중심 프레임 위협)
  · "Language as Cost" 류 자율 VLM이 어디까지 커버하나 → 사람 층 residue 추정

데이터: SEESys/Datasets/.../EuRoC/<seq>/mav0/cam0/data/*.png (실제 프레임)
        euroc_error_analysis/data/<seq>/features.csv (RelativeError = 드리프트 GT)
        May10 run EuRoC_ZeroShot_<fold>_Y_est.npy (DeepSEE 예측)

사용: python3 vlm_vision_gate.py [seq] [n_per_class]   기본 MH_04_difficult 15
"""
import os, sys, json, re, glob, base64, hashlib
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
EA    = "/home/junhyun/euroc_error_analysis/data"
IMG   = "/home/junhyun/SEESys/Datasets/SenseTime/EuRoC/{seq}/mav0/cam0/data"
RUN   = "/home/junhyun/final_DeepSEE/DeepSEE/Training/runs/May10_23-39-25_AHRI-Junhyun"
CACHE = os.path.join(_HERE, ".vlm_vision_cache.json")
OUT   = os.path.join(_HERE, "vlm_vision_results.json")

SEQS  = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
TAU_ABS = 0.05          # RelativeError 이벤트 임계 (절대)
VLM_MODEL = "claude-sonnet-4-6"   # 결정론(temperature=0). 더 센 천장 원하면 claude-opus-4-8

PROMPT = """You are a visual-SLAM reliability expert. Look ONLY at this single camera frame from a robot.
Estimate the risk that visual SLAM (ORB-SLAM3) tracking will DRIFT or fail around here, judging from
what you can see: texture richness, repetitive/uniform patterns, reflective/transparent surfaces,
motion blur, lighting, feature density.
Return ONLY JSON: {"risk": <float 0.0-1.0>, "reason": "<one short phrase>"}"""


def auc_roc(label, score):
    label = np.asarray(label); score = np.asarray(score, float)
    P, N = label.sum(), (1 - label).sum()
    if P == 0 or N == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    tp = fp = auc = ptp = pfp = 0.0; prev = None
    for i in order:
        if score[i] != prev:
            auc += (fp - pfp) * (tp + ptp) / 2.0; ptp, pfp, prev = tp, fp, score[i]
        tp += label[i] == 1; fp += label[i] == 0
    auc += (fp - pfp) * (tp + ptp) / 2.0
    return auc / (P * N)


def load_cache():
    try:
        return json.load(open(CACHE))
    except Exception:
        return {}


def vlm_risk(client, cache, path):
    key = hashlib.md5((VLM_MODEL + ":" + os.path.basename(path)).encode()).hexdigest()
    if key in cache:
        return cache[key]
    if client is None:
        return None
    b64 = base64.standard_b64encode(open(path, "rb").read()).decode()
    kw = {} if VLM_MODEL.startswith("claude-opus-4-8") else {"temperature": 0.0}  # opus4.8는 temp 미지원
    msg = client.messages.create(
        model=VLM_MODEL, max_tokens=200, **kw,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": PROMPT}]}])
    raw = re.sub(r'```[a-z]*\n?', '', msg.content[0].text.strip()).strip().rstrip('`')
    v = float(json.loads(raw)["risk"]); cache[key] = v
    return v


def main():
    seq = sys.argv[1] if len(sys.argv) > 1 else "MH_04_difficult"
    k   = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    fold = SEQS.index(seq)

    df = pd.read_csv(f"{EA}/{seq}/features.csv").reset_index(drop=True)
    frames = sorted(glob.glob(IMG.format(seq=seq) + "/*.png"))
    fr_ts = np.array([int(os.path.basename(f).split('.')[0]) / 1e9 for f in frames])

    # DeepSEE est (세그먼트) → 프레임에 broadcast
    est_seg = np.concatenate([np.asarray(x).flatten()
              for x in np.load(f"{RUN}/EuRoC_ZeroShot_{fold}_Y_est.npy", allow_pickle=True)])
    n_seg = len(est_seg)

    # stratified 샘플: RelErr 상위 k + 하위 k
    rel = df["RelativeError"].to_numpy()
    valid = np.where(np.isfinite(rel))[0]
    order = valid[np.argsort(rel[valid])]
    pick = np.concatenate([order[:k], order[-k:]])

    cache = load_cache()
    client = None
    try:
        from dotenv import load_dotenv
        import anthropic
        load_dotenv(os.path.join(_HERE, "..", ".env"))
        if os.environ.get("ANTHROPIC_API_KEY"):
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except Exception as e:
        print(f"[anthropic 실패] {e}")

    rows = []
    print(f"seq={seq}  표본={len(pick)}  VLM={VLM_MODEL}  τ={TAU_ABS}")
    for j, ridx in enumerate(pick):
        t = df["TimeStamp"].iloc[ridx]
        fi = int(np.argmin(np.abs(fr_ts - t)))           # 타임스탬프 최근접 프레임
        seg = min(int(ridx * n_seg / len(df)), n_seg - 1)
        try:
            vr = vlm_risk(client, cache, frames[fi])
        except Exception as e:
            print(f"  [VLM 오류 {os.path.basename(frames[fi])}] {e}"); vr = None
        rows.append({"relerr": float(rel[ridx]), "event": int(rel[ridx] > TAU_ABS),
                     "vlm": vr, "deepsee": float(est_seg[seg])})
        if (j + 1) % 10 == 0:
            print(f"  {j+1}/{len(pick)} 완료")

    json.dump(cache, open(CACHE, "w"))
    have = [r for r in rows if r["vlm"] is not None]
    lab = [r["event"] for r in have]
    vlm = [r["vlm"] for r in have]
    deep = [r["deepsee"] for r in have]
    relv = [r["relerr"] for r in have]

    def corr(a):
        a = np.asarray(a, float); b = np.asarray(relv, float)
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")

    res = {"seq": seq, "n": len(have), "n_events": int(sum(lab)),
           "VLM":     {"AUC": round(auc_roc(lab, vlm), 3),  "corr_relerr": round(corr(vlm), 3)},
           "DeepSEE": {"AUC": round(auc_roc(lab, deep), 3), "corr_relerr": round(corr(deep), 3)}}
    print("\n" + "=" * 60)
    print(f"VLM-자율   : AUC={res['VLM']['AUC']}  corr={res['VLM']['corr_relerr']}")
    print(f"DeepSEE    : AUC={res['DeepSEE']['AUC']}  corr={res['DeepSEE']['corr_relerr']}")
    print("=" * 60)
    verdict = ("VLM이 DeepSEE 이상 → 자율 VLM 강함, DeepSEE-중심 위협"
               if res['VLM']['AUC'] >= res['DeepSEE']['AUC'] else
               "DeepSEE 우위 → 학습된 신호가 자율 VLM보다 드리프트 예측 잘함")
    print("판정:", verdict)
    res["verdict"] = verdict
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
