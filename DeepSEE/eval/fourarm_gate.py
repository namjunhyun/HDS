"""
실증1 — 4-arm 게이트: 사람(LLM) vs VLM-on-frame vs 규칙. "사람만 잡는다"를 데이터로.

각 sensor-blind 시나리오에서 3개 소스가 위험을 flag 하나 비교:
  · 규칙(키워드, text)  · VLM(현재 프레임 image)  · 사람(자연어 text → LLM)
핵심 지표:
  · sensor-blind 위험(gt=1, VLM·규칙 둘 다 miss)에서 사람 복구율  ← Language as Cost와 차별
  · 외양모순 안전(gt=0, VLM·규칙 오발화)에서 사람 정정율          ← 사람만 가능(−델타)

프레임: clean=정상(위험 시야밖), degraded=프레임에 위험 보임 — EuRoC에서 RelErr 최저/최고 자동 선택.

사용: python3 fourarm_gate.py
"""
import os, sys, json, re, glob, base64, hashlib
import numpy as np, pandas as pd
from vlm_vision_gate import vlm_risk, CACHE, load_cache, IMG, VLM_MODEL

_HERE = os.path.dirname(os.path.abspath(__file__))
SCEN  = os.path.join(_HERE, "fourarm_scenarios.json")
EA    = "/home/junhyun/euroc_error_analysis/data"
OUT   = os.path.join(_HERE, "fourarm_results.json")

THR_VLM, THR_DELTA = 0.5, 0.15   # flag 임계
PROMPT = """You are a SLAM drift-risk expert. A human operator reviewing the upcoming path says (Korean):
"{text}"
The operator may signal INCREASED risk, or REASSURE that an apparent hazard is actually safe (REDUCES risk).
Return ONLY JSON: {{"risk_delta": <float -0.3..0.5>, "reason": "<one phrase>"}}"""


def rule_delta(t):
    t = t.lower()
    if any(k in t for k in ["암흑", "매우 어두", "특징 없", "특징점이 거의 없"]): return 0.32
    if any(k in t for k in ["어두", "어둡", "블러", "급격", "회전", "반사", "유리", "거울", "젖", "반복"]): return 0.20
    return 0.10


def llm_delta(client, cache, text):
    key = "fourarm:" + hashlib.md5(text.encode()).hexdigest()
    if key in cache: return cache[key]
    if client is None: return None
    msg = client.messages.create(model=VLM_MODEL, max_tokens=128, temperature=0.0,
        messages=[{"role": "user", "content": PROMPT.format(text=text)}])
    raw = re.sub(r'```[a-z]*\n?', '', msg.content[0].text.strip()).strip().rstrip('`')
    d = round(float(json.loads(raw)["risk_delta"]), 4); cache[key] = d; return d


def pick_frame(seq, which):
    df = pd.read_csv(f"{EA}/{seq}/features.csv")
    rel = df["RelativeError"].to_numpy()
    valid = np.where(np.isfinite(rel))[0]
    ridx = valid[np.argmin(rel[valid])] if which == "clean" else valid[np.argmax(rel[valid])]
    t = df["TimeStamp"].iloc[ridx]
    frames = sorted(glob.glob(IMG.format(seq=seq) + "/*.png"))
    fr_ts = np.array([int(os.path.basename(f).split('.')[0]) / 1e9 for f in frames])
    return frames[int(np.argmin(np.abs(fr_ts - t)))]


def main():
    scen = json.load(open(SCEN))["scenarios"]
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

    clean_f = pick_frame("MH_01_easy", "clean")
    degr_f  = pick_frame("MH_04_difficult", "degraded")
    print(f"clean 프레임: {os.path.basename(clean_f)}  degraded: {os.path.basename(degr_f)}\n")

    rows = []
    print(f"{'id':<4}{'cat':<22}{'gt':>3}{'rule':>6}{'VLM':>6}{'사람':>6}")
    print("-"*52)
    for s in scen:
        frame = clean_f if s["frame_type"] == "clean" else degr_f
        rd = rule_delta(s["text"]); vr = vlm_risk(client, cache, frame); ld = llm_delta(client, cache, s["text"])
        json.dump(cache, open(CACHE, "w"))
        rflag = int(rd >= THR_DELTA); vflag = int(vr >= THR_VLM) if vr is not None else None
        lflag = int(ld >= THR_DELTA) if ld is not None else None
        rows.append({**{k: s[k] for k in ("id","cat","gt_risky","frame_type")},
                     "rule": rd, "rflag": rflag, "vlm": vr, "vflag": vflag, "llm": ld, "lflag": lflag})
        print(f"{s['id']:<4}{s['cat']:<22}{s['gt_risky']:>3}{rd:>6.2f}"
              f"{(vr if vr is not None else float('nan')):>6.2f}{(ld if ld is not None else float('nan')):>6.2f}"
              f"   [{'R' if rflag else '-'}{'V' if vflag else '-'}{'H' if lflag else '-'}]")

    have = [r for r in rows if r["vflag"] is not None and r["lflag"] is not None]
    def acc(flagkey): return sum(1 for r in have if r[flagkey] == r["gt_risky"]) / len(have)
    # 핵심1: 위험인데 VLM·규칙 둘 다 miss → 사람이 복구?
    blind = [r for r in have if r["gt_risky"] == 1 and r["vflag"] == 0 and r["rflag"] == 0]
    recov = sum(1 for r in blind if r["lflag"] == 1) / len(blind) if blind else float('nan')
    # 핵심2: 안전인데 VLM·규칙 오발화 → 사람이 정정(flag 안 함)?
    over = [r for r in have if r["gt_risky"] == 0 and (r["vflag"] == 1 or r["rflag"] == 1)]
    corr = sum(1 for r in over if r["lflag"] == 0) / len(over) if over else float('nan')

    print("\n" + "="*52)
    print(f"소스별 탐지 정확도: 규칙={acc('rflag')*100:.0f}%  VLM={acc('vflag')*100:.0f}%  사람={acc('lflag')*100:.0f}%")
    print(f"★ sensor-blind 위험(규칙·VLM 둘다 miss {len(blind)}건) → 사람 복구율 = {recov*100:.0f}%")
    print(f"★ 외양모순 안전(규칙·VLM 오발화 {len(over)}건)      → 사람 정정율 = {corr*100:.0f}%")
    print("="*52)
    print("해석: 사람 복구·정정율이 높으면 → VLM·센서가 구조적으로 못 하는 걸 사람이 함 (노벨티 정당화).")
    json.dump({"acc":{"rule":acc('rflag'),"vlm":acc('vflag'),"human":acc('lflag')},
               "blind_recovery":recov,"overalarm_correction":corr,"rows":rows},
              open(OUT,"w"), ensure_ascii=False, indent=2)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
