"""
E1 Human Study 분석 — 운영자 작성 시나리오 + 전문가 라벨 → LLM vs 규칙 논문 표.

입력 (docs/human_study/templates/):
  scenarios_raw.csv : scenario_id, operator_id, clip_ref, free_text   (운영자 작성)
  gt_labels.csv     : scenario_id, expert_id, label                    (전문가 라벨, 5단계)

처리:
  · 전문가 라벨 → 다수결(중앙값) 합의 밴드 + Fleiss κ(일치도)
  · 각 운영자 서술 → LLM(temp=0, 캐싱) 및 강한 키워드 규칙으로 risk_delta
  · 합의 밴드 적중 = 정답.  McNemar(쌍체) + 부트스트랩 CI + 규칙실패 LLM복구율.

사용: python3 analyze_human_study.py
"""
import os, json, re, csv, hashlib, math
from collections import defaultdict, Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
HS    = os.path.join(_HERE, "..", "..", "docs", "human_study", "templates")
SCEN_CSV = os.path.join(HS, "scenarios_raw.csv")
GT_CSV   = os.path.join(HS, "gt_labels.csv")
CACHE = os.path.join(_HERE, ".human_study_llm_cache.json")
OUT   = os.path.join(_HERE, "human_study_results.json")

LABELS  = ["strong_dec", "mild_dec", "neutral", "mild_inc", "strong_inc"]
ORDINAL = {"strong_dec": -2, "mild_dec": -1, "neutral": 0, "mild_inc": 1, "strong_inc": 2}
BAND    = {"strong_dec": (-0.50, -0.20), "mild_dec": (-0.30, -0.05), "neutral": (-0.10, 0.10),
           "mild_inc": (0.10, 0.30), "strong_inc": (0.30, 0.50)}

PROMPT = """You are a SLAM drift-risk expert. A human operator reviewing the robot's upcoming path says (Korean):
"{text}"

The operator may signal INCREASED risk, or REASSURE that an apparent sensor hazard is actually safe (REDUCES risk).
Return ONLY JSON: {{"risk_delta": <float -0.5..0.5>, "reason": "<one sentence>"}}
  positive = more drift risk · negative = safer than sensors suggest · ~0 = neutral"""


def rule_delta(text):
    t = text.lower()
    severe   = ["암흑", "칠흑", "매우 어두", "특징 없", "특징이 없", "no feature"]
    moderate = ["어두", "저조도", "dark", "블러", "blur", "흔들", "급격", "빠른", "회전",
                "반사", "유리", "거울", "젖", "반복"]
    if any(k in t for k in severe):   return 0.32
    if any(k in t for k in moderate): return 0.20
    return 0.10


def in_band(v, band):
    return band[0] - 1e-9 <= v <= band[1] + 1e-9


def fleiss_kappa(item_counts, n_raters):
    """item_counts: [{label:count}] per item. 동일 n_raters 가정."""
    N = len(item_counts)
    if N == 0 or n_raters < 2:
        return float("nan")
    p_j = {l: 0 for l in LABELS}
    P_i = []
    for c in item_counts:
        tot = sum(c.get(l, 0) for l in LABELS)
        if tot < 2:
            P_i.append(1.0);
            for l in LABELS: p_j[l] += c.get(l, 0)
            continue
        for l in LABELS: p_j[l] += c.get(l, 0)
        s = sum(c.get(l, 0) ** 2 for l in LABELS)
        P_i.append((s - tot) / (tot * (tot - 1)))
    total = sum(sum(c.get(l, 0) for l in LABELS) for c in item_counts)
    p_j = {l: p_j[l] / total for l in LABELS}
    Pbar = sum(P_i) / N
    Pe = sum(v * v for v in p_j.values())
    return (Pbar - Pe) / (1 - Pe) if (1 - Pe) > 1e-9 else float("nan")


def mcnemar_p(b, c):
    """discordant b,c → 양측 이항검정 정확 p."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * p)


def llm_delta(client, cache, text):
    key = hashlib.md5(text.encode()).hexdigest()
    if key in cache:
        return cache[key]
    if client is None:
        return None
    msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=128, temperature=0.0,
                                 messages=[{"role": "user", "content": PROMPT.format(text=text)}])
    raw = re.sub(r'```[a-z]*\n?', '', msg.content[0].text.strip()).strip().rstrip('`')
    d = round(float(json.loads(raw)["risk_delta"]), 4)
    cache[key] = d
    return d


def main():
    if not (os.path.exists(SCEN_CSV) and os.path.exists(GT_CSV)):
        print("입력 CSV 없음 — docs/human_study/templates/ 채운 뒤 실행."); return

    scen = list(csv.DictReader(open(SCEN_CSV, encoding="utf-8")))
    gt   = list(csv.DictReader(open(GT_CSV, encoding="utf-8")))

    # 전문가 라벨 집계 → 합의(중앙값) + Fleiss κ
    by_item = defaultdict(list)
    for r in gt:
        by_item[r["scenario_id"]].append(r["label"].strip())
    consensus, item_counts = {}, []
    for sid, labs in by_item.items():
        ords = sorted(ORDINAL[l] for l in labs if l in ORDINAL)
        med = ords[len(ords)//2]
        consensus[sid] = [k for k, v in ORDINAL.items() if v == med][0]
        item_counts.append(Counter(labs))
    n_experts = max((len(v) for v in by_item.values()), default=0)
    kappa = fleiss_kappa(item_counts, n_experts)

    # LLM 클라이언트
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    client = None
    try:
        from dotenv import load_dotenv
        import anthropic
        load_dotenv(os.path.join(_HERE, "..", ".env"))
        if os.environ.get("ANTHROPIC_API_KEY"):
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except Exception as e:
        print(f"[anthropic 초기화 실패 → LLM 스킵] {e}")

    rows = []
    for s in scen:
        sid = s["scenario_id"]
        if sid not in consensus:
            continue
        band = BAND[consensus[sid]]
        rd = rule_delta(s["free_text"])
        ld = llm_delta(client, cache, s["free_text"])
        rows.append({"sid": sid, "op": s.get("operator_id", ""), "text": s["free_text"],
                     "consensus": consensus[sid], "band": band,
                     "rule": rd, "rule_ok": in_band(rd, band),
                     "llm": ld, "llm_ok": (in_band(ld, band) if ld is not None else None)})
    json.dump(cache, open(CACHE, "w"), ensure_ascii=False, indent=2)

    have = [r for r in rows if r["llm_ok"] is not None]
    n = len(have)
    if n == 0:
        print("채점 가능한 항목 없음 (LLM 미응답 또는 입력 비어있음).")
        json.dump({"rows": rows, "fleiss_kappa": kappa}, open(OUT, "w"), ensure_ascii=False, indent=2)
        return
    rule_acc = sum(r["rule_ok"] for r in have) / n
    llm_acc  = sum(r["llm_ok"] for r in have) / n
    b = sum(1 for r in have if (not r["rule_ok"]) and r["llm_ok"])   # 규칙X LLM O
    c = sum(1 for r in have if r["rule_ok"] and (not r["llm_ok"]))   # 규칙O LLM X
    p = mcnemar_p(b, c)
    fail = [r for r in have if not r["rule_ok"]]
    rescue = sum(r["llm_ok"] for r in fail) / len(fail) if fail else 0.0

    print("="*64)
    print(f"E1 Human Study 결과  (항목 {n}, 시나리오 {len(by_item)}, 전문가 {n_experts}명)")
    print("="*64)
    print(f"전문가 합의 Fleiss κ : {kappa:.3f}")
    print(f"규칙 정확도          : {rule_acc*100:.1f}%")
    print(f"LLM  정확도          : {llm_acc*100:.1f}%")
    print(f"규칙실패 LLM 복구율   : {rescue*100:.1f}%  ({sum(r['llm_ok'] for r in fail)}/{len(fail)})")
    print(f"McNemar (b={b}, c={c}) : p={p:.4f}  {'(유의)' if p < 0.05 else '(비유의)'}")
    print("="*64)

    json.dump({"n_items": n, "n_experts": n_experts, "fleiss_kappa": round(kappa, 3),
               "rule_acc": round(rule_acc, 3), "llm_acc": round(llm_acc, 3),
               "llm_rescue_rate": round(rescue, 3), "mcnemar_b": b, "mcnemar_c": c,
               "mcnemar_p": round(p, 4), "rows": rows}, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
