"""
게이트 실험 — LLM(휴먼 자연어)이 키워드 규칙을 실제로 이기는가?

DeepSEE(센서)와 이미지규칙이 둘 다 실패하도록 설계된 sensor-blind 시나리오에서,
사람 자연어 입력을 (a) LLM, (b) 강한 키워드 규칙 으로 risk_delta 변환 → 정답 밴드와 비교.

GO/NO-GO 판정:
  · 규칙이 틀린(rules-fail) 시나리오에서 LLM 정답률 ≥ 70%  AND  LLM 전체정확도 − 규칙 전체정확도 ≥ 0.3
  → GO (LLM 기여 입증, ICRA 풀악셀)
  아니면 NO-GO (LLM은 장식 → 헤드라인 피벗)

사용: python3 gate_llm_vs_rules.py
"""
import os, json, re, hashlib

_HERE = os.path.dirname(os.path.abspath(__file__))
SCEN  = os.path.join(_HERE, "sensor_blind_scenarios.json")
CACHE = os.path.join(_HERE, ".gate_llm_cache.json")
OUT   = os.path.join(_HERE, "gate_results.json")

PROMPT = """You are a SLAM drift-risk expert. A human operator reviewing the robot's upcoming path says (Korean):
"{text}"

The operator may signal INCREASED risk, or may REASSURE that an apparent sensor hazard is actually safe (which REDUCES risk).
Return ONLY JSON: {{"risk_delta": <float -0.3..0.5>, "reason": "<one sentence>"}}
  positive = more drift risk · negative = safer than sensors suggest · ~0 = neutral
Guidelines: severe +0.30..0.50 · moderate +0.15..0.30 · minor +0.05..0.15 · neutral 0 · reassuring-safe -0.30..-0.05"""


def rule_delta(text):
    """공정한 '강한' 키워드 규칙 베이스라인 (개방어휘 일부까지 커버)."""
    t = text.lower()
    severe   = ["암흑", "칠흑", "매우 어두", "특징 없", "특징이 없", "no feature"]
    moderate = ["어두", "저조도", "dark", "블러", "blur", "흔들", "급격", "빠른", "회전",
                "반사", "유리", "거울", "젖", "반복"]
    if any(k in t for k in severe):
        return 0.32
    if any(k in t for k in moderate):
        return 0.20
    return 0.10


def in_band(v, band):
    return band[0] - 1e-9 <= v <= band[1] + 1e-9


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
    scen = json.load(open(SCEN))["scenarios"]
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    client = None
    try:
        from dotenv import load_dotenv
        import anthropic
        load_dotenv(os.path.join(_HERE, "..", ".env"))
        if os.environ.get("ANTHROPIC_API_KEY"):
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except Exception as e:
        print(f"[anthropic 초기화 실패] {e}")

    rows = []
    print(f"{'id':<4}{'cat':<13}{'band':<14}{'rule':>6}{'ok':>4}{'LLM':>7}{'ok':>4}")
    print("-"*56)
    for s in scen:
        band = s["gt_band"]
        rd = rule_delta(s["text"])
        ld = llm_delta(client, cache, s["text"])
        r_ok = in_band(rd, band)
        l_ok = in_band(ld, band) if ld is not None else None
        rows.append({**{k: s[k] for k in ("id", "cat", "text", "gt_band")},
                     "rule": rd, "rule_ok": r_ok, "llm": ld, "llm_ok": l_ok})
        print(f"{s['id']:<4}{s['cat']:<13}[{band[0]:+.2f},{band[1]:+.2f}]{rd:>6.2f}"
              f"{'✓' if r_ok else '✗':>4}{('--' if ld is None else f'{ld:+.2f}'):>7}"
              f"{('?' if l_ok is None else ('✓' if l_ok else '✗')):>4}")

    json.dump(cache, open(CACHE, "w"), ensure_ascii=False, indent=2)

    n = len(rows)
    rule_acc = sum(r["rule_ok"] for r in rows) / n
    have_llm = [r for r in rows if r["llm_ok"] is not None]
    llm_acc = sum(r["llm_ok"] for r in have_llm) / len(have_llm) if have_llm else 0.0
    fail = [r for r in rows if not r["rule_ok"] and r["llm_ok"] is not None]
    llm_rescue = sum(r["llm_ok"] for r in fail) / len(fail) if fail else 0.0

    print("\n" + "="*56)
    print(f"규칙 전체정확도   : {rule_acc*100:.0f}%  ({sum(r['rule_ok'] for r in rows)}/{n})")
    print(f"LLM  전체정확도   : {llm_acc*100:.0f}%  ({sum(r['llm_ok'] for r in have_llm)}/{len(have_llm)})")
    print(f"규칙 실패 케이스   : {len(fail)}개")
    print(f"  └ 그중 LLM 복구 : {llm_rescue*100:.0f}%  ({sum(r['llm_ok'] for r in fail)}/{len(fail)})  ← 핵심 지표")
    go = (llm_rescue >= 0.70) and (llm_acc - rule_acc >= 0.30)
    print("="*56)
    print(f"판정: {'✅ GO — LLM 기여 입증. ICRA 헤드라인으로 밀어도 됨.' if go else '❌ NO-GO — LLM이 규칙 대비 우위 불충분. 헤드라인 피벗 고려.'}")
    print("="*56)

    json.dump({"rule_acc": rule_acc, "llm_acc": llm_acc, "llm_rescue_rate": llm_rescue,
               "go": go, "rows": rows}, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
