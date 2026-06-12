"""
어노테이션 텍스트 → LLM risk_delta 변환 (재현 가능).

· Claude(temperature=0)로 운영자 문장을 위험 가산치로 변환. 동일 텍스트는 캐시 재사용.
· API 불가(키 없음/네트워크) 시 키워드 규칙으로 폴백 → 항상 결과 파일이 나옴.
· hds_g1_local.py의 SymbolicLayer와 동일 프롬프트/스케일 사용.

입력 : annotations_euroc.json
출력 : annotations_euroc_scored.json  (각 항목에 risk_delta, score_src 추가)

사용 : python3 score_annotations.py
"""
import os, json, re, hashlib

_HERE = os.path.dirname(os.path.abspath(__file__))
ANNOT = os.path.join(_HERE, "annotations_euroc.json")
OUT   = os.path.join(_HERE, "annotations_euroc_scored.json")
CACHE = os.path.join(_HERE, ".llm_cache.json")

PROMPT = """You are a SLAM drift risk expert. A human operator reviewing the robot's path plan says:
"{text}"

Decide how much to increase drift risk score (0~1 scale) for the upcoming zone.
Return ONLY JSON: {{"risk_delta": <float 0.0-0.5>, "reason": "<one sentence>"}}

Guidelines:
- Minor degradation: 0.05~0.15
- Moderate (low light or fast motion): 0.15~0.30
- Severe (dark + no features + rapid motion): 0.30~0.50"""


def _keyword_fallback(text):
    t = text.lower()
    if any(k in t for k in ["매우", "암흑", "severe"]):
        return 0.32
    if any(k in t for k in ["상당히", "급격", "moderate"]):
        return 0.20
    return 0.10


def _load_cache():
    try:
        with open(CACHE) as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    with open(ANNOT) as f:
        data = json.load(f)
    cache = _load_cache()

    client = None
    try:
        from dotenv import load_dotenv
        import anthropic
        load_dotenv(os.path.join(_HERE, "..", ".env"))
        if os.environ.get("ANTHROPIC_API_KEY"):
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except Exception as e:
        print(f"[anthropic 초기화 실패 → 폴백] {e}")

    n_llm = n_cache = n_fb = 0
    for seq, annots in data.items():
        for a in annots:
            key = hashlib.md5(a["text"].encode()).hexdigest()
            if key in cache:
                a["risk_delta"], a["score_src"] = cache[key], "cache"; n_cache += 1
                continue
            delta, src = None, "fallback"
            if client is not None:
                try:
                    msg = client.messages.create(
                        model="claude-sonnet-4-6", max_tokens=128, temperature=0.0,
                        messages=[{"role": "user", "content": PROMPT.format(text=a["text"])}])
                    raw = re.sub(r'```[a-z]*\n?', '', msg.content[0].text.strip()).strip().rstrip('`')
                    delta = float(json.loads(raw)["risk_delta"]); src = "llm"; n_llm += 1
                except Exception as e:
                    print(f"  [LLM 오류→폴백] {e}")
            if delta is None:
                delta = _keyword_fallback(a["text"]); n_fb += 1
            a["risk_delta"], a["score_src"] = round(float(delta), 4), src
            cache[key] = a["risk_delta"]

    with open(CACHE, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    with open(OUT, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n채점 완료: llm={n_llm} cache={n_cache} fallback={n_fb}")
    print(f"저장: {OUT}")
    if n_fb and not n_llm:
        print("주의: 전부 폴백값. API 키/네트워크 확보 후 재실행하면 LLM 값으로 갱신됨.")


if __name__ == "__main__":
    main()
