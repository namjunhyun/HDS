# E1 Human Study 키트 — LLM vs 규칙 (논문 증거용)

게이트 실험(`eval/gate_llm_vs_rules.py`)을 **자기충족 위험 없이** 논문 증거로 격상하기 위한 키트.
설계 근거는 [`../EXPERIMENT_DESIGN.md`](../EXPERIMENT_DESIGN.md) §2 참조.

## 자기충족 위험 제거 3원칙
1. **시나리오 = 운영자(P≥3)가 직접 작성** — 저자 작성 금지.
2. **정답(GT) = SLAM 전문가(Q≥2)가 독립 라벨 후 합의** — Fleiss κ 보고.
3. **규칙 베이스라인 = 제3자가 공개 사전으로 튜닝** — 저자가 약하게 만들지 않음.

## 진행 순서
1. 운영자에게 [`OPERATOR_GUIDE.md`](OPERATOR_GUIDE.md) 배포 → `templates/scenarios_raw.csv` 작성.
2. 전문가에게 [`EXPERT_LABELING_GUIDE.md`](EXPERT_LABELING_GUIDE.md) 배포 → `templates/gt_labels.csv` 작성.
3. `python3 ../../DeepSEE/eval/analyze_human_study.py` 실행 → 표·통계 산출.

## 산출물
- LLM vs 규칙 정확도, **규칙 실패 케이스 LLM 복구율**, 위험하향(−δ) 정확도
- McNemar 검정(쌍체), 부트스트랩 95% CI, 전문가 합의 Fleiss κ
- → 논문 Table (E1)
