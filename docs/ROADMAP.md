# HDS 로드맵 — ICRA 2027 (~2026-09 마감)

작성 2026-06-13. 목표: **9월까지 "완벽한 HDS"**(드리프트 예지 + 휴먼-인-더-루프 + 완화) 완성·투고.
VLA/RL 드리프트-회피 항법은 **장기 비전 → 논문 Future Work**로만. 설계 근거: [`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md).

## 확정된 전략
- **기여 = "센서·규칙이 원리적으로 못 하는 운영자 지식의 자연어 주입"** (게이트: LLM 100% vs 규칙 45%).
- DeepSEE는 강한 베이스라인(in-domain AUC 0.80). 본 연구가 이긴다.
- **장기 비전**: HDS = 미래 RL/VLA 드리프트-회피 항법의 *보상 oracle*. (지금은 안 함 — 현실적 드리프트 시뮬이 미해결.)

## 크리티컬 패스
**G1 클로즈드루프(예측→완화→ATE↓)**가 ICRA 당락 + 최고 리스크 → 조기 착수·조기 de-risk.

---

## Phase 0 — De-risk & 토대 (W1–2, ~6월 말)
목표: 아키텍처 경계 확정 + 논문 타당성 막는 코드 결함 제거.
- [ ] **VLM-비전 게이트** (반나절): EuRoC 프레임 → VLM 자율 판정. "VLM이 S1~S6 어디까지 잡나 / DeepSEE 이기나 / 사람 층은 얼마나 얇아도 되나"를 데이터로 확정. → 기여 경계 고정.
- [ ] **정규화 불일치 검증**: 배포 모델(May18) vs norm 통계(May13) 동일성 확인. 다르면 결과 재산출.
- [ ] PSD 폴백 OOD: 평가 시 실제 PSD 프레임만 채점.
- [ ] **미니 E2 de-risk**(하루): S5·S6 2시나리오 × 3반복만 G1에서 → C2가 C1 이기는 기미 확인. 안 보이면 프레이밍 재검토.

## Phase 1 — 핵심 증거 (W3–6, 7월)
목표: "LLM>규칙" + 일반화 + 베이스라인 정량화.
- [ ] **E1 Human Study** (`docs/human_study/`): 운영자 3명+ 시나리오 작성, 전문가 2명+ GT 라벨 → `analyze_human_study.py` → McNemar·Fleiss κ. (게이트의 논문판 격상)
- [ ] **일반화 확장**: TUM-VI + OpenLORIS zero-shot 추론 재실행 → `eval_offline.py`에 추가. 표본↑ → CI 좁힘.
- [ ] **베이스라인**: BA chi2 임계, 추적특징점 임계, 가드레일 단독 — HDS 우월성 입증.

## Phase 2 — G1 클로즈드루프 (W5–9, 7월말–8월) ★ 메인
목표: 실로봇에서 경고→완화→드리프트 감소 입증.
- [ ] reference 궤적 파이프라인 1개 (LiDAR-SLAM/모캡/loop-closure) → TUM 포맷
- [ ] `hds_g1_local`의 `/dev/shm/hds_score` → `ros2_mitigation_node` → 보행 컨트롤러 속도게인 연결
- [ ] 조건 C0/C1/C2 × 반복 ≥5, sensor-blind 시나리오 7종(`SCENARIOS.md`)
- [ ] `traj_eval.py`로 ATE/RPE/추적실패 비교, Wilcoxon. **C2>C1>C0 (특히 sensor-blind 구간)** 확인.

## Phase 3 — 작성 & 마감 (W9–13, 8월–9월)
- [ ] 그림·표 (in-domain AUC, zero-shot ablation+CI, E1 human study, G1 ATE)
- [ ] **Future Work**: "HDS as drift-aware reward for end-to-end RL/VLA navigation" (장기 비전을 강점으로)
- [ ] 한계 정직 기술 (zero-shot modest, sim-to-real, 클라우드 LLM 의존+로컬 폴백)
- [ ] 내부 리뷰 / `/code-review ultra` / 초고 → 투고

---

## 리스크 & 대응
| 리스크 | 영향 | 대응 |
|---|---|---|
| G1 클로즈드루프에서 C2가 C1 못 이김 | ICRA 붕괴 | **Phase 0 미니 E2로 조기 확인** → 안 되면 RA-L로 후퇴 또는 재프레임 |
| reference 궤적 부정확 | ATE 신뢰도↓ | 다중 소스 교차검증 |
| Human study 운영자 모집 지연 | E1 지연 | W1에 모집 시작, 게이트가 예비 결과로 백업 |
| zero-shot 수치 modest | 설득력↓ | 표본 확대 + in-domain·G1로 보강, modest는 도메인시프트 동기로 정직 프레이밍 |

## 투고 결정 게이트 (8월 중순 체크)
- C2>C1 실로봇 입증 O + E1 유의 O → **ICRA 풀악셀**
- 둘 중 하나라도 약하면 → **RA-L 우선** (focused contribution)

## 산출물 매핑 (이미 있음)
`eval/{eval_offline,gate_llm_vs_rules,make/score_annotations}.py`, `make_ds_calib.py`,
`experiments/g1_closedloop/{traj_eval,mitigation_policy,ros2_mitigation_node}.py`,
`docs/human_study/`, `docs/{ARCHITECTURE,EXPERIMENT_PLAN,EXPERIMENT_DESIGN}.md`.
