# HDS 실험 설계서 (Experiment Design)

**대상 논문**: Human-in-the-Loop Neuro-Symbolic Drift Anticipation for Reliable Visual SLAM
**목표 마감**: ~2026-09
**작성**: 2026-06-12 · 전략/매트릭스는 [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md), 구조는 [`ARCHITECTURE.md`](ARCHITECTURE.md)

---

## 0. 연구 질문과 기여

**핵심 주장 (검증된 방향):**
> 로봇 센서가 원천적으로 인지할 수 없는 운영자 지식을, 자연어로 SLAM 드리프트 위험에 주입한다.
> 이를 통해 SOTA 신경 예측기(DeepSEE)만으론 놓치는 드리프트를 사전에 회피한다.

**기여 경계 (성과 부각 지점):**
- DeepSEE(신경 예측기)는 **선행연구이자 강한 베이스라인** — 본 연구가 이긴다.
- 본 연구 기여 = **(C1) 센서 비관측 지식의 자연어 주입(LLM 기호 레이어)** + **(C2) 선행경보→완화 클로즈드루프**.
- 게이트 실험(§3-bis, EXPERIMENT_PLAN)에서 LLM이 규칙 대비 100% vs 45%로 우위 확인 → C1 정당.

---

## 1. 가설

| ID | 가설 | 검증 실험 |
|---|---|---|
| **H1** | DeepSEE는 in-domain에서 유효하나 zero-shot 도메인에서 신뢰성이 급락한다 | E3 |
| **H2** | 운영자 자연어 입력(LLM)은 키워드 규칙이 원리적으로 못 푸는 위험을 처리한다 (개방어휘·부정·위험하향·합성추론) | E1 |
| **H3** | HDS 선행경보로 완화 동작을 트리거하면 실로봇 궤적오차(ATE)·추적실패가 감소한다 | E2 |
| **H4** | 각 레이어(G, Symbolic)는 base 대비 조기경보 재현율/lead-time을 증가시킨다 | E4 |

---

## 2. 실험 E1 — Human Study: LLM vs 규칙 (가설 H2)

게이트 실험(`gate_llm_vs_rules.py`)의 **논문용 격상 버전**. 자기충족 위험 제거가 목적.

### 설계
- **독립변수**: 위험 변환기 {키워드 규칙(베이스라인), LLM} — within-scenario.
- **종속변수**: 정답 밴드 적중(정확도), 위험 방향(부호) 정확도.
- **자극(시나리오)**: sensor-blind 케이스 N≥40개. 4범주 균형 — 개방어휘 / 위험하향 / 합성·인과 / 부정문 + 대조군.

### 절차 (자기충족 제거)
1. **운영자 P명(P≥3)이 시나리오 텍스트를 직접 작성** (저자 작성 금지). 실제 로봇 주행 영상/경로 미리보기를 보고 자유 서술.
2. **정답(GT) 라벨**: 별도 SLAM 전문가 Q명(Q≥2)이 각 시나리오의 올바른 위험 방향·강도를 독립 라벨 → 합의(불일치 시 토론). Cohen's κ 보고.
3. **규칙 베이스라인은 제3자가 튜닝** (저자가 약하게 만들지 않았음을 보장). 공개 키워드 사전 사용.
4. LLM은 `temperature=0`, 응답 캐싱(재현성). 프롬프트 공개.

### 지표 & 성공 기준
- 전체 정확도(LLM vs 규칙), **규칙 실패 케이스에서 LLM 복구율**, 위험하향(−δ) 케이스 정확도.
- 통계: McNemar 검정(쌍체 정확도), 부트스트랩 CI.
- **성공**: 규칙 실패 케이스 LLM 복구율 ≥ 70% AND 전체 정확도 차 ≥ 30%p (게이트 예비결과: 100% / +55%p).

### 예비 결과 (게이트, 2026-06-12)
규칙 45% vs LLM 100%, 규칙 실패 6/6 복구. → 본 설계로 재현 시 유의미 예상.

---

## 3. 실험 E2 — G1 클로즈드루프 완화 (가설 H3) ★ 본선 핵심

선행경보가 **실제 행동을 바꿔 드리프트를 줄이는가**. 로봇 학회의 핵심 증거.

### 셋업
- 플랫폼: Unitree G1 + RealSense D435i, ORB-SLAM3(RGBD), HDS 실시간(`hds_g1_local.py`).
- **드리프트 기준(reference)**: G1엔 모캡 부재 → 다음 중 택1로 pseudo-GT 확보:
  - (권장) 모캡룸 1곳에서 측정, 또는
  - 고정밀 LiDAR-SLAM(예: FAST-LIO)을 동시 구동해 reference 궤적으로, 또는
  - 동일 경로 왕복 후 loop-closure 보정 궤적을 reference로.
- 사전지식 주입: 각 시도 전 운영자가 다가올 sensor-blind 구간을 자연어로 입력.

### 조건 (within-route, 반복 R≥5회/조건)
| 조건 | 예측기 | 완화 동작 |
|---|---|---|
| C0 baseline | 없음 | 없음 (정속 주행) |
| C1 DeepSEE-only | DeepSEE | ALERT 시 완화 |
| C2 HDS (full) | HDS(DS+G+Symbolic) | ALERT 시 완화 |

**완화 정책** (ALERT 시 택일·고정): 보행속도 ↓(예 −50%) / 키프레임 삽입율 ↑ / relocalization 탐색 강화 / 회전 감속. 사전 등록·고정.

### 시나리오 (차이가 터지는 곳을 의도 설계)
- 반복 텍스처 복도, 유리·거울 벽, 동적 보행자, 역광 모퉁이, 무텍스처 천장 등 **DeepSEE도 약한 sensor-blind 구간** 포함.

### 지표 & 성공 기준
- **ATE / RPE**(reference 대비), **추적실패(track-lost) 횟수·시간**, ALERT lead-time, 완화 개입 빈도.
- 통계: 조건 간 Wilcoxon signed-rank, mean±std (R회 반복).
- **성공**: C2가 C0 대비 ATE 유의 감소; **C2가 C1 대비** sensor-blind 구간에서 추가 감소(= 휴먼주입 효과).

---

## 4. 실험 E3 — Zero-shot 일반화 & In-domain 검증 (가설 H1)

오프라인, 결정론적. 하니스 `eval/eval_offline.py` 이미 구현.

### 데이터 & 절차
- **In-domain**: SenseTime test fold → base DeepSEE AUC. (예비: **AUC 0.803, CI 0.777–0.828**)
- **Zero-shot**: EuRoC(구현 완료) + **TUM-VI, OpenLORIS(추론 재실행 필요)**.
- 이벤트: `Y_gt = log1p(10⁴·clip(RelErr,0.001,0.02)) > τ`, τ∈{4.2,4.6,5.0} 스윕(절대 임계).
- 점수: **causal 정규화**(과거만, 누수 없음) 주, 배포충실 고정캘리브 보조.

### 지표
EWR@{1,3,5}s, Precision, False-Alarm/min, alert duty%, mean-lead, AUC-ROC/PR, 부트스트랩 95% CI.

### 결과 (EuRoC, 배포 모델 May18, `eval/deployed_ewr.py`)
DS recall(EWR@1) 55.1% (precision 88%, AUC 0.61) → Full 81.9% (precision 81%), lead 5.5→6.4s.
**해석**: HDS 레이어가 조기경보 recall +27%p·lead +0.9초. 일반화는 도메인 불균일(EuRoC 0.65, TUMVI 0.33).

---

## 5. 실험 E4 — Ablation & 베이스라인 (가설 H4)

### Ablation (E3 하니스 내)
DeepSEE / DS+G / DS+Sym / Full — 각 레이어의 EWR·precision·lead 기여 분리.

### 베이스라인 (별도 구현 필요)
- 추적 특징점 수 임계, local BA chi2(`local_visual_BA_Err`) 임계
- 이미지 품질 가드레일 G 단독
- (가능 시) 기존 introspective SLAM failure 예측기 1종
→ HDS가 단순 휴리스틱·내부신호 임계보다 우월함을 입증.

---

## 6. 측정·분석 표준

- **EWR@t**: 드리프트 이벤트 직전 H초 내 최초 경보의 lead-time ≥ t인 이벤트 비율.
- **Precision**: 경보 onset 중 H초 내 실제 이벤트가 뒤따른 비율 (헛경보 차단).
- **재현성**: 전 실험 고정 seed, LLM `temperature=0`+캐싱, 코드·프롬프트·시나리오 공개.
- **통계**: 쌍체는 Wilcoxon/McNemar, 구간추정은 부트스트랩 95% CI, 다중비교 보정.

---

## 7. 타당성 위협 (Threats to Validity)

| 위협 | 대응 |
|---|---|
| Human study 자기충족 | 시나리오=운영자 작성, GT=전문가 합의, 규칙=제3자 튜닝 |
| pseudo-GT 부정확(G1) | LiDAR/모캡/loop-closure 중 다중 reference 교차검증 |
| 정규화 run 불일치(May13 vs May18) | 통계 동일성 검증 후 결과 재확인 |
| zero-shot 표본 부족 | EuRoC+TUM-VI+OpenLORIS 통합, CI 보고 |
| 완화정책 cherry-pick | 정책 사전등록·고정, 조건 간 동일 |
| G 오프라인 산출(csv 기반) | 라이브 cv2 산출과 스케일 차 명시, 상대비교로 한정 |

---

## 8. 실행 순서 & 타임라인 (마감 ~2026-09 가정)

| 주차 | 작업 | 산출 |
|---|---|---|
| W1–2 | E1 human study 설계·운영자 모집·시나리오 수집 | 시나리오셋 + GT |
| W2–4 | E2 G1 셋업(reference 궤적 파이프라인) + 완화정책 구현 | 클로즈드루프 시스템 |
| W4–7 | E2 본실험(조건×반복) + E1 분석 | ATE/추적 결과, H2 검정 |
| W6–8 | E3 TUM-VI/OpenLORIS 추론 + E4 베이스라인 | 일반화·ablation 표 |
| W8–10 | 정규화 불일치 검증, 그림, 통계 | 최종 figure/table |
| W10–12 | 작성·내부리뷰 | 투고 |

**선결 게이트(완료)**: LLM>규칙(E1 예비) GO. **다음 착수**: E2 reference 파이프라인 또는 E1 운영자 모집.
