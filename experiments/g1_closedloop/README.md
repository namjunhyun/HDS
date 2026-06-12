# E2 — G1 클로즈드루프 완화 실험

선행경보(HDS)가 **실제 행동을 바꿔 드리프트를 줄이는가**. ICRA 핵심 결과.
설계 전문은 [`../../docs/EXPERIMENT_DESIGN.md`](../../docs/EXPERIMENT_DESIGN.md) §3.

## 구성 파일
| 파일 | 역할 | 상태 |
|---|---|---|
| `traj_eval.py` | 추정 vs reference 정렬(Umeyama) → ATE/RPE | ✅ 테스트 통과 (`--selftest`) |
| `mitigation_policy.py` | HDS 점수 → 완화 동작(감속·키프레임·reloc), 순수함수 | ✅ 테스트 통과 (`--selftest`) |
| `ros2_mitigation_node.py` | 정책을 ROS2 토픽에 연결 | 🤖 스켈레톤 (G1+ROS 필요) |

## 조건 (within-route, 반복 R≥5)
| 조건 | 예측기 | 완화 |
|---|---|---|
| C0 | 없음 | 없음 (정속) |
| C1 | DeepSEE only | 동일 정책 |
| C2 | HDS (DS+G+Symbolic) | 동일 정책 |

완화 정책은 **사전 등록·고정**(C1=C2 동일, 입력 점수만 다름) — cherry-pick 방지.

## reference(pseudo-GT) 확보 (택1, 다중이면 교차검증)
- 모캡룸 1곳, 또는 **LiDAR-SLAM(FAST-LIO 등) 동시구동**, 또는 동일경로 왕복 loop-closure 보정 궤적.
- 추정·reference 모두 **TUM 포맷**(`ts tx ty tz qx qy qz qw`)으로 저장.

## 실행 흐름 (G1 머신)
1. `hds_g1_local.py`에 HDS 점수 퍼블리셔 추가 → `/hds/score` (TODO, 스켈레톤 주석 참조).
2. `ros2 run ... ros2_mitigation_node.py` — `/hds/speed_scale`, `/hds/trigger_reloc` 발행.
3. 베이스 보행 컨트롤러가 `/hds/speed_scale` 반영하도록 연결.
4. 각 조건×반복 주행 후 추정궤적·reference 저장.
5. 평가:
   ```
   python3 traj_eval.py est_C2_run1.tum ref_run1.tum --align sim3
   ```
6. 조건 간 ATE/추적실패 Wilcoxon 검정 (mean±std, R회).

## 성공 기준
- C2가 C0 대비 ATE 유의 감소.
- **C2가 C1 대비** sensor-blind 구간에서 추가 감소 = 휴먼-인-더-루프 주입 효과(핵심 기여).
