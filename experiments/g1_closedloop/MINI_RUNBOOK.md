# 미니 E2 실행 가이드 (go/no-go, ~1–2시간)

목표: **"센서 맹점에서 사람 입력이 드리프트를 줄이나"**를 싸게 확인. 모캡 불필요(LiDAR reference).

## 셋업 (최소)
- **고정 경로**: 바닥 테이프로 표시. teleop으로 같은 선 따라 주행(또는 waypoint 고정).
- **구간 3개**: ① 정상(텍스처 풍부) → ② **sensor-blind**(반복텍스처 복도 또는 역광 모퉁이; 유리/거울은 LiDAR도 약해 미니선 피함) → ③ 정상.
- **동시 구동**: ORB-SLAM3(+HDS) + **LiDAR-SLAM(FAST-LIO 등, reference)**.
- **운영자**: 노트북에서 ② 진입 *직전* 한국어 한 줄 (C2만).

## 주행
- **C1 (사람 X, 완화 X)** × 3회
- **C2 (사람 사전지식 입력 → 완화 발동)** × 3회
- 모든 런: 같은 경로·시작점. *유일한 차이 = 사람 입력.*

## 녹화 (런마다 4개 파일, `cond`∈{C1,C2}, `r`=1..3)
| 파일 | 내용 |
|---|---|
| `orb_<cond>_<r>.tum` | ORB-SLAM3 궤적 (TUM: `ts tx ty tz qx qy qz qw`) |
| `lidar_<cond>_<r>.tum` | LiDAR-SLAM 궤적 (reference, 같은 TUM) |
| `hds_<cond>_<r>.csv` | HDS 로그 (헤더 `t,hds`) — `hds_g1_local`가 이미 CSV 로깅 |
| `segments_<cond>_<r>.csv` | 구간 (헤더 `t0,t1,label`, label∈`normal`/`blind`) — 테이프 통과 시각 기록 |

→ 한 폴더(`run_dir`)에 다 모음.

## 분석 (1줄)
```
python3 g1_closedloop_analyze.py <run_dir>
```
출력: 구간별 RPE(C1 vs C2) + 예측 AUC.

## go / no-go 판정
- ✅ **GO**: blind에서 **C2 RPE ≪ C1** (예: 30%+ 개선) AND normal에서 **C1≈C2** → 사람이 *센서 맹점에서만* 도움 → 풀 캠페인.
- ❌ **NO-GO**: blind에서 C2≈C1 → 완화 발동/시나리오 난이도 점검(§아래). 그래도 안 되면 Plan B(자율 시스템 논문)로 프레이밍 전환.

## null 나오면 점검 (좌절 전에)
1. **완화가 실제로 발동했나?** `mitig_log_*.csv`에서 speed_scale<1 확인.
2. **시나리오가 진짜 sensor-blind였나?** 그 구간에서 ORB 드리프트가 실제로 컸나(LiDAR 대비)? 안 컸으면 더 어려운 구간 필요.
3. **HDS 점수가 올랐나?** 운영자 입력이 risk_delta로 반영됐나.

## 검증
`python3 g1_closedloop_analyze.py --demo` → 합성 데이터로 파이프라인 동작 확인 완료
(blind 70% 개선 / normal 6% = 특이성 정상).
