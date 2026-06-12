"""
완화 정책 (E2 클로즈드루프) — HDS 점수 → 완화 동작 파라미터.

설계 원칙(타당성):
  · 정책은 실험 전 **사전 등록·고정** (cherry-pick 방지). 조건 C1/C2 동일 정책, 입력 점수만 다름.
  · 순수 함수 `decide()` 는 로봇 없이 테스트 가능. ROS2 노드는 이를 감싸는 얇은 껍데기.

동작(고정):
  · 경보 시 보행속도 스케일 ↓ (위험에 비례, 하한 SPEED_MIN)
  · 키프레임 삽입율 ↑ (추적 강건성)
  · HIGH 임계 상향 돌파 시 relocalization 1회 트리거(히스테리시스로 떨림 방지)

사용(정책 테스트): python3 mitigation_policy.py --selftest
"""
from dataclasses import dataclass


@dataclass
class MitigationConfig:
    alert_thr: float = 0.60     # 경보 임계
    high_thr:  float = 0.80     # relocalization 트리거 임계
    speed_min: float = 0.40     # 속도 스케일 하한 (정지 방지)
    kf_boost:  float = 1.8      # 키프레임 삽입율 배수(경보 시)


@dataclass
class MitigationAction:
    speed_scale: float          # [speed_min, 1.0]
    keyframe_rate_mult: float   # ≥1.0
    trigger_reloc: bool
    active: bool


class MitigationPolicy:
    def __init__(self, cfg: MitigationConfig = MitigationConfig()):
        self.cfg = cfg
        self._reloc_armed = True   # 히스테리시스: high 아래로 내려가면 재무장

    def decide(self, hds_score: float) -> MitigationAction:
        c = self.cfg
        if hds_score < c.alert_thr:
            if hds_score < c.high_thr * 0.9:
                self._reloc_armed = True
            return MitigationAction(1.0, 1.0, False, active=False)
        # 경보: 위험이 클수록 더 감속 (alert_thr→1.0 구간을 1.0→speed_min 로 사상)
        frac = (hds_score - c.alert_thr) / max(1e-6, 1.0 - c.alert_thr)
        speed = max(c.speed_min, 1.0 - frac * (1.0 - c.speed_min))
        reloc = False
        if hds_score >= c.high_thr and self._reloc_armed:
            reloc = True; self._reloc_armed = False
        return MitigationAction(round(speed, 3), c.kf_boost, reloc, active=True)


def _selftest():
    p = MitigationPolicy()
    seq = [0.2, 0.5, 0.62, 0.85, 0.9, 0.7, 0.3, 0.85]
    relocs = 0
    for s in seq:
        a = p.decide(s)
        relocs += a.trigger_reloc
        assert p.cfg.speed_min - 1e-9 <= a.speed_scale <= 1.0
        if s < p.cfg.alert_thr:
            assert not a.active and a.speed_scale == 1.0
        if s >= p.cfg.alert_thr:
            assert a.active and a.speed_scale < 1.0
    # 0.85,0.9 연속(한 번만), 0.3로 재무장 후 0.85에서 다시 → 총 2회
    assert relocs == 2, f"reloc 트리거 {relocs}회 (기대 2)"
    print(f"[selftest] PASS — 경보 감속·relocalization 히스테리시스 정상 (reloc {relocs}회)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("ROS2 노드는 ros2_mitigation_node.py(스켈레톤) 참조. 정책 테스트: --selftest")
