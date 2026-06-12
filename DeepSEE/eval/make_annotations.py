"""
EuRoC용 운영자 어노테이션 *새로* 생성 (오라클 배치 X).

원칙 — 라벨 누수 금지:
  · 에러 GT(RelativeError)는 절대 참조하지 않는다.
  · 오직 '운영자가 경로 미리보기에서 관측 가능한' 단서로만 위험 구간을 잡는다:
      - 어두움(Brightness↓), 텍스처/특징 부족(MatchedInlier·Entropy↓), 급회전(|Yaw/Pitch/Roll|↑)
  · 어노테이션은 구간 시작보다 LEAD 세그먼트 '앞서' 배치 → 경로 사전지식(foreknowledge) 모사.
    (현재 프레임엔 아직 안 보이는 다가올 조건 → 이것이 G(반응형)와 차별되는 symbolic의 가치)

출력: annotations_euroc.json
  { seq: [ {seg_start, seg_end, cue, severity, text}, ... ] }  (seg = est 인덱스 좌표)

사용: python3 make_annotations.py
"""
import os, json
import numpy as np
import pandas as pd

EA_DIR  = "/home/junhyun/euroc_error_analysis/data"
OUT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotations_euroc.json")

# fold 인덱스 ↔ 시퀀스 (npy EuRoC_ZeroShot_{i} 순서와 동일)
SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
N_EST = {"MH_01_easy": 114, "MH_02_easy": 92, "MH_03_medium": 80,
         "MH_04_difficult": 58, "MH_05_difficult": 65}

LEAD       = 3      # 구간 시작보다 몇 세그먼트 앞서 경고 (≈ 5초 사전지식)
MIN_ZONE   = 2      # 최소 지속 세그먼트
LOW_PCT    = 25     # 어두움/특징부족 판정 퍼센타일


def _seg_aggregate(df, n_est):
    """프레임 행 → n_est개 세그먼트 평균으로 집계."""
    idx = np.floor(np.arange(len(df)) * n_est / len(df)).astype(int)
    idx = np.clip(idx, 0, n_est - 1)
    out = {}
    for col in ["Brightness", "Entropy", "MatchedInlier", "Yaw", "Pitch", "Roll"]:
        v = df[col].to_numpy(dtype=float) if col in df else np.zeros(len(df))
        seg = np.array([v[idx == k].mean() if np.any(idx == k) else np.nan
                        for k in range(n_est)])
        out[col] = pd.Series(seg).interpolate().bfill().ffill().to_numpy()
    return out


def _zones(mask):
    """True 구간들을 [start, end] 리스트로 (gap<=1 병합)."""
    zones, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            zones.append([s, i - 1]); s = None
    if s is not None:
        zones.append([s, len(mask) - 1])
    # gap 1 병합
    merged = []
    for z in zones:
        if merged and z[0] - merged[-1][1] <= 1:
            merged[-1][1] = z[1]
        else:
            merged.append(z)
    return [z for z in merged if z[1] - z[0] + 1 >= MIN_ZONE]


def _severity(val, lo, hi):
    """관측 단서 강도 → mild/moderate/severe (에러와 무관)."""
    if hi <= lo:
        return "moderate"
    z = (val - lo) / (hi - lo)
    return "severe" if z > 0.66 else "moderate" if z > 0.33 else "mild"


def _text(cue, sev):
    adj = {"mild": "다소", "moderate": "상당히", "severe": "매우"}[sev]
    if cue == "dark":
        return f"경로 계획상 전방에 조명이 {adj} 부족한 구간이 있습니다. 낮은 조도로 시각 특징 추적이 저하될 것으로 예상됩니다."
    if cue == "low_texture":
        return f"전방은 텍스처가 {adj} 부족한 구간입니다. 특징점 매칭이 불안정해질 우려가 있습니다."
    if cue == "turn":
        return f"전방에서 {adj} 급격한 방향 전환이 예정되어 있습니다. 카메라 모션 블러와 추적 불안정이 예상됩니다."
    return "전방 구간에서 시각 추적 저하가 예상됩니다."


def main():
    result = {}
    for seq in SEQS:
        fp = os.path.join(EA_DIR, seq, "features.csv")
        if not os.path.exists(fp):
            print(f"[skip] {seq}: features.csv 없음"); continue
        df  = pd.read_csv(fp)
        n   = N_EST[seq]
        agg = _seg_aggregate(df, n)

        bright = agg["Brightness"]
        inlier = agg["MatchedInlier"]
        turn   = np.abs(agg["Yaw"]) + np.abs(agg["Pitch"]) + np.abs(agg["Roll"])

        b_thr = np.percentile(bright, LOW_PCT)
        i_thr = np.percentile(inlier, LOW_PCT)
        t_thr = np.percentile(turn, 100 - LOW_PCT)

        # 단서별 zone 탐지
        cue_masks = {
            "dark":        bright < b_thr,
            "low_texture": inlier < i_thr,
            "turn":        turn   > t_thr,
        }
        annots = []
        for cue, mask in cue_masks.items():
            for z0, z1 in _zones(mask):
                seg_start = max(0, z0 - LEAD)        # 앞서 경고 (foreknowledge)
                if cue == "dark":
                    sev = _severity(b_thr - bright[z0:z1 + 1].min(), 0, b_thr)
                elif cue == "low_texture":
                    sev = _severity(i_thr - inlier[z0:z1 + 1].min(), 0, i_thr)
                else:
                    sev = _severity(turn[z0:z1 + 1].max() - t_thr, 0, t_thr)
                annots.append({
                    "seg_start": int(seg_start), "seg_end": int(z1),
                    "cue": cue, "severity": sev, "text": _text(cue, sev),
                })
        annots.sort(key=lambda a: a["seg_start"])
        result[seq] = annots
        print(f"{seq}: {len(annots)}개 어노테이션 "
              f"(dark/low_texture/turn = "
              f"{sum(a['cue']=='dark' for a in annots)}/"
              f"{sum(a['cue']=='low_texture' for a in annots)}/"
              f"{sum(a['cue']=='turn' for a in annots)})")

    with open(OUT, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
