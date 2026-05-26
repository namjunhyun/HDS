"""
HDS 실시간 테스트 스크립트
- 사전 계산된 DeepSEE 예측값을 프레임 단위로 스트리밍
- LLM symbolic delta는 시작 전 1회만 호출
- matplotlib 라이브 플롯 + 터미널 알람
"""

import os, sys, time, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dotenv import load_dotenv
import anthropic
import datetime

sys.path.insert(0, "/home/junhyun/SEESys/DeepSEE/Training")
load_dotenv("/home/junhyun/SEESys/DeepSEE/Training/.env")

# ── 설정 ──────────────────────────────────────────────────────────────────
SEQ         = "MH_04_difficult"   # 테스트할 시퀀스 (MH_01~05)
THRESHOLD   = 0.6                 # 경보 임계값
STREAM_HZ   = 30                  # 스트리밍 속도 (프레임/초), 높을수록 빠름
WINDOW      = 150                 # 라이브 플롯에 표시할 최근 프레임 수

RUN_DIR = "/home/junhyun/SEESys/DeepSEE/Training/runs/Apr04_00-06-40_AHRI-Junhyun"
CSV_DIR = "/home/junhyun/SEESys/DeepSEE/data"

SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]

CAM_FRAME_COUNT = {
    "MH_01_easy":      2912,
    "MH_02_easy":      3014,
    "MH_03_medium":    2700,
    "MH_04_difficult": 2033,
    "MH_05_difficult": 2273,
}
CAM_HZ = 20.0

W_SY = 0.3
W_HW = 0.0
_W_ENTROPY, _W_BRIGHT, _W_LAP = 0.335, 0.305, 0.262
_W_TOTAL = _W_ENTROPY + _W_BRIGHT + _W_LAP
W_ENTROPY = _W_ENTROPY / _W_TOTAL
W_BRIGHT  = _W_BRIGHT  / _W_TOTAL
W_LAP     = _W_LAP     / _W_TOTAL

HUMAN_ANNOTATIONS = {
    "MH_01_easy": [
        (2219, 2357, "Path plan indicates a dark machinery room with dense metallic pipes ahead. Low illumination combined with specular reflections will significantly degrade visual feature tracking."),
    ],
    "MH_03_medium": [
        (47,  176,  "Path plan enters a poorly lit industrial zone with only lateral lighting. Reduced illumination and dark shadows will significantly degrade feature detection."),
        (500,  870,  "Path plan shows a rapid maneuver zone ahead. Expect heavy camera shake and feature tracking instability."),
        (1150, 1320, "Upcoming low-light corridor in path plan. Poor illumination will reduce visual features."),
        (2070, 2240, "Path plan indicates unstable motion zone ahead. Camera blur expected."),
    ],
    "MH_04_difficult": [
        (77,   231,  "Path plan shows downward-looking viewpoint over dark industrial machinery at mission start. Low contrast scene with limited visual features anticipated."),
        (990,  1443, "Path plan enters extremely dark zone ahead. Near-zero ambient lighting — almost silhouette-only visibility. Severe drift risk."),
        (1358, 1589, "Path plan navigates through dense metallic tank and pipe cluster. Highly reflective surfaces and complex geometry cause severe feature confusion and mismatching."),
        (1630, 1826, "Path plan shows another dark low-visibility zone. Poor ambient lighting will cause repeated visual feature degradation."),
    ],
    "MH_05_difficult": [
        (0,    420,  "Unstable initial flight phase in path plan. High vibration expected."),
        (1070, 1530, "Dark low-feature zone detected ahead in path plan. Severe drift risk anticipated."),
        (1920, 2060, "Sudden trajectory change in path plan. Rapid motion change expected."),
    ],
}

# ── 1. DeepSEE 예측 로드 ──────────────────────────────────────────────────
def load_predictions(seq_name):
    fold_idx = SEQS.index(seq_name)
    gt   = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_gt.npy",  allow_pickle=True)
    est  = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_est.npy", allow_pickle=True)
    base = np.load(f"{RUN_DIR}/SupervisedFinetune_{fold_idx}_Y_base.npy", allow_pickle=True)

    gt_flat   = np.concatenate([np.array(x).flatten() for x in gt])
    est_flat  = np.concatenate([np.array(x).flatten() for x in est])
    base_flat = np.array(base).flatten()

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    return norm(gt_flat), norm(est_flat), norm(base_flat)

# ── 2. Image Quality Guardrail ─────────────────────────────────────────────
def compute_guardrail(seq_name, n_samples):
    csv_path = f"{CSV_DIR}/data_EuRoC_{seq_name}_0.csv"
    if not os.path.exists(csv_path):
        print(f"  [WARN] CSV not found: {csv_path}, guardrail=0")
        return np.zeros(n_samples)

    df = pd.read_csv(csv_path).dropna(subset=['Brightness', 'Entropy', 'Laplacian'])
    bright  = df['Brightness'].values
    entropy = df['Entropy'].values
    lap     = df['Laplacian'].values
    t       = np.arange(len(bright))
    t_new   = np.linspace(0, len(bright) - 1, n_samples)

    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)

    G = (W_ENTROPY * (1 - norm(np.interp(t_new, t, entropy))) +
         W_BRIGHT  * (1 - norm(np.interp(t_new, t, bright)))  +
         W_LAP     *      norm(np.interp(t_new, t, lap)))
    return G

# ── 3. LLM 호출 (시작 전 1회) ─────────────────────────────────────────────
def get_symbolic_deltas(seq_name, deepsee_pred):
    annotations = HUMAN_ANNOTATIONS.get(seq_name, [])
    if not annotations:
        print("  [INFO] 이 시퀀스에 경로 계획 어노테이션 없음 → symbolic delta=0")
        return np.zeros(len(deepsee_pred)), []

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    n_samples = len(deepsee_pred)
    n_cam     = CAM_FRAME_COUNT[seq_name]
    delta_symbolic = np.zeros(n_samples)
    segment_info   = []

    print(f"\n[LLM] {len(annotations)}개 경로 계획 어노테이션 처리 중...")
    for cam_start, cam_end, human_text in annotations:
        m_start = int(cam_start * n_samples / n_cam)
        m_end   = min(int(cam_end   * n_samples / n_cam) + 1, n_samples)

        prompt = f"""You are a SLAM drift risk expert. A human operator says:
"{human_text}"
Sequence: {seq_name}

Return ONLY JSON:
{{"risk_delta": <float 0.0-0.5>, "reason": "<one sentence>"}}

Guidelines:
- Minor: 0.05~0.15
- Moderate (low light or fast motion): 0.15~0.30
- Severe (dark + no features + rapid motion): 0.30~0.50"""

        msg    = client.messages.create(model="claude-sonnet-4-6", max_tokens=128,
                                        messages=[{"role": "user", "content": prompt}])
        raw    = re.sub(r'```[a-z]*\n?', '', msg.content[0].text.strip()).strip().rstrip('`')
        result = json.loads(raw)
        risk_delta = float(result['risk_delta'])

        gap     = np.maximum(0.0, 0.65 - deepsee_pred)
        applied = np.zeros(n_samples)
        applied[m_start:m_end] = np.minimum(gap[m_start:m_end], risk_delta)
        delta_symbolic += applied

        print(f"  frames {m_start:4d}~{m_end:4d} | delta={risk_delta:.3f} | {result['reason'][:70]}")
        segment_info.append((m_start, m_end, human_text))

    return delta_symbolic, segment_info

# ── 4. 실시간 스트리밍 ─────────────────────────────────────────────────────
def stream(seq_name, gt, deepsee, hds, segment_info):
    n = len(gt)
    frame_delay = 1.0 / STREAM_HZ

    plt.ion()
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f"HDS 실시간 테스트 — {seq_name}", fontsize=13, fontweight='bold')

    ax = fig.add_subplot(111)
    ax.set_xlim(0, WINDOW)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Drift Risk Score")
    ax.set_xlabel("Recent Frames")
    ax.axhline(THRESHOLD, color='gray', ls='--', lw=1.2, alpha=0.7, label=f"Threshold {THRESHOLD}")

    # 배경 경보 패치 (처음엔 비어 있음)
    alert_patch = ax.axvspan(0, 0, color='red', alpha=0.0)

    line_gt,      = ax.plot([], [], color='black',  alpha=0.3, lw=1.2, label='GT')
    line_deepsee, = ax.plot([], [], color='#3498db', alpha=0.8, lw=1.5, label='DeepSEE')
    line_hds,     = ax.plot([], [], color='#e74c3c', lw=2.0,   label='HDS (Ours)')

    # 심볼릭 구간 표시
    for m_start, m_end, _ in segment_info:
        ax.axvspan(max(0, m_start - n), max(0, m_end - n),  # 상대 위치로 고정 마킹
                   color='#f1c40f', alpha=0.07)

    ax.legend(loc='upper left', fontsize=9)
    status_text = ax.text(0.99, 1.06, "", transform=ax.transAxes,
                          ha='right', va='top', fontsize=10, fontweight='bold', color='green')

    print(f"\n[STREAM] {n} 프레임 스트리밍 시작 @ {STREAM_HZ} fps ({n/STREAM_HZ:.1f}초)")
    print(f"  Ctrl+C 로 중단\n")

    alert_active = False
    alert_start  = None

    for i in range(n):
        t0 = time.perf_counter()

        # 롤링 윈도우 슬라이스
        win_s = max(0, i + 1 - WINDOW)
        win_e = i + 1
        xs    = np.arange(win_e - win_s)

        line_gt.set_data(xs,      gt[win_s:win_e])
        line_deepsee.set_data(xs, deepsee[win_s:win_e])
        line_hds.set_data(xs,     hds[win_s:win_e])
        ax.set_xlim(0, max(WINDOW, win_e - win_s))

        # 심볼릭 구간 하이라이트 (현재 창에 맞게)
        # (구간이 현재 롤링 윈도우에 들어올 때만 표시)
        for collection in ax.collections:
            if collection.get_label() == '_sym_zone':
                collection.remove()
        for m_start, m_end, _ in segment_info:
            if m_end > win_s and m_start < win_e:
                rel_s = max(m_start - win_s, 0)
                rel_e = min(m_end   - win_s, win_e - win_s)
                zone = ax.axvspan(rel_s, rel_e, color='#f1c40f', alpha=0.15, label='_sym_zone')

        # 경보 상태
        cur_hds = hds[i]
        if cur_hds >= THRESHOLD and not alert_active:
            alert_active = True
            alert_start  = i
            elapsed_sec  = i * (CAM_FRAME_COUNT[seq_name] / CAM_HZ) / n
            print(f"  *** ALERT *** frame {i:4d} | t={elapsed_sec:.1f}s | HDS={cur_hds:.3f}")

        elif cur_hds < THRESHOLD and alert_active:
            alert_active = False

        if alert_active:
            status_text.set_text(f"DRIFT RISK  HDS={cur_hds:.3f}")
            status_text.set_color('red')
        else:
            status_text.set_text(f"Normal  HDS={cur_hds:.3f}")
            status_text.set_color('green')

        fig.canvas.draw()
        fig.canvas.flush_events()

        elapsed = time.perf_counter() - t0
        remaining = frame_delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

    print("\n[DONE] 스트리밍 완료. 창을 닫으면 종료됩니다.")
    plt.ioff()
    plt.show()

# ── 메인 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",    default=SEQ,        choices=SEQS)
    parser.add_argument("--hz",     default=STREAM_HZ,  type=float, help="스트리밍 속도 (fps)")
    parser.add_argument("--window", default=WINDOW,     type=int,   help="라이브 플롯 윈도우 크기")
    args = parser.parse_args()

    STREAM_HZ = args.hz
    WINDOW    = args.window
    seq_name  = args.seq

    print(f"=== HDS 실시간 테스트: {seq_name} ===")
    print(f"  스트리밍: {STREAM_HZ} fps | 윈도우: {WINDOW} 프레임\n")

    print("[1/3] 예측값 로드...")
    gt, deepsee, baseline = load_predictions(seq_name)
    n = len(gt)
    print(f"  총 {n} 모델 프레임")

    print("[2/3] 이미지 품질 guardrail 계산...")
    G = compute_guardrail(seq_name, n)

    print("[2/3] LLM symbolic delta 계산...")
    delta_symbolic, segment_info = get_symbolic_deltas(seq_name, deepsee)

    print("[3/3] HDS 점수 생성...")
    hds = np.clip(deepsee + delta_symbolic + W_HW * G, 0, 1)

    print(f"\n  DeepSEE 평균: {deepsee.mean():.3f} | HDS 평균: {hds.mean():.3f}")
    print(f"  HDS 경보 프레임: {(hds >= THRESHOLD).sum()} / {n}\n")

    stream(seq_name, gt, deepsee, hds, segment_info)
