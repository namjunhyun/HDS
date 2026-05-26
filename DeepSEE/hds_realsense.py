"""
HDS RealSense D455 실시간 추론
- IMU (accel+gyro) → DeepSEE neural prediction
- RGB 이미지 품질 → Hardware guardrail G
- 사용자 텍스트 입력 → LLM symbolic delta (별도 스레드)
- r_tilde = r_hat + delta_symbolic + W_HW * G
"""

import sys, time, collections, threading, queue, json, re, os
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyrealsense2 as rs
from dotenv import load_dotenv
import anthropic

load_dotenv("/home/junhyun/SEESys/DeepSEE/Training/.env")

sys.path.insert(0, "/home/junhyun/SEESys/DeepSEE/Training")
from models.DeepSEEModels import DeepSEEModel, MultiModalCrossAttentionConfig
from transformers import PatchTSMixerConfig, TimesformerConfig

# ── 설정 ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "/home/junhyun/SEESys/DeepSEE/Training/runs/Apr04_00-06-40_AHRI-Junhyun/SupervisedFinetune_3_best_model.pth"
THRESHOLD   = 0.6
IMU_WINDOW  = 30        # DeepSEE IMU 입력 길이
STREAM_HZ   = 10        # 추론 Hz (IMU 버퍼가 30개 쌓일 때마다)
PLOT_WINDOW = 200       # 라이브 플롯 표시 프레임 수
W_HW              = 0.3    # guardrail 가중치
SYMBOLIC_DURATION = 80     # symbolic delta 유지 프레임 수 (STREAM_HZ 기준 ~8초)

# Spearman 기반 이미지 품질 가중치
_W_E, _W_B, _W_L = 0.335, 0.305, 0.262
_WT = _W_E + _W_B + _W_L
W_ENTROPY = _W_E / _WT
W_BRIGHT  = _W_B / _WT
W_LAP     = _W_L / _WT

device = torch.device('cpu')

# ── LLM Symbolic Layer ────────────────────────────────────────────────────
class SymbolicLayer:
    """
    사용자 텍스트 입력을 받아 LLM으로 risk_delta 계산.
    입력 스레드 + LLM 호출 스레드가 비동기로 동작해 추론 루프를 블락하지 않음.
    """
    def __init__(self):
        self._input_q  = queue.Queue()   # 사용자 입력 대기열
        self._active   = []              # [(risk_delta, remaining_frames, reason), ...]
        self._lock     = threading.Lock()
        self._client   = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._running  = True

        # 입력 스레드
        t_in = threading.Thread(target=self._input_thread, daemon=True)
        t_in.start()
        # LLM 호출 스레드
        t_llm = threading.Thread(target=self._llm_thread, daemon=True)
        t_llm.start()

    def _input_thread(self):
        print("\n[Symbolic] 위험 구간 텍스트를 입력하세요 (엔터 전송, 빈 줄 무시):")
        print("  예: '어두운 복도로 진입 예정' / 'narrow corridor ahead with low lighting'\n")
        while self._running:
            try:
                text = input()
                if text.strip():
                    self._input_q.put(text.strip())
                    print(f"  [입력 수신] LLM 호출 중...", flush=True)
            except EOFError:
                break

    def _llm_thread(self):
        while self._running:
            try:
                text = self._input_q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                prompt = f"""You are a SLAM drift risk expert. A human operator says:
"{text}"

Decide how much to increase drift risk score (0~1 scale) for the upcoming zone.
Return ONLY JSON: {{"risk_delta": <float 0.0-0.5>, "reason": "<one sentence>"}}

Guidelines:
- Minor degradation: 0.05~0.15
- Moderate (low light or fast motion): 0.15~0.30
- Severe (dark + no features + rapid motion): 0.30~0.50"""

                msg    = self._client.messages.create(
                    model="claude-sonnet-4-6", max_tokens=128,
                    messages=[{"role": "user", "content": prompt}]
                )
                raw    = re.sub(r'```[a-z]*\n?', '', msg.content[0].text.strip()).strip().rstrip('`')
                result = json.loads(raw)
                delta  = float(result['risk_delta'])
                reason = result['reason']

                with self._lock:
                    self._active.append([delta, SYMBOLIC_DURATION, reason])

                print(f"\n  [LLM] risk_delta={delta:.3f} | {reason}")
                print(f"  → {SYMBOLIC_DURATION}프레임({SYMBOLIC_DURATION/STREAM_HZ:.0f}초) 동안 적용\n", flush=True)

            except Exception as e:
                print(f"\n  [LLM 오류] {e}\n", flush=True)

    def get_delta(self, deepsee_score):
        """현재 활성 symbolic delta 합산 (gap-based boost)"""
        with self._lock:
            total = 0.0
            expired = []
            for i, (risk_delta, remaining, _) in enumerate(self._active):
                if remaining <= 0:
                    expired.append(i)
                    continue
                gap     = max(0.0, 0.65 - deepsee_score)
                total  += min(gap, risk_delta)
                self._active[i][1] -= 1
            for i in reversed(expired):
                self._active.pop(i)
        return float(total)

    def is_active(self):
        with self._lock:
            return any(r > 0 for _, r, _ in self._active)

    def stop(self):
        self._running = False

# ── 1. 모델 로드 ───────────────────────────────────────────────────────────
def load_model():
    pd_config = TimesformerConfig(
        image_size=128, patch_size=8, num_channels=3,
        num_frames=4, num_hidden_layers=3, hidden_size=192, intermediate_size=256
    )
    ts_config = PatchTSMixerConfig(
        context_length=30, patch_len=5, num_input_channels=6, d_model=64
    )
    ca_config = MultiModalCrossAttentionConfig(
        ca_d_model=128, reg_d_fc=128, ts_num_input_channels=6,
        ts_d_model=64, pd_width=96, pd_height=128, pd_d_model=192,
        ts_context_length=30
    )
    ca_config.pe_max_len = 10000

    model = DeepSEEModel(pd_config, ts_config, ca_config)
    model.ca_regressor.ts_proj.projection = nn.Linear(64, 128)
    import __main__
    if not hasattr(__main__, 'loss_fn'):
        __main__.loss_fn = lambda pred, target: pred  # unpickling 우회용 dummy
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# ── 2. IMU 추론 ────────────────────────────────────────────────────────────
def infer_deepsee(model, imu_window):
    """imu_window: (30, 6) numpy array [gx,gy,gz,ax,ay,az]"""
    imu_norm = (imu_window - imu_window.mean(0)) / (imu_window.std(0) + 1e-8)
    x = torch.tensor(imu_norm[None], dtype=torch.float32)  # (1, 30, 6)
    with torch.no_grad():
        ts_out  = model.ts_encoder(x).last_hidden_state
        ts_hs   = ts_out.mean(dim=1)
        ts_feat = model.ca_regressor.ts_proj.projection(ts_hs).view(1, -1)[:, :128]
        combined = torch.cat([ts_feat, torch.zeros(1, 4224)], dim=1)
        out = model.ca_regressor.fc1(combined)
        out = model.ca_regressor.fc2(out)
    return float(out.item())

# ── 3. 이미지 품질 guardrail ───────────────────────────────────────────────
def image_quality_guardrail(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    brightness = gray.mean() / 255.0
    # 엔트로피 (히스토그램 기반)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist + 1e-10))) / 8.0  # 0~1 정규화
    # Laplacian (blur 감지)
    lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
    lap_norm = min(lap / 1000.0, 1.0)

    # 낮은 밝기/엔트로피, 높은 Laplacian → 위험 증가
    G = W_ENTROPY * (1 - entropy) + W_BRIGHT * (1 - brightness) + W_LAP * lap_norm
    return float(G)

# ── 4. RealSense 파이프라인 ────────────────────────────────────────────────
class RealSenseHDS:
    def __init__(self, model, symbolic):
        self.model    = model
        self.symbolic = symbolic
        self.imu_buf  = collections.deque(maxlen=IMU_WINDOW)
        self._last_accel = np.zeros(3)
        self._lock    = threading.Lock()

        self.history_deepsee  = []
        self.history_hds      = []
        self.history_G        = []
        self.history_symbolic = []
        self.history_alerts   = []

        self._raw_buf = collections.deque(maxlen=200)

    def _norm_running(self, val):
        self._raw_buf.append(val)
        arr = np.array(self._raw_buf)
        mn, mx = arr.min(), arr.max()
        rng = mx - mn
        # 변화량이 작으면 저위험으로 유지 (노이즈 증폭 방지)
        MIN_RANGE = 0.03
        if rng < MIN_RANGE:
            return float(np.clip((val - mn) / MIN_RANGE, 0, 0.35))
        return float(np.clip((val - mn) / rng, 0, 1))

    def run(self):
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
        cfg.enable_stream(rs.stream.gyro,  rs.format.motion_xyz32f, 200)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        print("[RealSense] 파이프라인 시작...")
        profile = pipeline.start(cfg)
        print("[RealSense] 연결됨. IMU 워밍업 중 (2초)...")
        time.sleep(2.0)

        # 라이브 플롯 초기화
        plt.ion()
        fig = plt.figure(figsize=(14, 6))
        fig.suptitle("HDS 실시간 — RealSense D455", fontsize=13, fontweight='bold')
        gs = gridspec.GridSpec(2, 1, hspace=0.4)

        ax1 = fig.add_subplot(gs[0])
        ax1.set_ylim(-0.05, 1.15)
        ax1.set_ylabel("Drift Risk Score")
        ax1.axhline(THRESHOLD, color='gray', ls='--', lw=1.2, label=f'Threshold {THRESHOLD}')
        line_ds,  = ax1.plot([], [], color='#3498db', lw=1.5, label='DeepSEE')
        line_hds, = ax1.plot([], [], color='#e74c3c', lw=2.0, label='HDS (Ours)')
        ax1.legend(loc='upper left', fontsize=9)
        status_txt = ax1.text(0.99, 1.08, "", transform=ax1.transAxes,
                              ha='right', fontsize=10, fontweight='bold', color='green')
        sym_txt = ax1.text(0.01, 1.08, "", transform=ax1.transAxes,
                           ha='left', fontsize=9, color='#e67e22')

        ax2 = fig.add_subplot(gs[1])
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("Guardrail / Symbolic Delta")
        line_G,   = ax2.plot([], [], color='#f39c12', lw=1.5, label='G (image quality)')
        line_sym, = ax2.plot([], [], color='#9b59b6', lw=1.5, ls='--', label='Symbolic delta')
        ax2.legend(loc='upper left', fontsize=9)

        print("[HDS] 추론 시작. Ctrl+C 로 종료.\n")
        frame_bgr   = None
        last_infer  = time.time()
        infer_count = 0

        try:
            while True:
                frames = pipeline.wait_for_frames(timeout_ms=1000)

                # IMU 수집
                for frame in frames:
                    profile_stream = frame.get_profile().stream_type()
                    if profile_stream == rs.stream.accel:
                        a = frame.as_motion_frame().get_motion_data()
                        with self._lock:
                            self._last_accel = np.array([a.x, a.y, a.z])
                    elif profile_stream == rs.stream.gyro:
                        g = frame.as_motion_frame().get_motion_data()
                        with self._lock:
                            imu_sample = np.array([g.x, g.y, g.z,
                                                   self._last_accel[0],
                                                   self._last_accel[1],
                                                   self._last_accel[2]])
                            self.imu_buf.append(imu_sample)

                # 컬러 프레임 수집
                color_frame = frames.get_color_frame()
                if color_frame:
                    frame_bgr = np.asanyarray(color_frame.get_data())

                # 추론 주기 제어
                now = time.time()
                if now - last_infer < 1.0 / STREAM_HZ:
                    continue
                last_infer = now

                with self._lock:
                    buf_len = len(self.imu_buf)
                    imu_win = np.array(self.imu_buf) if buf_len == IMU_WINDOW else None

                if imu_win is None:
                    print(f"\r  IMU 버퍼 채우는 중... {buf_len}/{IMU_WINDOW}", end='')
                    continue

                # DeepSEE 추론
                raw_pred = infer_deepsee(self.model, imu_win)
                ds_score = self._norm_running(raw_pred)

                # Guardrail
                G = image_quality_guardrail(frame_bgr) if frame_bgr is not None else 0.0

                # Symbolic delta (LLM, gap-based)
                delta_sym = self.symbolic.get_delta(ds_score)

                # HDS 점수
                hds_score = float(np.clip(ds_score + delta_sym + W_HW * G, 0, 1))

                self.history_deepsee.append(ds_score)
                self.history_hds.append(hds_score)
                self.history_G.append(G)
                self.history_symbolic.append(delta_sym)

                alert = hds_score >= THRESHOLD
                self.history_alerts.append(alert)
                infer_count += 1

                # 터미널 출력
                bar = '#' * int(hds_score * 20)
                tag = "*** ALERT ***" if alert else "Normal       "
                sym_tag = f" SYM+{delta_sym:.2f}" if delta_sym > 0 else ""
                print(f"\r  [{tag}] DS={ds_score:.3f} G={G:.3f}{sym_tag} HDS={hds_score:.3f}  [{bar:<20}]", end='')

                # 라이브 플롯 업데이트
                n   = len(self.history_hds)
                win = min(n, PLOT_WINDOW)
                xs  = np.arange(win)

                line_ds.set_data(xs,  self.history_deepsee[-win:])
                line_hds.set_data(xs, self.history_hds[-win:])
                line_G.set_data(xs,   self.history_G[-win:])
                line_sym.set_data(xs, self.history_symbolic[-win:])

                for ax in [ax1, ax2]:
                    ax.set_xlim(0, max(PLOT_WINDOW, win))

                if alert:
                    status_txt.set_text(f"DRIFT RISK  {hds_score:.3f}")
                    status_txt.set_color('red')
                else:
                    status_txt.set_text(f"Normal  {hds_score:.3f}")
                    status_txt.set_color('green')

                if self.symbolic.is_active():
                    sym_txt.set_text(f"[Symbolic ON  +{delta_sym:.3f}]")
                else:
                    sym_txt.set_text("")

                fig.canvas.draw()
                fig.canvas.flush_events()

                # 카메라 화면 (별도 창)
                if frame_bgr is not None:
                    disp = frame_bgr.copy()
                    color = (0, 0, 255) if alert else (0, 200, 0)
                    cv2.putText(disp, f"HDS: {hds_score:.3f}", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    cv2.putText(disp, "DRIFT RISK!" if alert else "Normal",
                                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    if self.symbolic.is_active():
                        cv2.putText(disp, f"[Symbolic +{delta_sym:.2f}]",
                                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 80, 255), 2)
                    cv2.imshow("RealSense D455", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            print("\n\n[HDS] 종료 요청.")
        finally:
            self.symbolic.stop()
            pipeline.stop()
            cv2.destroyAllWindows()
            print(f"[HDS] 총 {infer_count}회 추론 완료.")
            plt.ioff()
            plt.show()

# ── 메인 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== HDS RealSense D455 실시간 추론 ===\n")
    print("[1/3] DeepSEE 모델 로드...")
    model = load_model()
    print("  OK\n")

    print("[2/3] Symbolic Layer 초기화...")
    symbolic = SymbolicLayer()

    print("[3/3] RealSense 스트리밍 시작...")
    hds = RealSenseHDS(model, symbolic)
    hds.run()
