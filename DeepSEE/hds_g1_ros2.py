"""
HDS G1 ROS2 실시간 추론
- G1 D435i IMU (/camera/imu) → DeepSEE 오차 예측
- RGB 이미지 품질 (/local/color/image_raw) → Hardware guardrail G
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
from dotenv import load_dotenv
import anthropic

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge

load_dotenv("/home/junhyun/SEESys/DeepSEE/Training/.env")

sys.path.insert(0, "/home/junhyun/SEESys/DeepSEE/Training")
from models.DeepSEEModels import DeepSEEModel, MultiModalCrossAttentionConfig
from transformers import PatchTSMixerConfig, TimesformerConfig

# ── 설정 ──────────────────────────────────────────────────────────────────
MODEL_PATH      = "/home/junhyun/SEESys/DeepSEE/Training/runs/May13_10-25-55_AHRI-Junhyun/SupervisedFinetune_1_best_model.pth"
THRESHOLD       = 0.6
IMU_WINDOW      = 30
STREAM_HZ       = 10
PLOT_WINDOW     = 200
W_HW            = 0.3
SYMBOLIC_DURATION = 80

_W_E, _W_B, _W_L = 0.335, 0.305, 0.262
_WT = _W_E + _W_B + _W_L
W_ENTROPY = _W_E / _WT
W_BRIGHT  = _W_B / _WT
W_LAP     = _W_L / _WT

device = torch.device('cpu')

# ── LLM Symbolic Layer ────────────────────────────────────────────────────
class SymbolicLayer:
    def __init__(self):
        self._input_q = queue.Queue()
        self._active  = []
        self._lock    = threading.Lock()
        self._client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._running = True

        threading.Thread(target=self._input_thread, daemon=True).start()
        threading.Thread(target=self._llm_thread,   daemon=True).start()

    def _input_thread(self):
        print("\n[Symbolic] 위험 구간 텍스트 입력 (엔터 전송):")
        print("  예: '어두운 복도로 진입 예정' / 'narrow corridor ahead'\n")
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
        with self._lock:
            total, expired = 0.0, []
            for i, (risk_delta, remaining, _) in enumerate(self._active):
                if remaining <= 0:
                    expired.append(i)
                    continue
                gap    = max(0.0, 0.65 - deepsee_score)
                total += min(gap, risk_delta)
                self._active[i][1] -= 1
            for i in reversed(expired):
                self._active.pop(i)
        return float(total)

    def is_active(self):
        with self._lock:
            return any(r > 0 for _, r, _ in self._active)

    def stop(self):
        self._running = False

# ── ROS2 노드 ─────────────────────────────────────────────────────────────
class G1SensorNode(Node):
    def __init__(self):
        super().__init__('hds_g1_node')
        self.bridge       = CvBridge()
        self.imu_buf      = collections.deque(maxlen=IMU_WINDOW)
        self.frame_bgr    = None
        self._last_accel  = np.zeros(3)
        self._lock        = threading.Lock()
        self.imu_count    = 0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(Imu, '/camera/imu',
                                 self._imu_cb, sensor_qos)
        self.create_subscription(Image, '/local/color/image_raw',
                                 self._color_cb, sensor_qos)
        self.get_logger().info("G1 센서 구독 시작: /camera/imu, /local/color/image_raw")

    def _imu_cb(self, msg: Imu):
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        gx = msg.angular_velocity.x
        gy = msg.angular_velocity.y
        gz = msg.angular_velocity.z
        with self._lock:
            self._last_accel = np.array([ax, ay, az])
            self.imu_buf.append(np.array([gx, gy, gz, ax, ay, az]))
            self.imu_count += 1

    def _color_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self._lock:
                self.frame_bgr = bgr
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}")

    def get_state(self):
        with self._lock:
            imu_win   = np.array(self.imu_buf) if len(self.imu_buf) == IMU_WINDOW else None
            frame_bgr = self.frame_bgr.copy() if self.frame_bgr is not None else None
            imu_count = self.imu_count
        return imu_win, frame_bgr, imu_count

# ── 모델 로드 ─────────────────────────────────────────────────────────────
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
        __main__.loss_fn = lambda pred, target: pred

    ckpt       = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# ── DeepSEE 추론 ─────────────────────────────────────────────────────────
def infer_deepsee(model, imu_window):
    imu_norm = (imu_window - imu_window.mean(0)) / (imu_window.std(0) + 1e-8)
    x = torch.tensor(imu_norm[None], dtype=torch.float32)
    with torch.no_grad():
        ts_out   = model.ts_encoder(x).last_hidden_state
        ts_hs    = ts_out.mean(dim=1)
        ts_feat  = model.ca_regressor.ts_proj.projection(ts_hs).view(1, -1)[:, :128]
        combined = torch.cat([ts_feat, torch.zeros(1, 4224)], dim=1)
        out      = model.ca_regressor.fc1(combined)
        out      = model.ca_regressor.fc2(out)
    return float(out.item())

# ── 이미지 품질 guardrail ─────────────────────────────────────────────────
def image_quality_guardrail(frame_bgr):
    gray       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    brightness = gray.mean() / 255.0
    hist, _    = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist       = hist[hist > 0]
    entropy    = float(-np.sum(hist * np.log2(hist + 1e-10))) / 8.0
    lap        = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
    lap_norm   = min(lap / 1000.0, 1.0)
    return float(W_ENTROPY * (1 - entropy) + W_BRIGHT * (1 - brightness) + W_LAP * lap_norm)

# ── 메인 루프 ─────────────────────────────────────────────────────────────
def run_hds(node: G1SensorNode, model, symbolic: SymbolicLayer):
    raw_buf = collections.deque(maxlen=200)

    def norm_running(val):
        raw_buf.append(val)
        arr = np.array(raw_buf)
        mn, mx = arr.min(), arr.max()
        rng = mx - mn
        if rng < 0.03:
            return float(np.clip((val - mn) / 0.03, 0, 0.35))
        return float(np.clip((val - mn) / rng, 0, 1))

    hist_ds, hist_hds, hist_G, hist_sym = [], [], [], []

    plt.ion()
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("HDS 실시간 — Unitree G1 D435i (ROS2)", fontsize=13, fontweight='bold')
    gs  = gridspec.GridSpec(2, 1, hspace=0.4)

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
    last_infer  = time.time()
    infer_count = 0

    try:
        while rclpy.ok():
            now = time.time()
            if now - last_infer < 1.0 / STREAM_HZ:
                time.sleep(0.005)
                fig.canvas.flush_events()
                continue
            last_infer = now

            imu_win, frame_bgr, imu_count = node.get_state()

            if imu_win is None:
                print(f"\r  IMU 버퍼 채우는 중... {imu_count}/{IMU_WINDOW}", end='')
                fig.canvas.flush_events()
                continue

            raw_pred = infer_deepsee(model, imu_win)
            ds_score = norm_running(raw_pred)

            G         = image_quality_guardrail(frame_bgr) if frame_bgr is not None else 0.0
            delta_sym = symbolic.get_delta(ds_score)
            hds_score = float(np.clip(ds_score + delta_sym + W_HW * G, 0, 1))

            hist_ds.append(ds_score)
            hist_hds.append(hds_score)
            hist_G.append(G)
            hist_sym.append(delta_sym)
            infer_count += 1
            alert = hds_score >= THRESHOLD

            bar = '#' * int(hds_score * 20)
            tag = "*** ALERT ***" if alert else "Normal       "
            sym_tag = f" SYM+{delta_sym:.2f}" if delta_sym > 0 else ""
            print(f"\r  [{tag}] DS={ds_score:.3f} G={G:.3f}{sym_tag} HDS={hds_score:.3f}  [{bar:<20}]", end='')

            n   = len(hist_hds)
            win = min(n, PLOT_WINDOW)
            xs  = np.arange(win)

            line_ds.set_data(xs,  hist_ds[-win:])
            line_hds.set_data(xs, hist_hds[-win:])
            line_G.set_data(xs,   hist_G[-win:])
            line_sym.set_data(xs, hist_sym[-win:])

            for ax in [ax1, ax2]:
                ax.set_xlim(0, max(PLOT_WINDOW, win))

            if alert:
                status_txt.set_text(f"DRIFT RISK  {hds_score:.3f}")
                status_txt.set_color('red')
            else:
                status_txt.set_text(f"Normal  {hds_score:.3f}")
                status_txt.set_color('green')

            sym_txt.set_text(f"[Symbolic ON  +{delta_sym:.3f}]" if symbolic.is_active() else "")

            fig.canvas.draw()
            fig.canvas.flush_events()

            if frame_bgr is not None:
                disp  = frame_bgr.copy()
                color = (0, 0, 255) if alert else (0, 200, 0)
                cv2.putText(disp, f"HDS: {hds_score:.3f}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.putText(disp, "DRIFT RISK!" if alert else "Normal",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                if symbolic.is_active():
                    cv2.putText(disp, f"[Symbolic +{delta_sym:.2f}]",
                                (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 80, 255), 2)
                cv2.imshow("G1 D435i", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n\n[HDS] 종료 요청.")
    finally:
        symbolic.stop()
        cv2.destroyAllWindows()
        print(f"[HDS] 총 {infer_count}회 추론 완료.")
        plt.ioff()
        plt.show()

# ── 메인 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== HDS G1 ROS2 실시간 추론 ===\n")
    print(f"모델: {MODEL_PATH}\n")

    print("[1/3] DeepSEE 모델 로드...")
    model = load_model()
    print("  OK\n")

    print("[2/3] Symbolic Layer 초기화...")
    symbolic = SymbolicLayer()
    print("  OK\n")

    print("[3/3] ROS2 초기화 및 G1 구독 시작...")
    rclpy.init()
    node = G1SensorNode()

    # ROS2 spin을 별도 스레드에서 실행
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    print("  OK\n")

    run_hds(node, model, symbolic)

    node.destroy_node()
    rclpy.shutdown()
