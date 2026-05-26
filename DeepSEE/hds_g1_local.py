"""
HDS G1 추론 (conda deepsee Python 3.9으로 실행)
/dev/shm에서 IMU + 이미지 읽어서 DeepSEE 추론
"""
import sys, time, collections, threading, queue, json, re, os
import numpy as np
import cv2
import torch
import torch.nn as nn
from dotenv import load_dotenv
import anthropic

_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"))

sys.path.insert(0, _HERE)
from models.DeepSEEModels import DeepSEEModel, MultiModalCrossAttentionConfig
from transformers import PatchTSMixerConfig, TimesformerConfig

# 모델 가중치 경로: runs/ 폴더에 직접 배치하거나 아래 경로 수정
MODEL_PATH      = os.path.join(_HERE, "runs", "SupervisedFinetune_1_best_model.pth")
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

# ── Symbolic Layer ────────────────────────────────────────────────────────
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
        print("  예: '어두운 복도로 진입 예정'\n")
        while self._running:
            try:
                text = input()
                if text.strip():
                    self._input_q.put(text.strip())
                    print("  [입력 수신] LLM 호출 중...", flush=True)
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
                print(f"  → {SYMBOLIC_DURATION/STREAM_HZ:.0f}초 동안 적용\n", flush=True)
            except Exception as e:
                print(f"\n  [LLM 오류] {e}\n", flush=True)

    def get_delta(self, ds_score):
        with self._lock:
            total, expired = 0.0, []
            for i, (risk_delta, remaining, _) in enumerate(self._active):
                if remaining <= 0:
                    expired.append(i)
                    continue
                gap    = max(0.0, 0.65 - ds_score)
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

# ── 모델 로드 ─────────────────────────────────────────────────────────────
def load_model():
    pd_config = TimesformerConfig(
        image_size=128, patch_size=8, num_channels=3,
        num_frames=4, num_hidden_layers=3, hidden_size=192, intermediate_size=256
    )
    # patch_len=1 → 30 patches (fc1이 (30+4)*128=4352 기대)
    ts_config = PatchTSMixerConfig(
        context_length=30, patch_len=1, num_input_channels=6, d_model=64
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

    # ts_proj.forward 패치: Linear(64,128)에 맞게 mean pooling 후 projection
    orig_ts_proj = model.ca_regressor.ts_proj
    def _ts_proj_forward(ts_hs):
        # ts_hs: (batch, channels, patches, d_model) = (1, 6, 30, 64)
        hs = ts_hs.mean(dim=1)          # (1, 30, 64)
        return orig_ts_proj.projection(hs)  # (1, 30, 128)
    model.ca_regressor.ts_proj.forward = _ts_proj_forward

    model.eval()
    return model

_orb = cv2.ORB_create(500)

def _read_psd():
    """C++에서 저장한 PSD 버퍼 읽기: (4,3,96,128) float32"""
    try:
        raw = np.fromfile('/dev/shm/psd_buffer.bin', dtype=np.float32)
        if raw.size == 4 * 3 * 96 * 128:
            return raw.reshape(4, 3, 96, 128)
    except Exception:
        pass
    return None

def infer_deepsee(model, imu_window, frame_bgr):
    # ── IMU (RTS 대용) ────────────────────────────────────────────────────
    imu_norm = (imu_window - imu_window.mean(0)) / (imu_window.std(0) + 1e-8)
    ts = torch.tensor(imu_norm[None], dtype=torch.float32)   # (1, 30, 6)

    # ── PSD ──────────────────────────────────────────────────────────────
    psd_np = _read_psd()
    if psd_np is None:
        # PSD 없으면 ORB 키포인트로 간이 대체
        small = cv2.resize(frame_bgr, (128, 96))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kps   = _orb.detect(gray, None)
        canvas = np.zeros((96, 128), dtype=np.float32)
        for kp in kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= x < 128 and 0 <= y < 96:
                canvas[y, x] = min(kp.response / 1000.0, 1.0)
        ch = canvas[None]  # (1, 96, 128)
        frame = np.concatenate([ch, ch, ch], axis=0)  # (3, 96, 128)
        psd_np = np.stack([frame] * 4, axis=0)        # (4, 3, 96, 128)

    # (4, 3, 96, 128) → (1, 3, 4, 96, 128)
    pd = torch.tensor(psd_np, dtype=torch.float32).permute(1, 0, 2, 3).unsqueeze(0)

    with torch.no_grad():
        out = model(pd, ts)
    return float(out.item())

def image_quality_guardrail(frame_bgr):
    gray       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    brightness = gray.mean() / 255.0
    hist, _    = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist       = hist[hist > 0]
    entropy    = float(-np.sum(hist * np.log2(hist + 1e-10))) / 8.0
    lap        = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
    lap_norm   = min(lap / 1000.0, 1.0)
    kps        = _orb.detect(gray.astype(np.uint8), None)
    feat_score = 1.0 - min(len(kps) / 300.0, 1.0)  # 특징점 적을수록 위험
    # 어두움·저엔트로피·블러·특징점 부족 → G 올라감
    return float(W_ENTROPY * (1 - entropy) + W_BRIGHT * (1 - brightness) + W_LAP * (1 - lap_norm) + 0.1 * feat_score)

def get_state():
    try:
        imu_win   = np.load('/dev/shm/imu_buffer.npy')
        frame_bgr = cv2.imread('/dev/shm/latest_frame.jpg')
        if imu_win.shape[0] < IMU_WINDOW:
            return None, None
        return imu_win, frame_bgr
    except:
        return None, None

# ── 메인 루프 ─────────────────────────────────────────────────────────────
def run_hds(model, symbolic):
    raw_buf  = collections.deque(maxlen=300)
    ema      = [0.0]

    def norm_running(val):
        raw_buf.append(val)
        arr = np.array(raw_buf)
        mn, mx = arr.min(), arr.max()
        rng = mx - mn
        if rng < 1e-4:
            normalized = 0.0
        else:
            normalized = float(np.clip((val - mn) / rng, 0, 1))
        ema[0] = 0.85 * ema[0] + 0.15 * normalized
        return ema[0]

    print("[HDS] 추론 시작. Ctrl+C 로 종료.\n")
    last_infer   = time.time()
    infer_count  = 0
    wait_printed = False
    ds_score = hds_score = G = delta_sym = delta_hw = 0.0
    alert = False

    try:
        while True:
            imu_win, frame_bgr = get_state()

            if imu_win is None:
                if not wait_printed:
                    print("  IMU 버퍼 대기 중...")
                    wait_printed = True
                if frame_bgr is not None:
                    disp = frame_bgr.copy()
                    cv2.putText(disp, "IMU waiting...", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                    cv2.imshow("G1 HDS", disp)
                    cv2.waitKey(1)
                time.sleep(0.1)
                continue
            wait_printed = False

            now = time.time()
            if now - last_infer >= 1.0 / STREAM_HZ:
                last_infer = now
                raw_pred  = infer_deepsee(model, imu_win, frame_bgr)
                ds_score  = norm_running(raw_pred)
                G         = image_quality_guardrail(frame_bgr) if frame_bgr is not None else 0.0
                delta_sym = symbolic.get_delta(ds_score)
                delta_hw  = W_HW * G
                hds_score = float(np.clip(ds_score + delta_sym + delta_hw, 0, 1))
                alert     = hds_score >= THRESHOLD
                infer_count += 1

                bar = '#' * int(hds_score * 20)
                tag = "*** ALERT ***" if alert else "Normal       "
                print(f"\r  [{tag}] DS={ds_score:.3f} G={G:.3f} HDS={hds_score:.3f}  [{bar:<20}]", end='')

            if frame_bgr is not None:
                disp  = frame_bgr.copy()
                color = (0, 0, 255) if alert else (0, 200, 0)
                cv2.putText(disp, f"DeepSEE: {ds_score:.3f}",  (10, 35),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.putText(disp, f"HDS:     {hds_score:.3f}", (10, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(disp, f"G:       {G:.3f}  hw:{delta_hw:.3f}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)
                cv2.putText(disp, "DRIFT RISK!" if alert else "Normal",
                            (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                if symbolic.is_active():
                    cv2.putText(disp, f"Symbolic +{delta_sym:.2f}",
                                (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 80, 255), 2)
                cv2.imshow("G1 HDS", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n[HDS] 종료.")
    finally:
        symbolic.stop()
        cv2.destroyAllWindows()
        print(f"[HDS] 총 {infer_count}회 추론 완료.")

if __name__ == '__main__':
    print("=== HDS G1 Local 추론 ===\n")
    print(f"모델: {MODEL_PATH}\n")
    print("[1/2] DeepSEE 모델 로드...")
    model = load_model()
    print("  OK\n")
    print("[2/2] Symbolic Layer 초기화...")
    symbolic = SymbolicLayer()
    print("  OK\n")
    run_hds(model, symbolic)
