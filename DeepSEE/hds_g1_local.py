"""
HDS G1 추론 (conda deepsee Python 3.9으로 실행)
/dev/shm에서 RTS(16ch) + PSD 읽어서 TS2Vec 인코딩 후 DeepSEE 추론
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
from transformers import TimesformerConfig, PatchTSMixerConfig

MODEL_PATH    = os.path.join(_HERE, "runs", "SupervisedFinetune_1_best_model.pth")
TS2VEC_PATH   = os.path.join(_HERE, "runs", "pretrained_model.pkl")
NORM_DIR      = os.path.join(_HERE, "runs")
CALIB_PATH    = os.path.join(_HERE, "runs", "ds_calib.npz")   # make_ds_calib.py로 생성


def load_ds_calib():
    """DeepSEE raw 출력 → [0,1] 고정 affine 변환 (lo, hi). 없으면 None."""
    try:
        z = np.load(CALIB_PATH)
        return float(z['lo']), float(z['hi'])
    except Exception:
        return None

THRESHOLD         = 0.6
STREAM_HZ         = 10
W_HW              = 0.3
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
                    model="claude-sonnet-4-6", max_tokens=128, temperature=0.0,
                    messages=[{"role": "user", "content": prompt}]
                )
                if not msg.content:
                    raise ValueError("LLM 응답이 비어 있음")
                raw    = re.sub(r'```[a-z]*\n?', '', msg.content[0].text.strip()).strip().rstrip('`')
                result = json.loads(raw)
                delta  = float(result['risk_delta'])
                reason = result['reason']
                with self._lock:
                    self._active.append([delta, SYMBOLIC_DURATION, reason])
                print(f"\n  [LLM] risk_delta={delta:.3f} | {reason}")
                print(f"  → {SYMBOLIC_DURATION/STREAM_HZ:.0f}초 동안 적용\n", flush=True)
            except Exception as e:
                # 네트워크/API 장애 시 키워드 기반 규칙으로 폴백 (실시간 안전 보장)
                delta = self._keyword_fallback(text)
                if delta > 0:
                    with self._lock:
                        self._active.append([delta, SYMBOLIC_DURATION, f"[fallback] {text[:40]}"])
                    print(f"\n  [LLM 오류→폴백] risk_delta={delta:.3f} ({e})\n", flush=True)
                else:
                    print(f"\n  [LLM 오류] {e}\n", flush=True)

    @staticmethod
    def _keyword_fallback(text):
        """LLM 불가 시 운영자 텍스트에서 위험 키워드를 규칙 매칭."""
        t = text.lower()
        severe   = ['암흑', '칠흑', 'pitch dark', '특징점 없', 'no feature', '급격', 'rapid']
        moderate = ['어두', '저조도', 'dark', 'low light', '블러', 'blur', '흔들', '빠른', 'fast']
        if any(k in t for k in severe):
            return 0.30
        if any(k in t for k in moderate):
            return 0.15
        return 0.0

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


# ── TS2Vec 래퍼 ───────────────────────────────────────────────────────────
class TS2VecEncoder:
    """pretrained_model.pkl 로드 후 추론 (input=16, output=64)"""
    def __init__(self, ckpt_path):
        from ts2vec.ts2vec import TS2Vec
        self.model = TS2Vec(input_dims=16, output_dims=64, device='cpu')
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        result = self.model.net.load_state_dict(new_state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"  [TS2Vec 경고] missing={len(result.missing_keys)} "
                  f"unexpected={len(result.unexpected_keys)} — config 불일치 의심")
        self.model.net.eval()

    def encode(self, x_np):
        """x_np: (30, 16) → (30, 64)"""
        inp = x_np[np.newaxis]  # (1, 30, 16)
        rep = self.model.encode(inp, causal=True, sliding_length=1, sliding_padding=5)
        return rep[0]           # (30, 64)


# ── 정규화 ────────────────────────────────────────────────────────────────
def load_norm_stats():
    lower = np.load(os.path.join(NORM_DIR, "rts_norm_lower.npy"))
    upper = np.load(os.path.join(NORM_DIR, "rts_norm_upper.npy"))
    std   = np.load(os.path.join(NORM_DIR, "rts_norm_std.npy"))
    return lower, upper, std

def normalize_rts(rts_window, lower, upper, std):
    """(30, 16) → 훈련과 동일한 IQR clip + std 정규화"""
    x = np.clip(rts_window, lower, upper)
    return (x / (std + 1e-8)).astype(np.float32)


# ── 모델 로드 ─────────────────────────────────────────────────────────────
def load_model():
    pd_config = TimesformerConfig(
        image_size=128, patch_size=8, num_channels=3,
        num_frames=4, num_hidden_layers=3, num_attention_heads=12,
        hidden_size=192, intermediate_size=256, hidden_dropout_prob=0,
    )
    ts_config = PatchTSMixerConfig(
        context_length=30, patch_len=1, num_input_channels=6, d_model=64
    )
    # ts2vec_only=True: PatchTSMixer 건너뜀, ts_proj = Linear(64, 128)
    ca_config = MultiModalCrossAttentionConfig(
        ts2vec_only=True, ts2vec_dim=64,
        ca_d_model=128, ca_num_head=16, ca_num_layers=2,
        reg_d_fc=128,
        ts_num_input_channels=64, ts_d_model=192, ts_time_step=33,
        pd_width=96, pd_height=128, pd_d_model=192,
        pd_time_step=330,
        ts_context_length=30,
        pe_max_len=10000,
    )
    model = DeepSEEModel(pd_config, ts_config, ca_config)

    import __main__
    if not hasattr(__main__, 'loss_fn'):
        __main__.loss_fn = lambda pred, target: pred

    ckpt       = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # 핵심 모듈(회귀기/인코더) 가중치가 누락되면 랜덤 초기화로 조용히 동작 → 중단
    _critical = ('ca_regressor', 'pd_encoder')
    crit_missing = [k for k in missing if k.startswith(_critical)]
    if crit_missing:
        raise RuntimeError(
            f"가중치 로드 실패: 핵심 모듈 키 {len(crit_missing)}개 누락 "
            f"(예: {crit_missing[:3]}). 모델 config가 체크포인트와 불일치합니다.")
    if missing or unexpected:
        print(f"  [경고] missing={len(missing)} unexpected={len(unexpected)} "
              f"(비핵심 키만 누락이면 정상)")
    model.eval()
    return model


# ── 버퍼 읽기 ────────────────────────────────────────────────────────────
def _read_rts():
    """C++ ORB-SLAM3가 쓴 rts_buffer.bin: (30, 16) float32"""
    try:
        raw = np.fromfile('/dev/shm/rts_buffer.bin', dtype=np.float32)
        if raw.size == 30 * 16:
            return raw.reshape(30, 16)
    except Exception:
        pass
    return None

def _read_psd():
    """C++ ORB-SLAM3가 쓴 psd_buffer.bin: (4, 3, 96, 128) float32"""
    try:
        raw = np.fromfile('/dev/shm/psd_buffer.bin', dtype=np.float32)
        if raw.size == 4 * 3 * 96 * 128:
            return raw.reshape(4, 3, 96, 128)
    except Exception:
        pass
    return None

_orb = cv2.ORB_create(500)

def _fallback_psd(frame_bgr):
    small  = cv2.resize(frame_bgr, (128, 96))
    gray   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    kps    = _orb.detect(gray, None)
    canvas = np.zeros((96, 128), dtype=np.float32)
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= x < 128 and 0 <= y < 96:
            canvas[y, x] = min(kp.response / 1000.0, 1.0)
    ch = canvas[None]
    frame = np.concatenate([ch, ch, ch], axis=0)
    return np.stack([frame] * 4, axis=0)


# ── DeepSEE 추론 ──────────────────────────────────────────────────────────
def infer_deepsee(model, ts2vec, norm_stats, frame_bgr):
    lower, upper, std = norm_stats

    rts_raw = _read_rts()
    if rts_raw is None:
        return None

    rts_norm = normalize_rts(rts_raw, lower, upper, std)   # (30, 16)
    ts_enc   = ts2vec.encode(rts_norm)                     # (30, 64)
    ts = torch.tensor(ts_enc[np.newaxis], dtype=torch.float32)  # (1, 30, 64)

    psd_np = _read_psd()
    if psd_np is None and frame_bgr is not None:
        psd_np = _fallback_psd(frame_bgr)
    if psd_np is None:
        return None

    pd = torch.tensor(psd_np, dtype=torch.float32).unsqueeze(0)  # (1, 4, 3, 96, 128)

    with torch.no_grad():
        out = model(pd, ts)
    val = float(out.item())
    if not np.isfinite(val):
        return None
    return val


# ── Image Quality Guardrail ───────────────────────────────────────────────
def image_quality_guardrail(frame_bgr):
    gray       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    brightness = gray.mean() / 255.0
    hist, _    = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist       = hist[hist > 0]
    entropy    = float(-np.sum(hist * np.log2(hist + 1e-10))) / 8.0
    lap        = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
    lap_norm   = min(lap / 1000.0, 1.0)
    kps        = _orb.detect(gray.astype(np.uint8), None)
    feat_score = 1.0 - min(len(kps) / 300.0, 1.0)
    return float(W_ENTROPY * (1 - entropy) + W_BRIGHT * (1 - brightness)
                 + W_LAP * (1 - lap_norm) + 0.1 * feat_score)


def get_frame():
    try:
        data = open('/dev/shm/latest_frame.jpg', 'rb').read()
        arr  = np.frombuffer(data, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


# ── 메인 루프 ─────────────────────────────────────────────────────────────
def run_hds(model, ts2vec, norm_stats, symbolic):
    raw_buf = collections.deque(maxlen=300)
    ema     = [0.0]

    import csv
    log_path = os.path.join(_HERE, f"hds_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    log_f = open(log_path, 'w', newline='')
    log_w = csv.writer(log_f)
    log_w.writerow(['t', 'ds_raw', 'ds', 'G', 'hw', 'sym', 'hds', 'alert'])
    print(f"[HDS] 로그 기록: {log_path}")

    def norm_running(val):
        if not np.isfinite(val):
            return ema[0]
        raw_buf.append(val)
        arr = np.array(raw_buf)
        mn, mx = arr.min(), arr.max()
        rng = mx - mn
        normalized = 0.0 if rng < 1e-4 else float(np.clip((val - mn) / rng, 0, 1))
        ema[0] = 0.85 * ema[0] + 0.15 * normalized
        return ema[0]

    calib = load_ds_calib()
    if calib is not None:
        print(f"[HDS] 고정 캘리브레이션 사용: lo={calib[0]:.3f} hi={calib[1]:.3f} (결정론적, 무지연)")
    else:
        print("[HDS] ds_calib.npz 없음 → rolling min-max+EMA 폴백 (재현성/지연 주의)")

    print("[HDS] 추론 시작. Ctrl+C 로 종료.\n")
    last_infer   = time.time()
    infer_count  = 0
    wait_printed = False
    ds_score = hds_score = G = delta_sym = delta_hw = 0.0
    alert = False

    try:
        while True:
            frame_bgr = get_frame()

            if not os.path.exists('/dev/shm/rts_buffer.bin'):
                if not wait_printed:
                    print("  RTS 버퍼 대기 중 (ORB-SLAM3 실행 필요)...")
                    wait_printed = True
                if frame_bgr is not None:
                    disp = frame_bgr.copy()
                    cv2.putText(disp, "RTS waiting...", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    cv2.imshow("G1 HDS", disp)
                    cv2.waitKey(1)
                time.sleep(0.1)
                continue
            wait_printed = False

            now = time.time()
            if now - last_infer >= 1.0 / STREAM_HZ:
                last_infer = now
                raw_pred  = infer_deepsee(model, ts2vec, norm_stats, frame_bgr)
                if raw_pred is not None:
                    if calib is not None:
                        ds_score = float(np.clip((raw_pred - calib[0]) / (calib[1] - calib[0]), 0, 1))
                    else:
                        ds_score = norm_running(raw_pred)
                    G         = image_quality_guardrail(frame_bgr) if frame_bgr is not None else 0.0
                    delta_sym = symbolic.get_delta(ds_score)
                    delta_hw  = max(0.0, G - 0.20) * W_HW
                    hds_score = float(np.clip(ds_score + delta_sym + delta_hw, 0, 1))
                    if not np.isfinite(hds_score):
                        hds_score = 0.0
                    alert     = hds_score >= THRESHOLD
                    infer_count += 1

                    log_w.writerow([f"{now:.3f}", f"{raw_pred:.5f}", f"{ds_score:.4f}",
                                    f"{G:.4f}", f"{delta_hw:.4f}", f"{delta_sym:.4f}",
                                    f"{hds_score:.4f}", int(alert)])
                    log_f.flush()

                    bar = '#' * int(hds_score * 20)
                    tag = "*** ALERT ***" if alert else "Normal       "
                    print(f"\r  [{tag}] DS={ds_score:.3f} G={G:.3f} hw={delta_hw:.3f} HDS={hds_score:.3f}  [{bar:<20}]", end='')

            if frame_bgr is not None:
                disp  = frame_bgr.copy()
                color = (0, 0, 255) if alert else (0, 200, 0)
                cv2.putText(disp, f"DeepSEE: {ds_score:.3f}",  (10, 35),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(disp, f"HDS:     {hds_score:.3f}", (10, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(disp, f"G:{G:.3f} hw:{delta_hw:.3f}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
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
        log_f.close()
        print(f"[HDS] 총 {infer_count}회 추론 완료. 로그 저장: {log_path}")


if __name__ == '__main__':
    print("=== HDS G1 Local 추론 ===\n")
    print(f"모델:   {MODEL_PATH}")
    print(f"TS2Vec: {TS2VEC_PATH}\n")

    print("[1/4] 정규화 통계 로드...")
    norm_stats = load_norm_stats()
    print("  OK\n")

    print("[2/4] TS2Vec 인코더 로드...")
    ts2vec = TS2VecEncoder(TS2VEC_PATH)
    print("  OK\n")

    print("[3/4] DeepSEE 모델 로드...")
    model = load_model()
    print("  OK\n")

    print("[4/4] Symbolic Layer 초기화...")
    symbolic = SymbolicLayer()
    print("  OK\n")

    run_hds(model, ts2vec, norm_stats, symbolic)
