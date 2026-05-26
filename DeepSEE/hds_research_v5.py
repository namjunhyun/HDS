import os, sys, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# 1. 경로 설정
MODEL_PATH = "/home/junhyun/SEESys/DeepSEE/Training/runs/Mar23_15-17-03_AHRI-Junhyun/test_model.pth"
CAM_DIR    = "/home/junhyun/SEESys/Datasets/SenseTime/EuRoC/MH_02_easy/mav0/cam0/data"
IMU_CSV    = "/home/junhyun/SEESys/Datasets/SenseTime/EuRoC/MH_02_easy/mav0/imu0/data.csv"
GT_CSV     = "/home/junhyun/SEESys/Datasets/SenseTime/EuRoC/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv"
OUT_DIR    = "/home/junhyun/SEESys/HDS_research_output"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cpu')
sys.path.insert(0, "/home/junhyun/SEESys/DeepSEE/Training")
from models.DeepSEEModels import DeepSEEModel, MultiModalCrossAttentionConfig
from transformers import PatchTSMixerConfig, TimesformerConfig

# 이미지 전처리 (논문 표준 규격)
img_transform = transforms.Compose([
    transforms.Resize((128, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. 모델 로드 (Forward Path를 위한 순정 상태 복구)
def load_official_model():
    pd_config = TimesformerConfig(image_size=128, patch_size=8, num_channels=3, num_frames=4, num_hidden_layers=3, hidden_size=192, intermediate_size=256)
    # patch_len을 조절하여 내부 permute 에러가 나지 않도록 최적화 (64차원 유지)
    ts_config = PatchTSMixerConfig(context_length=30, patch_len=30, num_input_channels=6, d_model=64)
    ca_config = MultiModalCrossAttentionConfig(ca_d_model=128, reg_d_fc=128, ts_num_input_channels=6, ts_d_model=64, pd_width=96, pd_height=128, pd_d_model=192, ts_context_length=30)
    ca_config.pe_max_len = 10000
    
    model = DeepSEEModel(pd_config, ts_config, ca_config)
    
    # 체크포인트 규격과 레이어 입력 차원을 물리적으로 일치시킴 (64 -> 128)
    model.ca_regressor.ts_proj.projection = nn.Linear(64, 128)
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# 3. 데이터 로딩 (Real Multimodal)
def load_real_data(num_samples=40):
    print(f"📊 Real Data Loading... (Samples: {num_samples})")
    imu = pd.read_csv(IMU_CSV)
    img_list = sorted(glob.glob(os.path.join(CAM_DIR, "*.png")))
    
    pd_batches, ts_batches = [], []
    for i in range(0, num_samples * 20, 20):
        frames = [img_transform(Image.open(img_list[i+j*3]).convert('RGB')) for j in range(4)]
        pd_batches.append(torch.stack(frames))
        ts_batches.append(torch.tensor(imu.iloc[i:i+30, 1:7].values, dtype=torch.float32))
        
    return torch.stack(pd_batches), torch.stack(ts_batches)

# 4. 정석 Inference 및 Latent Gating (HDS)
print("🚀 [Research V5] 공식 Forward Path 기반 추론 시작...")
model = load_official_model()
pd_in, ts_in = load_real_data()

preds_base, preds_hds = [], []
with torch.no_grad():
    for i in range(len(ts_in)):
        # (1) Baseline: 공식 Forward Path 호출 (순정 모델 출력)
        # pd_in[i:i+1] shape: (1, 4, 3, 128, 96), ts_in[i:i+1] shape: (1, 30, 6)
        out_base = model(pd_in[i:i+1], ts_in[i:i+1])
        preds_base.append(out_base.item())
        
        # (2) HDS: Latent Feature Modulation (HDS 핵심 논리)
        # 단순 사후 처리가 아니라, 모델 인코더 결과값(Hidden States)을 가로채서 
        # 하드웨어 제약 조건(Velocity)에 따른 Gating을 수행하는 시뮬레이션
        pd_hs = model.pd_encoder(pd_in[i:i+1]).last_hidden_state
        ts_hs_raw = model.ts_encoder(ts_in[i:i+1]).last_hidden_state
        ts_hs = ts_hs_raw.mean(dim=1) # 64차원 압축
        
        # Symbolic Gate: 고속 주행 시 IMU 특징의 영향력을 강화 (1.35x)
        speed = torch.norm(ts_in[i, -1, 3:]).item()
        gate = 1.35 if speed > 1.1 else 1.0
        
        # Gating이 적용된 Hidden State로 Regressor 직접 호출 (Internal Reasoning)
        out_hds = model.ca_regressor(pd_hs, ts_hs * gate) 
        preds_hds.append(out_hds.item())

# 5. 성능 평가 및 MAE 산출 (정밀 분석)
gt = pd.read_csv(GT_CSV)
gt.columns = [c.strip() for c in gt.columns]
rpe_gt = np.linalg.norm(np.diff(gt[['p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]']].values, axis=0), axis=1)
rpe_gt_resampled = np.interp(np.linspace(0, len(rpe_gt), len(preds_base)), np.arange(len(rpe_gt)), rpe_gt)
rpe_gt_norm = (rpe_gt_resampled - rpe_gt_resampled.min()) / (rpe_gt_resampled.max() - rpe_gt_resampled.min() + 1e-8)

mae_b = np.mean(np.abs(np.array(preds_base) - rpe_gt_norm))
mae_h = np.mean(np.abs(np.array(preds_hds) - rpe_gt_norm))

print(f"\n📈 [논문용 실험 지표]\n- Baseline MAE: {mae_b:.4f}\n- HDS (Latent Gate) MAE: {mae_h:.4f}")
print(f"- Accuracy Improvement: {((mae_b - mae_h) / mae_b * 100):.2f}%")

# 6. 시각화 (Publication Quality)
plt.figure(figsize=(12, 6))
plt.plot(rpe_gt_norm, label='Ground Truth (EuRoC RPE)', color='black', alpha=0.15)
plt.plot(preds_base, label='Baseline (Official DeepSEE)', color='#3498db', ls='--')
plt.plot(preds_hds, label='HDS (Ours: Latent Gating)', color='#e74c3c', lw=2)
plt.title("HDS Validation: Neuro-Symbolic Latent Modulation on Real Data")
plt.legend()
plt.savefig(f"{OUT_DIR}/HDS_RESEARCH_V5_FINAL.png", dpi=300)
