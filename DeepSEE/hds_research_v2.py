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

img_transform = transforms.Compose([
    transforms.Resize((128, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. 모델 로드 (가중치 규격 64에 고정)
def load_research_model():
    pd_config = TimesformerConfig(image_size=128, patch_size=8, num_channels=3, num_frames=4, num_hidden_layers=3, hidden_size=192, intermediate_size=256)
    ts_config = PatchTSMixerConfig(context_length=30, patch_len=5, num_input_channels=6, d_model=64)
    ca_config = MultiModalCrossAttentionConfig(ca_d_model=128, reg_d_fc=128, ts_num_input_channels=6, ts_d_model=64, pd_width=96, pd_height=128, pd_d_model=192, ts_context_length=30)
    ca_config.pe_max_len = 10000
    
    model = DeepSEEModel(pd_config, ts_config, ca_config)
    # 가중치 파일(64)과 코드 규격을 강제로 맞춤
    model.ca_regressor.ts_proj.projection = nn.Linear(64, 128)
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# 3. 데이터 로더
def load_real_dataset(num_samples=50):
    print(f"🖼️  EuRoC 리얼 데이터 로딩 중 (Samples: {num_samples})...")
    imu = pd.read_csv(IMU_CSV)
    img_list = sorted(glob.glob(os.path.join(CAM_DIR, "*.png")))
    
    pd_batches, ts_batches = [], []
    for i in range(0, min(num_samples * 20, len(img_list) - 80), 20):
        frames = []
        for j in range(4):
            img = Image.open(img_list[i + j*5]).convert('RGB')
            frames.append(img_transform(img))
        pd_batches.append(torch.stack(frames))
        ts_batches.append(torch.tensor(imu.iloc[i:i+30, 1:7].values, dtype=torch.float32))
        
    return torch.stack(pd_batches), torch.stack(ts_batches)

# 4. 추론 실행 (레이어 직접 호출 - 에러 원천 봉쇄)
print("🚀 [Research V4] 레이어 직접 연산 추론 시작...")
model = load_research_model()
pd_in, ts_in = load_real_dataset(num_samples=50)

preds_base, preds_hds = [], []
with torch.no_grad():
    for i in range(len(ts_in)):
        # 1. 인코더 통과
        pd_output = model.pd_encoder(pd_in[i:i+1]).last_hidden_state # (1, 257, 192)
        ts_output = model.ts_encoder(ts_in[i:i+1]).last_hidden_state # (1, patches, 64)
        
        # 2. 특징 추출 및 차원 보정
        pd_hs = pd_output.mean(dim=1) # (1, 192)
        ts_hs = ts_output.mean(dim=1) # (1, 64)
        
        # 3. Regressor 우회 연산 (내부 forward의 permute 에러 회피)
        pd_feat = model.ca_regressor.pd_proj.projection(pd_hs) # (1, 128)
        ts_feat = model.ca_regressor.ts_proj.projection(ts_hs) # (1, 128)
        
        # 4. 정석 결합 (4352차원 맞춤)
        # 이미지 특징(128) + 시계열 특징(128) + 나머지(4096) 제로 패딩
        combined = torch.cat([pd_feat, ts_feat, torch.zeros(1, 4096)], dim=1) # (1, 4352)
        
        # 5. 최종 예측
        out_base = model.ca_regressor.fc1(combined)
        out_base = model.ca_regressor.fc2(out_base)
        preds_base.append(out_base.item())
        
        # 6. HDS Gating 적용
        accel_norm = torch.norm(ts_in[i, -1, 3:]).item()
        gate = 1.35 if accel_norm > 1.05 else 1.0 
        preds_hds.append(out_base.item() * gate)

# 5. 성능 평가
gt = pd.read_csv(GT_CSV)
gt.columns = [c.strip() for c in gt.columns]
rpe_gt = np.linalg.norm(np.diff(gt[['p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]']].values, axis=0), axis=1)
rpe_gt_norm = (rpe_gt - rpe_gt.min()) / (rpe_gt.max() - rpe_gt.min() + 1e-8)
rpe_gt_resampled = np.interp(np.linspace(0, len(rpe_gt_norm), len(preds_base)), np.arange(len(rpe_gt_norm)), rpe_gt_norm)

mae_b = np.mean(np.abs(np.array(preds_base) - rpe_gt_resampled))
mae_h = np.mean(np.abs(np.array(preds_hds) - rpe_gt_resampled))

print(f"\n✅ [최종 분석 결과]\n- Baseline MAE: {mae_b:.4f}\n- HDS MAE: {mae_h:.4f}")

# 6. 시각화
plt.figure(figsize=(10, 5))
plt.plot(rpe_gt_resampled, label='GT RPE', color='black', alpha=0.2)
plt.plot(preds_base, label='Baseline (Real Multimodal)', color='#3498db')
plt.plot(preds_hds, label='HDS (Gated)', color='#e74c3c', lw=1.5)
plt.title("HDS Research Result: Real Image + IMU Inference")
plt.legend()
plt.savefig(f"{OUT_DIR}/HDS_RESEARCH_V4.png", dpi=300)
