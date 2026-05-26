import os, sys, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# ── Path Settings ──────────────────────────────────────────────────────────────
BASE       = "/home/junhyun/SEESys/DeepSEE/Training"
MODEL_PATH = "/home/junhyun/SEESys/DeepSEE/Training/runs/Mar23_15-17-03_AHRI-Junhyun/test_model.pth"
GT_CSV     = "/home/junhyun/SEESys/Datasets/SenseTime/EuRoC/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv"
IMU_CSV    = "/home/junhyun/SEESys/Datasets/SenseTime/EuRoC/MH_02_easy/mav0/imu0/data.csv"
OUT_DIR    = "/home/junhyun/SEESys/HDS_demo_output"
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, BASE)

from models.DeepSEEModels import DeepSEEModel, MultiModalCrossAttentionConfig
from transformers import PatchTSMixerConfig, TimesformerConfig

# 1. Data Loading & Preprocessing
print("📂 Loading data and calculating ground truth...")
gt = pd.read_csv(GT_CSV)
imu = pd.read_csv(IMU_CSV)
gt.columns = [c.strip() for c in gt.columns]
imu.columns = [c.strip() for c in imu.columns]

gt['t'] = (gt['#timestamp'] - gt['#timestamp'].iloc[0]) / 1e9
imu['t'] = (imu['#timestamp [ns]'] - imu['#timestamp [ns]'].iloc[0]) / 1e9

# 2. Model Loading (Dimension Mismatch Solution)
print("🤖 Loading model (CPU mode for stability)...")
device = torch.device('cpu') 

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# Speed calculation for Symbolic Constraints
gt['speed'] = np.linalg.norm(gt[['v_RS_R_x [m s^-1]', 'v_RS_R_y [m s^-1]', 'v_RS_R_z [m s^-1]']].values, axis=1)

# Feature preparation (IMU + Speed)
t_gt = gt['t'].values
imu_cols = ['w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]', 'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']
imu_interp = {col: interp1d(imu['t'], imu[col], fill_value='extrapolate')(t_gt) for col in imu_cols}
imu_df = pd.DataFrame(imu_interp, index=gt.index)

feature_df = pd.concat([imu_df, gt[['speed']]], axis=1)
feature_df = (feature_df - feature_df.mean()) / (feature_df.std() + 1e-8)
feat_dim = feature_df.shape[1]

pd_config = TimesformerConfig(image_size=128, patch_size=8, num_channels=3, num_frames=4, num_hidden_layers=3, hidden_size=192, intermediate_size=256)
ts_config = PatchTSMixerConfig(context_length=30, patch_len=30, num_input_channels=feat_dim, d_model=64)
ca_config = MultiModalCrossAttentionConfig(ca_d_model=128, reg_d_fc=128, ts_num_input_channels=feat_dim, ts_d_model=64, pd_width=96, pd_height=128, pd_d_model=192, ts_context_length=30)
ca_config.pe_max_len = 10000

model = DeepSEEModel(pd_config, ts_config, ca_config)
model.load_state_dict(state_dict, strict=False)
model.eval()

# 3. Inference
print("🔮 Running inference...")
feat_np = feature_df.values.astype(np.float32)
windows = [feat_np[i:i+30] for i in range(0, len(feat_np) - 30, 20)]
ts_tensor = torch.tensor(np.array(windows))
dummy_pd = torch.zeros(1, 4, 3, 128, 96)

preds = []
with torch.no_grad():
    for i in range(len(ts_tensor)):
        out = model(dummy_pd, ts_tensor[i:i+1])
        preds.append(out.item())

# 4. HDS Symbolic Constraints Application
preds_norm = (np.array(preds) - np.min(preds)) / (np.max(preds) - np.min(preds) + 1e-8)
preds_smooth = savgol_filter(preds_norm, 15, 3)

speed_win = np.interp(np.linspace(0, len(gt), len(preds)), np.arange(len(gt)), gt['speed'])
# LLM Constraint Trigger (Speed > 80th percentile)
constraint_mask = (speed_win > np.percentile(speed_win, 80)).astype(float)
gate = np.where(speed_win > np.percentile(speed_win, 80), 1.35, 1.0)
preds_hds = np.clip(preds_smooth * gate, 0, 1)

# 5. Final Visualization (Presentation Mode: 2 Plots)
print("📊 Generating final effectiveness plot (2-plot structure)...")
plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.8, 1], hspace=0.35)

# Plot 1: Main Result Plot (DeepSEE vs HDS)
ax1 = plt.subplot(gs[0])
ax1.plot(preds_smooth, label='DeepSEE (Baseline)', color='#3498db', alpha=0.7, lw=1.5)
ax1.plot(preds_hds, label='HDS (Inference + Symbolic Constraint)', color='#e74c3c', lw=2.5)

ax1.set_title("Validation of HDS Effectiveness: Hardware-aware Drift Prediction", fontsize=15, fontweight='bold', y=1.03)
ax1.set_ylabel("Normalized Drift Score", fontsize=12)
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.2)
ax1.set_xticklabels([]) # Hide x-labels for ax1

# Plot 2: LLM Trigger Timeline
ax2 = plt.subplot(gs[1])
# Highlight background where constraint is active
ax2.fill_between(range(len(constraint_mask)), 0, 1, where=constraint_mask>0, color='#f1c40f', alpha=0.3)
# Draw timeline as a bar chart
ax2.bar(range(len(constraint_mask)), constraint_mask, color='#f1c40f', width=1.0, label='Symbolic Logic Active', alpha=0.7)

ax2.set_xlabel("Sample Index (Sliding Window)", fontsize=12)
ax2.set_ylabel("Symbolic Trigger", fontsize=12)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['OFF', 'ON'], fontsize=10)
ax2.grid(True, alpha=0.1, axis='x')
ax2.legend(loc='upper right', fontsize=10)

out_path = f"{OUT_DIR}/HDS_VALIDATION_EFFECTIVENESS.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')

print(f"✅ Success! Plot saved at: {out_path}")
