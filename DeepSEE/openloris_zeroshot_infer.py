"""
Apr07 SenseTime 모델로 OpenLORIS zero-shot 추론
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

random.seed(2024)
torch.manual_seed(2024)

sys.path.insert(0, "/home/junhyun/SEESys/DeepSEE/Training")

from deepSEEConfig import DeepSEEConfig
from ts2vec.ts2vec import TS2Vec
from datasets.data import DeepSEEData
from datasets import augmentation
from models.DeepSEEModels import (
    TimesformerConfig, PatchTSMixerConfig, MultiModalCrossAttentionConfig,
    DeepSEEModel
)
from accelerate import Accelerator
from accelerate.utils import set_seed
from sklearn.ensemble import RandomForestRegressor
from torchmetrics.regression import MeanAbsolutePercentageError

APR07_DIR   = "/home/junhyun/SEESys/DeepSEE/Training/runs/Apr07_13-17-22_AHRI-Junhyun"
OUT_DIR     = "/home/junhyun/SEESys/DeepSEE/Training/runs/OpenLORIS_zeroshot"
os.makedirs(OUT_DIR, exist_ok=True)

config = DeepSEEConfig(
    seed=2024, n_proc=8, window_step=5, lr=1e-6, epochs=2, batchsize=128,
    data_dir="/home/junhyun/SEESys/DeepSEE/Training/datasets",
    datasets=['SenseTime'],
    target_transform=True,
)

class SLAMDataset(Dataset):
    def __init__(self, slam_data, augment=False, onlyOneIter=False,
                 lower=None, upper=None, std=None):
        self.slam_data   = slam_data
        self.augment     = augment
        self.onlyOneIter = onlyOneIter
        self.lower = lower; self.upper = upper; self.std = std
        self.generate_sample_list(slam_data)

    def generate_sample_list(self, slam_data):
        self.sample_list = []
        for dataset in slam_data:
            for traj in slam_data[dataset]:
                for m, it in enumerate(slam_data[dataset][traj]):
                    for time_stamp in slam_data[dataset][traj][it]:
                        if self.onlyOneIter:
                            if m == 0:
                                self.sample_list.append([dataset, traj, it, time_stamp])
                        else:
                            self.sample_list.append([dataset, traj, it, time_stamp])

    def __len__(self): return len(self.sample_list)

    def __getitem__(self, idx):
        dataset, traj, it, time_stamp = self.sample_list[idx]
        data_dict = self.slam_data[dataset][traj][it][time_stamp]
        time_series    = data_dict['timeSeries'].values
        point_dist     = DeepSEEData.load_point_dist_from_path(data_dict['pointDist_path'])
        relative_error = data_dict['relativeError']
        if self.lower is not None:
            time_series = np.where(time_series < self.lower, self.lower, time_series)
            time_series = np.where(time_series > self.upper, self.upper, time_series)
            time_series = time_series / self.std
        return {
            'time_series':    torch.from_numpy(time_series),
            'point_dist':     torch.from_numpy(point_dist),
            'relative_error': torch.tensor(relative_error).reshape([1])
        }

def assign_traj(slam_data, set_config):
    set_data = {}
    for dataset in set_config:
        set_data[dataset] = {}
        for traj in set_config[dataset]:
            if traj in slam_data.data[dataset]:
                set_data[dataset][traj] = slam_data.data[dataset][traj]
    return set_data

def MergeDict(*dicts):
    res = {}
    for d in dicts:
        res.update(d)
    return res

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("SenseTime 데이터 로드 중...")
slam_data_st = DeepSEEData(root_dir=config.data_dir, dataset_list=["SenseTime"],
                            n_proc=config.n_proc, config=config)
print("OpenLORIS 데이터 로드 중...")
slam_data_ol = DeepSEEData(root_dir=config.data_dir, dataset_list=["OpenLORIS"],
                            n_proc=config.n_proc, config=config)

slam_data = slam_data_st
slam_data.data["OpenLORIS"] = slam_data_ol.data["OpenLORIS"]

# ── 통계 (SenseTime 기준) ─────────────────────────────────────────────────────
print("데이터 통계 계산 중...")
unsup_pretrain_data = assign_traj(slam_data, {
    "SenseTime": ['A0','A1','A2','A3','A4','A5','A6','A7',
                  'B0','B1','B2','B3','B4','B5','B6','B7']
})
unsup_set = SLAMDataset(unsup_pretrain_data, augment=False, onlyOneIter=True)
all_data = []
for i in tqdm(range(len(unsup_set)), desc="통계"):
    d = unsup_set[i]['time_series']
    all_data.append(d.numpy() if isinstance(d, torch.Tensor) else d)
all_data = np.stack(all_data)

Q1 = np.percentile(all_data, 25, axis=0)
Q3 = np.percentile(all_data, 75, axis=0)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
clipped = np.clip(all_data, lower_bound, upper_bound)
std = np.std(clipped, axis=0); std[std == 0] = 1.0
print("통계 완료")

# ── 모델 로드 ─────────────────────────────────────────────────────────────────
print("TS2Vec 로드...")
_sd = torch.load(f"{APR07_DIR}/pretrained_model.pkl", weights_only=False, map_location='cpu')
input_dims = _sd['module.input_fc.weight'].shape[1]
pretrain_model = TS2Vec(input_dims=input_dims, device=0, batch_size=128, output_dims=64)
pretrain_model.net.load_state_dict(_sd)

config_PD = TimesformerConfig(image_size=128, patch_size=8, num_channels=3, num_frames=4,
    num_hidden_layers=3, num_attention_heads=8, intermediate_size=256, hidden_size=192)
config_TS = PatchTSMixerConfig(context_length=30, prediction_length=1, num_input_channels=64,
    d_model=192, patch_len=10, patch_stride=5, use_positional_encoding=True,
    ca_d_model=128, num_layers=3, drop_out=0.0)
config_CA = MultiModalCrossAttentionConfig(ts2vec_only=True, ts2vec_dim=64,
    ts_context_length=30, ts_patch_len=5, ts_num_input_channels=64, ts_patch_stride=5,
    ts_d_model=192, ts_time_step=33, pd_d_model=192, pd_time_step=330, pe_max_len=10000,
    ca_d_model=128, ca_num_head=16, ca_num_layers=2, ca_dropout=0.0,
    ca_time_series_only=False, output_range=None)

model = DeepSEEModel(config_PD, config_TS, config_CA)
state_dict = torch.load(f"{APR07_DIR}/test_model.pth", weights_only=False)
model.load_state_dict(state_dict)
model.eval()
print("모델 로드 완료")

def ensemble_tree_data_loader(train_set, val_set, test_set):
    def load_data(ds):
        X, Y = [], []
        for i in range(len(ds)):
            item = ds[i]
            ts = item['time_series']
            if ts.dim() == 2: ts = ts[0]
            X.append(ts.numpy())
            Y.append(item['relative_error'].numpy().flatten()[0])
        return np.stack(X), np.array(Y)
    return (*load_data(train_set), *load_data(val_set), *load_data(test_set))

def evaluate_on_test(model, test_loader):
    set_seed(config.seed)
    accelerator = Accelerator(mixed_precision="bf16")
    model, test_dl = accelerator.prepare(model, test_loader)
    model.eval()
    Y_gt, Y_est = [], []
    for batch in test_dl:
        ts_np = batch['time_series'].cpu().numpy()
        ts_enc = pretrain_model.encode(ts_np, causal=True, sliding_length=1, sliding_padding=5)
        ts    = torch.from_numpy(ts_enc).to(accelerator.device, dtype=torch.float)
        pd    = batch['point_dist'].to(accelerator.device, dtype=torch.float)
        re    = batch['relative_error'].to(accelerator.device, dtype=torch.float)
        with torch.no_grad():
            out = model(pd, ts)
            accelerator.gather_for_metrics((out, re))
        Y_gt.append(re.cpu()); Y_est.append(out.cpu())
    return Y_gt, Y_est

# ── OpenLORIS 시퀀스 (유효한 것만) ───────────────────────────────────────────
SEQS = ["cafe1-1", "cafe1-2", "office1-2", "office1-5", "office1-6", "office1-7"]

sup_pretrain_train_data = assign_traj(slam_data, {"SenseTime": ['A0','A1','A2','A3']})

test_params  = {'batch_size': config.batchsize, 'shuffle': False, 'num_workers': 4}
train_params = {'batch_size': config.batchsize, 'shuffle': True,  'num_workers': 4}
val_params   = {'batch_size': config.batchsize, 'shuffle': False, 'num_workers': 4}

print("\n" + "="*60)
print("OpenLORIS Zero-Shot 추론 (Apr07 SenseTime 모델)")
print("="*60)

for idx, test_seq in enumerate(SEQS):
    print(f"\n[Fold {idx}] test: {test_seq}")
    train_seqs = [s for s in SEQS if s != test_seq]

    train_data = assign_traj(slam_data, {"OpenLORIS": train_seqs})
    val_data   = assign_traj(slam_data, {"OpenLORIS": [test_seq]})
    test_data  = assign_traj(slam_data, {"OpenLORIS": [test_seq]})

    train_set = SLAMDataset(train_data, augment=False, onlyOneIter=False,
                             lower=lower_bound, upper=upper_bound, std=std)
    val_set   = SLAMDataset(val_data,   augment=False, onlyOneIter=False,
                             lower=lower_bound, upper=upper_bound, std=std)
    test_set  = SLAMDataset(test_data,  augment=False, onlyOneIter=True,
                             lower=lower_bound, upper=upper_bound, std=std)

    print(f"  train: {len(train_set)} samples, test: {len(test_set)} samples")
    test_loader = DataLoader(test_set, **test_params)

    # RF baseline (다른 OpenLORIS 5개 + SenseTime 4개)
    train_data_bl = {**train_data, **sup_pretrain_train_data}
    train_set_bl  = SLAMDataset(train_data_bl, augment=False, onlyOneIter=False,
                                 lower=lower_bound, upper=upper_bound, std=std)
    XT, YT, XV, YV, XTe, YTe = ensemble_tree_data_loader(train_set_bl, val_set, test_set)
    regr = RandomForestRegressor(n_estimators=25, max_features='sqrt', min_samples_split=2,
                                  min_samples_leaf=2, bootstrap=False, max_depth=10,
                                  random_state=0, n_jobs=-1)
    regr.fit(XT, YT)
    YT_base = regr.predict(XTe)

    Y_gt, Y_est = evaluate_on_test(model, test_loader)

    np.save(f"{OUT_DIR}/SupervisedFinetune_{idx}_Y_gt.npy",   np.array(Y_gt,  dtype=object), allow_pickle=True)
    np.save(f"{OUT_DIR}/SupervisedFinetune_{idx}_Y_est.npy",  np.array(Y_est, dtype=object), allow_pickle=True)
    np.save(f"{OUT_DIR}/SupervisedFinetune_{idx}_Y_base.npy", YT_base, allow_pickle=True)
    print(f"  Saved fold {idx} → {test_seq}")

print(f"\n완료! 결과: {OUT_DIR}")
