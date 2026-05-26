"""
Apr07 SenseTime 모델로 EuRoC zero-shot 추론
학습 없이 추론만 수행 → SupervisedFinetune_{idx}_Y_*.npy 저장
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
import sys
import copy
import pickle
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

APR07_DIR = "/home/junhyun/SEESys/DeepSEE/Training/runs/Apr07_13-17-22_AHRI-Junhyun"

# ── Dummy writer (로깅용, 학습 없음) ─────────────────────────────────────────
class DummyWriter:
    def add_scalar(self, *args, **kwargs): pass
    def get_logdir(self): return APR07_DIR
writer = DummyWriter()

# ── Config ────────────────────────────────────────────────────────────────────
config = DeepSEEConfig(
    seed=2024, n_proc=8, window_step=5, lr=1e-6, epochs=2, batchsize=128,
    data_dir="/home/junhyun/SEESys/DeepSEE/Training/datasets",
    datasets=['SenseTime'],
    target_transform=True,
)
b_unsup_pretrain = True

# ── SLAMDataset (노트북과 동일) ───────────────────────────────────────────────
class SLAMDataset(Dataset):
    def __init__(self, slam_data, augment=False, onlyOneIter=False,
                 lower=None, upper=None, std=None):
        self.slam_data   = slam_data
        self.augment     = augment
        self.onlyOneIter = onlyOneIter
        self.lower = lower
        self.upper = upper
        self.std   = std
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

    def __len__(self):
        return len(self.sample_list)

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

        if self.augment:
            x = np.expand_dims(time_series, axis=0)
            x = augmentation.jitter(x, sigma=0.03)
            x = augmentation.scaling(x, sigma=0.1)
            x = augmentation.window_warp(x, window_ratio=0.1, scales=[0.75, 1.25])
            time_series = x.squeeze()

        return {
            'time_series':    torch.from_numpy(time_series),
            'point_dist':     torch.from_numpy(point_dist),
            'relative_error': torch.tensor(relative_error).reshape([1])
        }

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("SenseTime 데이터 로드 중...")
slam_data_st = DeepSEEData(
    root_dir=config.data_dir, dataset_list=["SenseTime"],
    n_proc=config.n_proc, config=config
)

print("EuRoC 데이터 로드 중...")
slam_data_euroc = DeepSEEData(
    root_dir=config.data_dir, dataset_list=["EuRoC"],
    n_proc=config.n_proc, config=config
)

slam_data = slam_data_st
slam_data.data["EuRoC"] = slam_data_euroc.data["EuRoC"]

# ── lower_bound, upper_bound, std 계산 (SenseTime unsup pretrain 데이터) ──────
print("데이터 통계 계산 중...")
unsup_data_config = {
    "SenseTime": ['A0','A1','A2','A3','A4','A5','A6','A7',
                  'B0','B1','B2','B3','B4','B5','B6','B7']
}

def assign_traj(slam_data, set_config):
    set_data = {}
    for dataset in set_config:
        set_data[dataset] = {}
        for traj in set_config[dataset]:
            if traj in slam_data.data[dataset]:
                set_data[dataset][traj] = slam_data.data[dataset][traj]
    return set_data

def MergeDict(dict1, dict2):
    return {**dict1, **dict2}

unsup_pretrain_data = assign_traj(slam_data, unsup_data_config)
unsup_set = SLAMDataset(unsup_pretrain_data, augment=False, onlyOneIter=True)

all_data = []
for i in tqdm(range(len(unsup_set)), desc="통계 계산"):
    d = unsup_set[i]['time_series']
    all_data.append(d.numpy() if isinstance(d, torch.Tensor) else d)
all_data = np.stack(all_data)

def remove_outliers_iqr(data):
    Q1 = np.percentile(data, 10, axis=0)
    Q3 = np.percentile(data, 90, axis=0)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

lower_bound, upper_bound = remove_outliers_iqr(all_data)
capped_data = np.clip(all_data, lower_bound, upper_bound)
std = np.std(capped_data, axis=0)
print("통계 계산 완료")

# ── TS2Vec pretrain_model 로드 ─────────────────────────────────────────────
print("TS2Vec 모델 로드 중...")
_sd = torch.load(f"{APR07_DIR}/pretrained_model.pkl", weights_only=False, map_location='cpu')
# input_dims: module.input_fc.weight shape = [output_dims, input_dims]
input_dims = _sd['module.input_fc.weight'].shape[1]
print(f"  input_dims={input_dims}")
pretrain_model = TS2Vec(input_dims=input_dims, device=0, batch_size=128, output_dims=64)
pretrain_model.net.load_state_dict(_sd)
print("TS2Vec 로드 완료")

# ── DeepSEE 모델 구조 정의 ────────────────────────────────────────────────────
config_PD = TimesformerConfig(
    image_size=128, patch_size=8, num_channels=3, num_frames=4,
    num_hidden_layers=3, num_attention_heads=8, intermediate_size=256,
    hidden_size=192
)
config_TS = PatchTSMixerConfig(
    context_length=30, prediction_length=1, num_input_channels=64,
    d_model=192, patch_len=10, patch_stride=5,
    use_positional_encoding=True, ca_d_model=128, num_layers=3, drop_out=0.0
)
config_CA = MultiModalCrossAttentionConfig(
    ts2vec_only=True, ts2vec_dim=64,
    ts_context_length=30, ts_patch_len=5, ts_num_input_channels=64,
    ts_patch_stride=5, ts_d_model=192, ts_time_step=33,
    pd_d_model=192, pd_time_step=330, pe_max_len=10000,
    ca_d_model=128, ca_num_head=16, ca_num_layers=2, ca_dropout=0.0,
    ca_time_series_only=False, output_range=None
)

model = DeepSEEModel(config_PD, config_TS, config_CA)

# Apr07 SupervisedPretrain 모델 로드
print("Apr07 SenseTime 모델 로드 중...")
# test_model.pth는 state_dict만 저장 (loss_fn 없음)
state_dict = torch.load(f"{APR07_DIR}/test_model.pth", weights_only=False)
model.load_state_dict(state_dict)
model.eval()
print("모델 로드 완료")

# ── loss_fn (노트북과 동일하게) ────────────────────────────────────────────────
class WeightedMSELoss(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(WeightedMSELoss, self).__init__()
        self.alpha = alpha
    def forward(self, predictions, targets):
        with torch.no_grad():
            weights = (1 + self.alpha * targets) ** 4
        loss = (predictions - targets) ** 2
        return torch.mean(weights * loss)

def loss_fn(outputs, targets, device):
    return MeanAbsolutePercentageError().to(device)(outputs, targets) * 2 + \
           torch.nn.MSELoss().to(device)(outputs, targets)

# ── RF baseline 관련 ──────────────────────────────────────────────────────────
def global_pooling_1D(features):
    return torch.mean(features, dim=0)

def ensemble_tree_data_loader(train_set, val_set, test_set):
    def load_data(ds):
        X, Y = [], []
        for i in range(len(ds)):
            item = ds[i]
            ts = item['time_series']
            if ts.dim() == 2:
                ts = ts[0]
            X.append(ts.numpy())
            Y.append(item['relative_error'].numpy().flatten()[0])
        return np.stack(X), np.array(Y)
    XT, YT = load_data(train_set)
    XV, YV = load_data(val_set)
    XTe, YTe = load_data(test_set)
    return XT, YT, XV, YV, XTe, YTe

# ── evaluate_on_test ──────────────────────────────────────────────────────────
def evaluate_on_test(model, test_loader):
    set_seed(config.seed)
    accelerator = Accelerator(mixed_precision="bf16")
    model, test_dl = accelerator.prepare(model, test_loader)
    model.eval()

    Y_gt, Y_est = [], []
    for batch in test_dl:
        time_series_np = batch['time_series'].cpu().numpy()
        time_series_enc = pretrain_model.encode(
            time_series_np, causal=True, sliding_length=1, sliding_padding=5
        )
        time_series = torch.from_numpy(time_series_enc).to(accelerator.device, dtype=torch.float)
        point_dist     = batch['point_dist'].to(accelerator.device, dtype=torch.float)
        relative_error = batch['relative_error'].to(accelerator.device, dtype=torch.float)

        with torch.no_grad():
            outputs = model(point_dist, time_series)
            all_out, all_gt = accelerator.gather_for_metrics((outputs, relative_error))

        Y_gt.append(relative_error.cpu())
        Y_est.append(outputs.cpu())

    return Y_gt, Y_est

# ── EuRoC fold config ─────────────────────────────────────────────────────────
sup_pretrain_train_data = assign_traj(slam_data, {
    "SenseTime": ['A0','A1','A2','A3']
})

euroc_finetune_config_list = [
    {"sup_finetune_train_traj": {"EuRoC": ["MH_02_easy","MH_03_medium","MH_04_difficult","MH_05_difficult"]},
     "sup_finetune_val_traj":   {"EuRoC": ["MH_01_easy"]},
     "sup_finetune_test_traj":  {"EuRoC": ["MH_01_easy"]}},
    {"sup_finetune_train_traj": {"EuRoC": ["MH_01_easy","MH_03_medium","MH_04_difficult","MH_05_difficult"]},
     "sup_finetune_val_traj":   {"EuRoC": ["MH_02_easy"]},
     "sup_finetune_test_traj":  {"EuRoC": ["MH_02_easy"]}},
    {"sup_finetune_train_traj": {"EuRoC": ["MH_01_easy","MH_02_easy","MH_04_difficult","MH_05_difficult"]},
     "sup_finetune_val_traj":   {"EuRoC": ["MH_03_medium"]},
     "sup_finetune_test_traj":  {"EuRoC": ["MH_03_medium"]}},
    {"sup_finetune_train_traj": {"EuRoC": ["MH_01_easy","MH_02_easy","MH_03_medium","MH_05_difficult"]},
     "sup_finetune_val_traj":   {"EuRoC": ["MH_04_difficult"]},
     "sup_finetune_test_traj":  {"EuRoC": ["MH_04_difficult"]}},
    {"sup_finetune_train_traj": {"EuRoC": ["MH_01_easy","MH_02_easy","MH_03_medium","MH_04_difficult"]},
     "sup_finetune_val_traj":   {"EuRoC": ["MH_05_difficult"]},
     "sup_finetune_test_traj":  {"EuRoC": ["MH_05_difficult"]}},
]

test_params  = {'batch_size': config.batchsize, 'shuffle': False, 'num_workers': config.n_proc}
train_params = {'batch_size': config.batchsize, 'shuffle': True,  'num_workers': config.n_proc}
val_params   = {'batch_size': config.batchsize, 'shuffle': False, 'num_workers': config.n_proc}

# ── 추론 ──────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EuRoC Zero-Shot 추론 시작 (Apr07 SenseTime 모델)")
print("="*60)

for idx, fc in enumerate(euroc_finetune_config_list):
    seq = list(fc["sup_finetune_test_traj"]["EuRoC"])[0]
    print(f"\n[Fold {idx}] test: {seq}")

    train_data = assign_traj(slam_data, fc["sup_finetune_train_traj"])
    val_data   = assign_traj(slam_data, fc["sup_finetune_val_traj"])
    test_data  = assign_traj(slam_data, fc["sup_finetune_test_traj"])

    train_set = SLAMDataset(train_data, augment=False, onlyOneIter=False,
                             lower=lower_bound, upper=upper_bound, std=std)
    val_set   = SLAMDataset(val_data,   augment=False, onlyOneIter=False,
                             lower=lower_bound, upper=upper_bound, std=std)
    test_set  = SLAMDataset(test_data,  augment=False, onlyOneIter=True,
                             lower=lower_bound, upper=upper_bound, std=std)

    test_loader = DataLoader(test_set, **test_params)

    # RF baseline
    train_data_bl = MergeDict(train_data, sup_pretrain_train_data)
    train_set_bl  = SLAMDataset(train_data_bl, augment=False, onlyOneIter=False,
                                 lower=lower_bound, upper=upper_bound, std=std)
    XT, YT, XV, YV, XTe, YTe = ensemble_tree_data_loader(train_set_bl, val_set, test_set)
    regr = RandomForestRegressor(n_estimators=25, max_features='sqrt', min_samples_split=2,
                                  min_samples_leaf=2, bootstrap=False, max_depth=10,
                                  random_state=0, n_jobs=-1)
    regr.fit(XT, YT)
    YT_base = regr.predict(XTe)

    # DeepSEE zero-shot 추론
    Y_gt, Y_est = evaluate_on_test(model, test_loader)

    np.save(f"{APR07_DIR}/SupervisedFinetune_{idx}_Y_gt.npy",   np.array(Y_gt,  dtype=object), allow_pickle=True)
    np.save(f"{APR07_DIR}/SupervisedFinetune_{idx}_Y_est.npy",  np.array(Y_est, dtype=object), allow_pickle=True)
    np.save(f"{APR07_DIR}/SupervisedFinetune_{idx}_Y_base.npy", YT_base, allow_pickle=True)
    print(f"  Saved fold {idx} → {seq}")

print("\n완료! hds_v1.py RUN_DIR을 Apr07_13-17-22_AHRI-Junhyun 으로 변경 후 실행하세요.")
