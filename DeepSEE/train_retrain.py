"""
Train DeepSEE with paper-identical conditions (SenseTime 6-fold cross-validation)
- Self-supervised pretrain: TS2Vec (n_iters=600) + PSD SSL (30 epochs)
- Supervised pretrain: 20 epochs (paper uses synthetic data; we use SenseTime real data)
- Supervised finetune: 10 epochs × 6 folds
- Saves SupervisedFinetune_{i}_Y_*.npy for HDS evaluation
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
import sys
import time
import copy
import random
import numpy as np
import torch
import torch.utils.tensorboard as tb
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(2024)
torch.manual_seed(2024)

sys.path.insert(0, '/home/junhyun/SEESys/DeepSEE/Training')
from deepSEEConfig import DeepSEEConfig
from ts2vec.ts2vec import TS2Vec
from datasets.data import DeepSEEData
from datasets import augmentation
from models.DeepSEEModels import (
    TimesformerConfig, TimesformerModel, PatchTSMixerConfig,
    MultiModalCrossAttentionConfig, PointDistProjection, DeepSEEModel
)
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from transformers import get_scheduler
from torchmetrics.regression import MeanAbsolutePercentageError
from sklearn.ensemble import RandomForestRegressor

# ── Config ────────────────────────────────────────────────────────────────────
config = DeepSEEConfig(
    seed=2024, n_proc=1, window_step=5, lr=1e-6,
    epochs=30,          # supervised pretrain epochs
    batchsize=32,       # reduced for RAM stability with full dataset
    data_dir="/home/junhyun/final_DeepSEE/DeepSEE_SLAM/train_data",
    datasets=['SenseTime', 'LivingRoom', 'Hall', 'Lab', 'Lab2', 'Apartment',
              'FireStationOffice', 'FireStationKitchen', 'FireStationGarage',
              'AbandonedFactory'],
    target_transform=True,
)

writer = SummaryWriter()
OUT_DIR = writer.get_logdir()
print(f"Output dir: {OUT_DIR}")

# ── Data Loading ──────────────────────────────────────────────────────────────
print("Loading SenseTime data...")
slam_data = DeepSEEData(
    root_dir=config.data_dir,
    dataset_list=config.datasets,
    n_proc=config.n_proc,
    config=config,
)

def assign_traj(slam_data, set_config):
    set_data = {}
    for dataset in set_config:
        set_data[dataset] = {}
        for traj in set_config[dataset]:
            try:
                set_data[dataset][traj] = slam_data.data[dataset][traj]
            except:
                print(f"  [WARN] traj not found: {traj} in {dataset}")
    return set_data

_syn_trajs = ['A0','A1','A2','A3','A4','A5','A6','A7','B0','B1','B2','B3','B4','B5','B6','B7']
_syn_trajs_s = ['A0','A1','A2','A3','A4','A5','A6','A7']  # Lab/Lab2/LivingRoom/Hall (A only)

split_config = {
    "unsup_pretrain_traj": {
        "Apartment":          _syn_trajs,
        "FireStationOffice":  _syn_trajs,
        "FireStationKitchen": _syn_trajs,
        "FireStationGarage":  _syn_trajs,
        "AbandonedFactory":   _syn_trajs,
        "Lab":                _syn_trajs_s,
        "Lab2":               _syn_trajs_s,
        "LivingRoom":         _syn_trajs_s,
        "Hall":               _syn_trajs_s,
        "SenseTime": ['A1','A2','A3','A4','A5','A6','A7','B0','B1','B2','B3','B4','B5','B6','B7'],
    },
    "sup_pretrain_train_traj": {
        "FireStationOffice":  _syn_trajs,
        "FireStationKitchen": _syn_trajs,
        "FireStationGarage":  _syn_trajs,
        "AbandonedFactory":   _syn_trajs,
        "Lab":                _syn_trajs_s,
        "Lab2":               _syn_trajs_s,
        "Hall":               _syn_trajs_s,
        "LivingRoom":         _syn_trajs_s,
    },
    "sup_pretrain_val_traj": {
        "Apartment": _syn_trajs,
    },
    "sup_pretrain_test_traj": {
        "SenseTime": ['A1','A2','A3','A4','A5','A6','A7','B0','B1','B2','B3','B4','B5','B6','B7'],
    },
}

unsup_pretrain_data     = assign_traj(slam_data, split_config["unsup_pretrain_traj"])
sup_pretrain_train_data = assign_traj(slam_data, split_config["sup_pretrain_train_traj"])
sup_pretrain_val_data   = assign_traj(slam_data, split_config["sup_pretrain_val_traj"])
sup_pretrain_test_data  = assign_traj(slam_data, split_config["sup_pretrain_test_traj"])

# ── Dataset ───────────────────────────────────────────────────────────────────
class SLAMDataset(Dataset):
    def __init__(self, slam_data, augment=False, onlyOneIter=False, lower=None, upper=None, std=None):
        self.slam_data    = slam_data
        self.onlyOneIter  = onlyOneIter
        self.augment      = augment
        self.lower        = lower
        self.upper        = upper
        self.std          = std
        self.generate_sample_list(slam_data)

    def generate_sample_list(self, slam_data):
        self.sample_list = []
        for dataset in slam_data:
            for traj in slam_data[dataset]:
                for m, iter_ in enumerate(slam_data[dataset][traj]):
                    for ts in slam_data[dataset][traj][iter_]:
                        if self.onlyOneIter and m != 0:
                            continue
                        self.sample_list.append([dataset, traj, iter_, ts])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        dataset, traj, iter_, ts = self.sample_list[idx]
        d = self.slam_data[dataset][traj][iter_][ts]
        time_series    = d['timeSeries'].values
        point_dist     = DeepSEEData.load_point_dist_from_path(d['pointDist_path'])
        relative_error = d['relativeError']

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
            'time_series':   torch.from_numpy(time_series),
            'point_dist':    torch.from_numpy(point_dist),
            'relative_error': torch.tensor(relative_error).reshape([1]),
        }

# ── Normalization bounds ───────────────────────────────────────────────────────
print("Computing normalization bounds...")
tmp_set = SLAMDataset(unsup_pretrain_data, augment=False)
all_data = np.stack([tmp_set[i]['time_series'][0].numpy() for i in range(len(tmp_set))])

def remove_outliers_iqr(data):
    Q1 = np.percentile(data, 10, axis=0)
    Q3 = np.percentile(data, 90, axis=0)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

lower_bound, upper_bound = remove_outliers_iqr(all_data)
capped = np.clip(all_data, lower_bound, upper_bound)
std = np.std(capped, axis=0)

unsup_pretrain_set       = SLAMDataset(unsup_pretrain_data,     augment=False, lower=lower_bound, upper=upper_bound, std=std)
sup_pretrain_train_set   = SLAMDataset(sup_pretrain_train_data, augment=False, lower=lower_bound, upper=upper_bound, std=std)
sup_pretrain_val_set     = SLAMDataset(sup_pretrain_val_data,   augment=False, onlyOneIter=True, lower=lower_bound, upper=upper_bound, std=std)
sup_pretrain_test_set    = SLAMDataset(sup_pretrain_test_data,  augment=False, onlyOneIter=True, lower=lower_bound, upper=upper_bound, std=std)

# ── TS2Vec Self-supervised Pretrain ───────────────────────────────────────────
print("Building TS2Vec pretrain dataset...")
pretrain_dataset = []
for i in range(len(unsup_pretrain_set)):
    pretrain_dataset.append(unsup_pretrain_set[i]['time_series'].unsqueeze(0).numpy())
random.shuffle(pretrain_dataset)
pretrain_dataset = np.concatenate(pretrain_dataset, axis=0)

print(f"TS2Vec fitting on {pretrain_dataset.shape} ...")
pretrain_model = TS2Vec(input_dims=pretrain_dataset.shape[-1], device=0, batch_size=128, output_dims=64)
loss_log = pretrain_model.fit(pretrain_dataset, n_iters=600, verbose=True)
pretrain_model.save(OUT_DIR + "/pretrained_model.pkl")
print("TS2Vec done.")

# ── PSD SSL Pretrain ──────────────────────────────────────────────────────────
config_PD = TimesformerConfig(
    image_size=128, patch_size=8, num_channels=3, num_frames=4,
    num_hidden_layers=3, num_attention_heads=12,
    hidden_size=192, intermediate_size=256, hidden_dropout_prob=0,
)

class PSDSSL(torch.nn.Module):
    def __init__(self, pd_config):
        super().__init__()
        self.pd_config  = pd_config
        self.pd_encoder = TimesformerModel(pd_config)
        self.fc1 = torch.nn.Linear(192, 192)
        self.fc2 = torch.nn.Linear(192, 192)

    def forward(self, point_dist):
        batch_size = point_dist.shape[0]
        binary_mask = torch.ones((batch_size, 4, 3, 12, 16), dtype=torch.float).to(point_dist.device)
        mi  = torch.randint(0, 4, (batch_size, 12, 16))
        mi1 = torch.randint(0, 4, (batch_size, 12, 16))
        for i in range(batch_size):
            for j in range(12):
                for k in range(16):
                    binary_mask[i, mi[i,j,k],  :, j, k] = 0
                    binary_mask[i, mi1[i,j,k], :, j, k] = 0
        binary_mask = torch.nn.functional.interpolate(binary_mask, scale_factor=(1,8,8), mode="nearest")
        masked = point_dist * binary_mask
        pd_hs = self.pd_encoder(masked).last_hidden_state[:, 1:, :]
        pd_hs = pd_hs.view(batch_size, 12, 16, 4, 192).reshape(batch_size, 192, 4, 192)
        pd_hs = pd_hs.permute(0, 2, 1, 3).reshape(batch_size, 4, 192, 192)
        pd_hs = self.fc1(pd_hs)
        pd_hs = torch.nn.functional.gelu(pd_hs)
        pd_hs = self.fc2(pd_hs)
        pd_hs = pd_hs.view(batch_size, 4, 12, 16, 3, 8, 8)
        pd_hs = pd_hs.permute(0, 1, 4, 2, 5, 3, 6)
        return pd_hs.reshape(batch_size, 4, 3, 96, 128)

psd_ssl_model = PSDSSL(config_PD)

def PSD_Pretrain(model, loader):
    set_seed(config.seed)
    acc   = Accelerator(mixed_precision="bf16")
    opt   = AdamW(model.parameters(), lr=1e-3)
    n_ep  = 30
    steps = n_ep * len(loader)
    sched = get_scheduler("constant_with_warmup", optimizer=opt, num_warmup_steps=int(0.1*steps), num_training_steps=steps)
    loader, model, opt, sched = acc.prepare(loader, model, opt, sched)
    losses = []
    for ep in range(n_ep):
        epoch_loss = []
        model.train()
        for batch in loader:
            pd = batch['point_dist'].to(acc.device, dtype=torch.float)
            out = model(pd)
            loss = torch.nn.MSELoss()(out, pd)
            epoch_loss.append(loss.item())
            acc.backward(loss)
            opt.step(); sched.step(); opt.zero_grad()
        losses.append(np.mean(epoch_loss))
        print(f"  PSD pretrain epoch {ep}: loss={losses[-1]:.6f}")
    return model, losses

pretrain_params = {'batch_size': config.batchsize, 'shuffle': True, 'num_workers': 0}
pretrain_loader = DataLoader(unsup_pretrain_set, **pretrain_params)
print("PSD SSL pretraining (30 epochs)...")
psd_ssl_model, _ = PSD_Pretrain(psd_ssl_model, pretrain_loader)
torch.save(psd_ssl_model.pd_encoder.state_dict(), OUT_DIR + "/psd_ssl_model.pth")
print("PSD SSL done.")

# ── DeepSEE Model ─────────────────────────────────────────────────────────────
config_TS = PatchTSMixerConfig(
    context_length=30, prediction_length=1, num_input_channels=64,
    d_model=192, patch_len=10, patch_stride=5,
    use_positional_encoding=True, ca_d_model=128,
    num_layers=3, drop_out=0.0,
)
config_CA = MultiModalCrossAttentionConfig(
    ts2vec_only=True, ts2vec_dim=64,
    ts_context_length=30, ts_patch_len=5, ts_num_input_channels=64,
    ts_patch_stride=5, ts_d_model=192, ts_time_step=33,
    pd_d_model=192, pd_time_step=33*10, pe_max_len=10000,
    ca_d_model=128, ca_num_head=16, ca_num_layers=2,
    ca_dropout=0.0, ca_time_series_only=False, output_range=None,
)
model = DeepSEEModel(config_PD, config_TS, config_CA)
model.pd_encoder.load_state_dict(torch.load(OUT_DIR + "/psd_ssl_model.pth", weights_only=False))
print("DeepSEE model created.")

# ── Loss / Training utils ─────────────────────────────────────────────────────
def loss_fn(outputs, targets, device):
    return (MeanAbsolutePercentageError().to(device)(outputs, targets) * 2
            + torch.nn.MSELoss().to(device)(outputs, targets))

class SaveBestModel:
    def __init__(self):
        self.best_valid_loss = float('inf')

    def __call__(self, val_loss, epoch, model, optimizer, criterion, folder, fname='best_model'):
        if epoch >= 1 and val_loss < self.best_valid_loss:
            self.best_valid_loss = val_loss
            print(f"  Saving best model at epoch {epoch+1} (val_loss={val_loss:.6f})")
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': criterion},
                       f'{folder}/{fname}.pth')

def train_with_validation(model, train_loader, val_loader, test_loader, cfg, mode="SupervisedPretrain"):
    set_seed(cfg.seed)
    acc   = Accelerator(mixed_precision="bf16")
    opt   = AdamW(model.parameters(), lr=cfg.lr)
    steps = cfg.epochs * len(train_loader)
    sched = get_scheduler("constant_with_warmup", optimizer=opt,
                          num_warmup_steps=int(0.1*steps), num_training_steps=steps)
    tr_dl, va_dl, te_dl, model, opt, sched = acc.prepare(
        train_loader, val_loader, test_loader, model, opt, sched)
    model = model.to(acc.device)

    for param in model.pd_encoder.parameters():
        param.requires_grad = False

    save_best = SaveBestModel()

    for epoch in range(cfg.epochs):
        model.train()
        tr_losses = []
        for batch in tr_dl:
            ts = pretrain_model.encode(batch['time_series'].cpu().numpy(), causal=True, sliding_length=1, sliding_padding=5)
            ts = torch.from_numpy(ts).to(acc.device, dtype=torch.float)
            pd = batch['point_dist'].to(acc.device, dtype=torch.float)
            re = batch['relative_error'].to(acc.device, dtype=torch.float)
            out  = model(pd, ts)
            loss = loss_fn(out, re, acc.device)
            tr_losses.append(loss.item())
            acc.backward(loss)
            opt.step(); sched.step(); opt.zero_grad()
        writer.add_scalar(f'{mode} epoch loss', np.mean(tr_losses), epoch)

        model.eval()
        va_losses = []
        with torch.no_grad():
            for batch in va_dl:
                ts = pretrain_model.encode(batch['time_series'].cpu().numpy(), causal=True, sliding_length=1, sliding_padding=5)
                ts = torch.from_numpy(ts).to(acc.device, dtype=torch.float)
                pd = batch['point_dist'].to(acc.device, dtype=torch.float)
                re = batch['relative_error'].to(acc.device, dtype=torch.float)
                out = model(pd, ts)
                all_out, all_re = acc.gather_for_metrics((out, re))
                va_losses.append(loss_fn(all_out, all_re, acc.device).item())
        val_loss = np.mean(va_losses)
        save_best(val_loss, epoch, model, opt, loss_fn, OUT_DIR, fname=f'{mode}_best_model')
        print(f"  [{mode}] epoch {epoch+1}/{cfg.epochs}  train={np.mean(tr_losses):.6f}  val={val_loss:.6f}")

    return model

def evaluate_on_test(model, test_loader, mode="SupervisePretrain"):
    set_seed(config.seed)
    acc = Accelerator(mixed_precision="bf16")
    model, te_dl = acc.prepare(model, test_loader)
    model.eval()
    Y_gt, Y_est = [], []
    with torch.no_grad():
        for batch in te_dl:
            ts = pretrain_model.encode(batch['time_series'].cpu().numpy(), causal=True, sliding_length=1, sliding_padding=5)
            ts = torch.from_numpy(ts).to(acc.device, dtype=torch.float)
            pd = batch['point_dist'].to(acc.device, dtype=torch.float)
            re = batch['relative_error'].to(acc.device, dtype=torch.float)
            out = model(pd, ts)
            Y_gt.append(re); Y_est.append(out)
    return Y_gt, Y_est

# ── Baseline (Random Forest) utils ────────────────────────────────────────────
indices_to_remove = [4,5,6,7,8,15]
all_indices = torch.arange(16)
mask = ~torch.tensor([i in indices_to_remove for i in all_indices])

def load_data_from_dataset(dataset):
    X, Y = [], []
    for idx in range(len(dataset)):
        X.append(torch.mean(dataset[idx]['time_series'], dim=0).unsqueeze(0))
        Y.append(dataset[idx]['relative_error'].unsqueeze(0))
    X = torch.cat(X)[:, mask]
    Y = torch.cat(Y)
    return X, Y

# ── Supervised Pretrain (20 epochs) ──────────────────────────────────────────
train_params = {'batch_size': config.batchsize, 'shuffle': True,  'num_workers': 0}
val_params   = {'batch_size': config.batchsize, 'shuffle': False, 'num_workers': 0}

train_loader = DataLoader(sup_pretrain_train_set, **train_params)
val_loader   = DataLoader(sup_pretrain_val_set,   **val_params)
test_loader  = DataLoader(sup_pretrain_test_set,  **val_params)

print(f"\n=== Supervised Pretrain ({config.epochs} epochs) ===")
model = train_with_validation(model, train_loader, val_loader, test_loader, config, mode="SupervisedPretrain")
torch.save(model.state_dict(), OUT_DIR + "/test_model.pth")
print("Supervised pretrain done.\n")

# ── 6-fold Cross-Validation Finetune ─────────────────────────────────────────
finetune_split_config_list = [
    {'sup_finetune_train_traj': {'SenseTime': ['A7','A3','A6','B3','B4','B7','B0','B2']},
     'sup_finetune_val_traj':   {'SenseTime': ['A4','B6','B1']},
     'sup_finetune_test_traj':  {'SenseTime': ['A2','B5','A5']}},
    {'sup_finetune_train_traj': {'SenseTime': ['A2','A3','A6','B3','B4','B5','A5','B0']},
     'sup_finetune_val_traj':   {'SenseTime': ['A7','B7','B1']},
     'sup_finetune_test_traj':  {'SenseTime': ['A4','B6','B2']}},
    {'sup_finetune_train_traj': {'SenseTime': ['A2','A6','B3','B4','B6','B7','A5','B2']},
     'sup_finetune_val_traj':   {'SenseTime': ['A7','B5','B0']},
     'sup_finetune_test_traj':  {'SenseTime': ['A4','A3','B1']}},
    {'sup_finetune_train_traj': {'SenseTime': ['A7','A6','B4','B5','B6','B7','A5','B1']},
     'sup_finetune_val_traj':   {'SenseTime': ['A2','A3','B2']},
     'sup_finetune_test_traj':  {'SenseTime': ['A4','B3','B0']}},
    {'sup_finetune_train_traj': {'SenseTime': ['A4','A3','A6','B4','B5','B6','A5','B1']},
     'sup_finetune_val_traj':   {'SenseTime': ['A2','B7','B2']},
     'sup_finetune_test_traj':  {'SenseTime': ['A7','B3','B0']}},
    {'sup_finetune_train_traj': {'SenseTime': ['A4','A6','B3','B4','B5','B6','A5','B0']},
     'sup_finetune_val_traj':   {'SenseTime': ['A2','B7','B1']},
     'sup_finetune_test_traj':  {'SenseTime': ['A7','A3','B2']}},
]

def MergeDict(d1, d2):
    return {**d1, **d2}

all_gt_list, all_est_list, all_base_list = [], [], []

for idx, fold_cfg in enumerate(finetune_split_config_list):
    print(f"\n=== Fold {idx} | test: {fold_cfg['sup_finetune_test_traj']['SenseTime']} ===")

    ft_train = assign_traj(slam_data, fold_cfg["sup_finetune_train_traj"])
    ft_val   = assign_traj(slam_data, fold_cfg["sup_finetune_val_traj"])
    ft_test  = assign_traj(slam_data, fold_cfg["sup_finetune_test_traj"])

    ft_train_set = SLAMDataset(ft_train, augment=False, onlyOneIter=False, lower=lower_bound, upper=upper_bound, std=std)
    ft_val_set   = SLAMDataset(ft_val,   augment=False, onlyOneIter=False, lower=lower_bound, upper=upper_bound, std=std)
    ft_test_set  = SLAMDataset(ft_test,  augment=False, onlyOneIter=True,  lower=lower_bound, upper=upper_bound, std=std)

    ft_train_loader = DataLoader(ft_train_set, **train_params)
    ft_val_loader   = DataLoader(ft_val_set,   **val_params)
    ft_test_loader  = DataLoader(ft_test_set,  **val_params)

    # Baseline (Random Forest)
    baseline_train_data = MergeDict(ft_train, sup_pretrain_train_data)
    baseline_train_set  = SLAMDataset(baseline_train_data, augment=False, onlyOneIter=False, lower=lower_bound, upper=upper_bound, std=std)
    XT_train, YT_train = load_data_from_dataset(baseline_train_set)
    XT_val,   YT_val   = load_data_from_dataset(ft_val_set)
    XT_test,  YT_test  = load_data_from_dataset(ft_test_set)
    regr = RandomForestRegressor(n_estimators=25, max_features='sqrt',
                                 min_samples_split=2, min_samples_leaf=2,
                                 bootstrap=False, max_depth=10, random_state=0, n_jobs=-1)
    regr.fit(XT_train.numpy(), YT_train.numpy().ravel())
    YT_test_pred = regr.predict(XT_test.numpy())

    # Load best supervised pretrain checkpoint
    best_cp = torch.load(f'{OUT_DIR}/SupervisedPretrain_best_model.pth', weights_only=False)
    model.load_state_dict(best_cp['model_state_dict'])

    # Finetune config
    ft_config = copy.deepcopy(config)
    ft_config.epochs = 20

    # Freeze all, unfreeze fc1/fc2
    for param in model.parameters():
        param.requires_grad = False
    for param in model.ca_regressor.fc1.parameters():
        param.requires_grad = True
    for param in model.ca_regressor.fc2.parameters():
        param.requires_grad = True

    model = train_with_validation(model, ft_train_loader, ft_val_loader, ft_test_loader,
                                  ft_config, mode=f"SupervisedFinetune_{idx}")

    # Load best finetune checkpoint
    best_ft_cp = torch.load(f'{OUT_DIR}/SupervisedFinetune_{idx}_best_model.pth', weights_only=False)
    model.load_state_dict(best_ft_cp['model_state_dict'])
    print(f"  Loaded best finetune model (epoch {best_ft_cp['epoch']})")

    Y_gt, Y_est = evaluate_on_test(model, ft_test_loader, mode=f"SupervisedFinetune_{idx}")

    # Save npy
    Y_gt_cat  = torch.cat(Y_gt,  dim=0).cpu().numpy()
    Y_est_cat = torch.cat(Y_est, dim=0).cpu().numpy()
    np.save(f'{OUT_DIR}/SupervisedFinetune_{idx}_Y_gt.npy',   [Y_gt_cat],    allow_pickle=True)
    np.save(f'{OUT_DIR}/SupervisedFinetune_{idx}_Y_est.npy',  [Y_est_cat],   allow_pickle=True)
    np.save(f'{OUT_DIR}/SupervisedFinetune_{idx}_Y_base.npy', YT_test_pred,  allow_pickle=True)
    print(f"  Saved npy for fold {idx}")

    all_gt_list.append(Y_gt_cat)
    all_est_list.append(Y_est_cat)
    all_base_list.append(np.expand_dims(YT_test_pred, 1))

# ── Final RMSE / MAPE ─────────────────────────────────────────────────────────
from sklearn.metrics import mean_absolute_percentage_error

def inv(x):
    return (np.exp(x) - 1) / 10000

print("\n" + "="*60)
print("Final Results (논문 형식: RMSE cm, MAPE)")
print("="*60)

all_rmse_ds, all_mape_ds = [], []
all_rmse_base, all_mape_base = [], []

for i in range(len(all_gt_list)):
    g  = inv(all_gt_list[i])
    e  = inv(all_est_list[i])
    b  = inv(all_base_list[i])
    rmse_ds   = float(np.sqrt(np.mean((g - e)**2)))
    rmse_base = float(np.sqrt(np.mean((g - b)**2)))
    mape_ds   = float(mean_absolute_percentage_error(g + 1e-9, e + 1e-9))
    mape_base = float(mean_absolute_percentage_error(g + 1e-9, b + 1e-9))
    all_rmse_ds.append(rmse_ds); all_mape_ds.append(mape_ds)
    all_rmse_base.append(rmse_base); all_mape_base.append(mape_base)
    print(f"Fold {i}:  DeepSEE RMSE={rmse_ds:.4f}  MAPE={mape_ds:.4f}  |  Base RMSE={rmse_base:.4f}  MAPE={mape_base:.4f}")

all_g = inv(np.concatenate(all_gt_list))
all_e = inv(np.concatenate(all_est_list))
all_b = inv(np.concatenate(all_base_list))
print(f"\nAll DeepSEE: RMSE={np.sqrt(np.mean((all_g-all_e)**2)):.4f}  MAPE={mean_absolute_percentage_error(all_g+1e-9, all_e+1e-9):.4f}")
print(f"All Base:    RMSE={np.sqrt(np.mean((all_g-all_b)**2)):.4f}  MAPE={mean_absolute_percentage_error(all_g+1e-9, all_b+1e-9):.4f}")
print(f"\n논문 DeepSEE: RMSE=0.235 cm  MAPE=0.507")
print(f"Output dir: {OUT_DIR}")
