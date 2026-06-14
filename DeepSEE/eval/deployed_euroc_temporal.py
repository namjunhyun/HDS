"""
배포 모델(May18)로 EuRoC MH_01~05 시퀀스별 *시계열* est 추출 (타임스탬프 포함).
한 iteration(가장 표본 많은)만 → 깨끗한 temporal 시퀀스.
→ results/deployed_euroc_temporal.npz : {seq: (ts[], est[], gt[])}
VLM 재비교(#1)와 EWR 재산출(#2)의 공통 토대.

실행: conda run -n deepsee --no-capture-output python3 deployed_euroc_temporal.py
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np, torch
sys.path.insert(0, "/home/junhyun/SEESys/DeepSEE/Training")
from deepSEEConfig import DeepSEEConfig
from ts2vec.ts2vec import TS2Vec
from datasets.data import DeepSEEData
from models.DeepSEEModels import (TimesformerConfig, PatchTSMixerConfig,
                                  MultiModalCrossAttentionConfig, DeepSEEModel)

DEPLOY = "/home/junhyun/HDS_repo/DeepSEE/runs"
MODEL  = "/home/junhyun/SEESys/DeepSEE/Training/runs/May18_14-44-57_AHRI-Junhyun/SupervisedFinetune_1_best_model.pth"
OUT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "deployed_euroc_temporal.npz")
SEQS   = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lower = np.load(f"{DEPLOY}/rts_norm_lower.npy"); upper = np.load(f"{DEPLOY}/rts_norm_upper.npy"); std = np.load(f"{DEPLOY}/rts_norm_std.npy")
ts2 = TS2Vec(input_dims=16, output_dims=64, device=0, batch_size=128)
_st = torch.load(f"{DEPLOY}/pretrained_model.pkl", map_location="cpu", weights_only=False)
ts2.net.load_state_dict({k.replace("module.", ""): v for k, v in _st.items()}, strict=False); ts2.net.eval()

pd_c = TimesformerConfig(image_size=128, patch_size=8, num_channels=3, num_frames=4, num_hidden_layers=3,
                         num_attention_heads=12, hidden_size=192, intermediate_size=256, hidden_dropout_prob=0)
ts_c = PatchTSMixerConfig(context_length=30, patch_len=1, num_input_channels=6, d_model=64)
ca_c = MultiModalCrossAttentionConfig(ts2vec_only=True, ts2vec_dim=64, ca_d_model=128, ca_num_head=16,
                                      ca_num_layers=2, reg_d_fc=128, ts_num_input_channels=64, ts_d_model=192,
                                      ts_time_step=33, pd_width=96, pd_height=128, pd_d_model=192,
                                      pd_time_step=330, ts_context_length=30, pe_max_len=10000)
model = DeepSEEModel(pd_c, ts_c, ca_c)
import __main__
if not hasattr(__main__, "loss_fn"): __main__.loss_fn = lambda p, t: p
ck = torch.load(MODEL, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
miss, un = model.load_state_dict(sd, strict=False)
assert not [k for k in miss if k.startswith(("ca_regressor", "pd_encoder"))], "핵심키 누락"
model.eval().to(dev)

cfg = DeepSEEConfig(seed=2024, n_proc=8, window_step=5,
                    data_dir="/home/junhyun/SEESys/DeepSEE/Training/datasets",
                    datasets=['EuRoC'], target_transform=True)
print("[EuRoC 로드]")
euroc = DeepSEEData(root_dir=cfg.data_dir, dataset_list=["EuRoC"], n_proc=cfg.n_proc, config=cfg).data["EuRoC"]

out = {}
for seq in SEQS:
    td = euroc[seq]
    best_it = max(td.keys(), key=lambda it: len(td[it]))   # 표본 가장 많은 iteration
    keys = sorted(td[best_it].keys(), key=lambda k: float(k))  # 타임스탬프 순
    ts_arr, est_arr, gt_arr = [], [], []
    for tk in keys:
        d = td[best_it][tk]
        ts = np.clip(d["timeSeries"].values.astype(np.float32), lower, upper) / (std + 1e-8)
        pd_np = DeepSEEData.load_point_dist_from_path(d["pointDist_path"]).astype(np.float32)
        enc = ts2.encode(ts[None], causal=True, sliding_length=1, sliding_padding=5)[0]
        with torch.no_grad():
            o = model(torch.tensor(pd_np[None], dtype=torch.float32, device=dev),
                      torch.tensor(enc[None], dtype=torch.float32, device=dev))
        v = float(o.item())
        if np.isfinite(v):
            ts_arr.append(float(tk)); est_arr.append(v); gt_arr.append(float(np.asarray(d["relativeError"]).flatten()[0]))
    out[seq + "_ts"] = np.array(ts_arr); out[seq + "_est"] = np.array(est_arr); out[seq + "_gt"] = np.array(gt_arr)
    print(f"  {seq}: iter={best_it} n={len(est_arr)} est[{min(est_arr):.2f},{max(est_arr):.2f}] gt[{min(gt_arr):.2f},{max(gt_arr):.2f}]")

np.savez(OUT, **out)
print("저장:", OUT)
