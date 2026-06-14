"""
배포 모델(May18 = ba8cbeb)로 EuRoC zero-shot 직접 재추론 — 진짜 모델 평가.
모델 설정은 hds_g1_local.py(검증됨)와 동일. 데이터는 DeepSEEData(EuRoC).
TS2Vec/norm은 배포 컴포넌트(HDS_repo/runs).

실행: conda run -n deepsee --no-capture-output python3 euroc_deployed_eval.py
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
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def auc_roc(label, score):
    label = np.asarray(label); score = np.asarray(score, float)
    P, N = label.sum(), (1 - label).sum()
    if P == 0 or N == 0: return float("nan")
    o = np.argsort(-score, kind="mergesort"); tp = fp = a = ptp = pfp = 0.0; prev = None
    for i in o:
        if score[i] != prev: a += (fp - pfp) * (tp + ptp) / 2; ptp, pfp, prev = tp, fp, score[i]
        tp += label[i] == 1; fp += label[i] == 0
    a += (fp - pfp) * (tp + ptp) / 2; return a / (P * N)


config = DeepSEEConfig(seed=2024, n_proc=8, window_step=5,
                       data_dir="/home/junhyun/SEESys/DeepSEE/Training/datasets",
                       datasets=['EuRoC'], target_transform=True)

print("[1] EuRoC 데이터 로드(DeepSEEData)...")
euroc = DeepSEEData(root_dir=config.data_dir, dataset_list=["EuRoC"],
                    n_proc=config.n_proc, config=config).data["EuRoC"]
print("  trajs:", list(euroc.keys()))

# 배포 정규화 통계
lower = np.load(f"{DEPLOY}/rts_norm_lower.npy")
upper = np.load(f"{DEPLOY}/rts_norm_upper.npy")
std   = np.load(f"{DEPLOY}/rts_norm_std.npy")

print("[2] TS2Vec(배포 pkl) 로드...")
ts2 = TS2Vec(input_dims=16, output_dims=64, device=0, batch_size=128)
_st = torch.load(f"{DEPLOY}/pretrained_model.pkl", map_location="cpu", weights_only=False)
ts2.net.load_state_dict({k.replace("module.", ""): v for k, v in _st.items()}, strict=False)
ts2.net.eval()

print("[3] DeepSEE 모델(May18=배포) 로드 — hds_g1_local 설정...")
pd_config = TimesformerConfig(image_size=128, patch_size=8, num_channels=3, num_frames=4,
                              num_hidden_layers=3, num_attention_heads=12, hidden_size=192,
                              intermediate_size=256, hidden_dropout_prob=0)
ts_config = PatchTSMixerConfig(context_length=30, patch_len=1, num_input_channels=6, d_model=64)
ca_config = MultiModalCrossAttentionConfig(ts2vec_only=True, ts2vec_dim=64, ca_d_model=128,
                                           ca_num_head=16, ca_num_layers=2, reg_d_fc=128,
                                           ts_num_input_channels=64, ts_d_model=192, ts_time_step=33,
                                           pd_width=96, pd_height=128, pd_d_model=192,
                                           pd_time_step=330, ts_context_length=30, pe_max_len=10000)
model = DeepSEEModel(pd_config, ts_config, ca_config)
import __main__
if not hasattr(__main__, "loss_fn"):
    __main__.loss_fn = lambda pred, target: pred
ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
missing, unexpected = model.load_state_dict(sd, strict=False)
crit = [k for k in missing if k.startswith(("ca_regressor", "pd_encoder"))]
print(f"  로드: missing={len(missing)} unexpected={len(unexpected)} critical_missing={len(crit)}")
if crit:
    print("  [경고] 핵심 키 누락 — 설정 불일치!"); print("  예:", crit[:3])
model.eval().to(dev)


def infer_traj(traj_data):
    est, gt = [], []
    for it in traj_data:
        for ts_key in traj_data[it]:
            d = traj_data[it][ts_key]
            ts = d["timeSeries"].values.astype(np.float32)          # (30,16) raw
            ts = np.clip(ts, lower, upper) / (std + 1e-8)           # 배포 정규화
            pd_np = DeepSEEData.load_point_dist_from_path(d["pointDist_path"]).astype(np.float32)
            enc = ts2.encode(ts[None], causal=True, sliding_length=1, sliding_padding=5)[0]  # (30,64)
            with torch.no_grad():
                out = model(torch.tensor(pd_np[None], dtype=torch.float32, device=dev),
                            torch.tensor(enc[None], dtype=torch.float32, device=dev))
            v = float(out.item())
            if np.isfinite(v):
                est.append(v); gt.append(float(np.asarray(d["relativeError"]).flatten()[0]))
    return np.array(est), np.array(gt)


print("\n[4] EuRoC 추론 (배포 모델)...")
allab, allsc, corrs = [], [], []
for traj in sorted(euroc.keys()):
    est, gt = infer_traj(euroc[traj])
    if len(est) < 5: print(f"  {traj}: n={len(est)} (스킵)"); continue
    r = np.corrcoef(est, gt)[0, 1]; corrs.append(r)
    lab = (gt > 4.6).astype(int)
    a = auc_roc(lab, est) if 0 < lab.sum() < len(lab) else float("nan")
    print(f"  {traj}: n={len(est):4d} corr={r:+.3f} AUC@4.6={a:.3f} ev={int(lab.sum())} "
          f"est[{est.min():.2f},{est.max():.2f}] gt[{gt.min():.2f},{gt.max():.2f}]")
    allab.append(lab); allsc.append(est)

lab = np.concatenate(allab); sc = np.concatenate(allsc)
print("\n" + "=" * 62)
print(f"배포 모델(May18) EuRoC zero-shot:")
print(f"  corr(est,gt) 평균 = {np.mean(corrs):+.3f}")
print(f"  AUC@4.6 (전체)    = {auc_roc(lab, sc):.3f}   (이벤트 {int(lab.sum())}/{lab.size})")
print("=" * 62)
print("  비교: 옛 May10 모델 EuRoC = corr -0.25~-0.43 / AUC ~0.13-0.49")
print("        in-domain(May18)    = corr +0.38 / AUC 0.80")
