"""
배포 모델(May18=ba8cbeb)로 zero-shot 일반화 전체 재추론 — 진짜 모델.
EuRoC + TUM-VI + OpenLORIS. 모델 설정은 hds_g1_local(검증됨)과 동일.
결과 est/gt 를 results/deployed_<ds>.npz 로 저장 → 재사용/그림용.

실행: conda run -n deepsee --no-capture-output python3 deployed_generalization.py
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np, torch

sys.path.insert(0, "/home/junhyun/SEESys/DeepSEE/Training")
from deepSEEConfig import DeepSEEConfig
from ts2vec.ts2vec import TS2Vec
from datasets.data import DeepSEEData
from models.DeepSEEModels import (TimesformerConfig, PatchTSMixerConfig,
                                  MultiModalCrossAttentionConfig, DeepSEEModel)

DEPLOY  = "/home/junhyun/HDS_repo/DeepSEE/runs"
MODEL   = "/home/junhyun/SEESys/DeepSEE/Training/runs/May18_14-44-57_AHRI-Junhyun/SupervisedFinetune_1_best_model.pth"
OUTDIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DATASETS = ["EuRoC", "TUMVI", "OpenLORIS"]
TAU = 4.6
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTDIR, exist_ok=True)


def auc_roc(label, score):
    label = np.asarray(label); score = np.asarray(score, float)
    P, N = label.sum(), (1 - label).sum()
    if P == 0 or N == 0: return float("nan")
    o = np.argsort(-score, kind="mergesort"); tp = fp = a = ptp = pfp = 0.0; prev = None
    for i in o:
        if score[i] != prev: a += (fp - pfp) * (tp + ptp) / 2; ptp, pfp, prev = tp, fp, score[i]
        tp += label[i] == 1; fp += label[i] == 0
    a += (fp - pfp) * (tp + ptp) / 2; return a / (P * N)


def build_model():
    pd_c = TimesformerConfig(image_size=128, patch_size=8, num_channels=3, num_frames=4,
                             num_hidden_layers=3, num_attention_heads=12, hidden_size=192,
                             intermediate_size=256, hidden_dropout_prob=0)
    ts_c = PatchTSMixerConfig(context_length=30, patch_len=1, num_input_channels=6, d_model=64)
    ca_c = MultiModalCrossAttentionConfig(ts2vec_only=True, ts2vec_dim=64, ca_d_model=128,
                                          ca_num_head=16, ca_num_layers=2, reg_d_fc=128,
                                          ts_num_input_channels=64, ts_d_model=192, ts_time_step=33,
                                          pd_width=96, pd_height=128, pd_d_model=192,
                                          pd_time_step=330, ts_context_length=30, pe_max_len=10000)
    m = DeepSEEModel(pd_c, ts_c, ca_c)
    import __main__
    if not hasattr(__main__, "loss_fn"):
        __main__.loss_fn = lambda pred, target: pred
    ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    miss, unexp = m.load_state_dict(sd, strict=False)
    crit = [k for k in miss if k.startswith(("ca_regressor", "pd_encoder"))]
    print(f"  모델 로드: missing={len(miss)} unexpected={len(unexp)} critical={len(crit)}")
    assert not crit, "핵심 키 누락 — 설정 불일치"
    return m.eval().to(dev)


lower = np.load(f"{DEPLOY}/rts_norm_lower.npy")
upper = np.load(f"{DEPLOY}/rts_norm_upper.npy")
std   = np.load(f"{DEPLOY}/rts_norm_std.npy")
print("[TS2Vec 로드]")
ts2 = TS2Vec(input_dims=16, output_dims=64, device=0, batch_size=128)
_st = torch.load(f"{DEPLOY}/pretrained_model.pkl", map_location="cpu", weights_only=False)
ts2.net.load_state_dict({k.replace("module.", ""): v for k, v in _st.items()}, strict=False)
ts2.net.eval()
print("[모델 로드]"); model = build_model()


def infer_traj(td):
    est, gt = [], []
    for it in td:
        for tk in td[it]:
            d = td[it][tk]
            ts = d["timeSeries"].values.astype(np.float32)
            ts = np.clip(ts, lower, upper) / (std + 1e-8)
            pd_np = DeepSEEData.load_point_dist_from_path(d["pointDist_path"]).astype(np.float32)
            enc = ts2.encode(ts[None], causal=True, sliding_length=1, sliding_padding=5)[0]
            with torch.no_grad():
                out = model(torch.tensor(pd_np[None], dtype=torch.float32, device=dev),
                            torch.tensor(enc[None], dtype=torch.float32, device=dev))
            v = float(out.item())
            if np.isfinite(v):
                est.append(v); gt.append(float(np.asarray(d["relativeError"]).flatten()[0]))
    return np.array(est), np.array(gt)


cfg = DeepSEEConfig(seed=2024, n_proc=8, window_step=5,
                    data_dir="/home/junhyun/SEESys/DeepSEE/Training/datasets",
                    datasets=['EuRoC'], target_transform=True)

summary = {}
for ds in DATASETS:
    print(f"\n===== {ds} 로드+추론 (배포 모델) =====")
    try:
        data = DeepSEEData(root_dir=cfg.data_dir, dataset_list=[ds],
                           n_proc=cfg.n_proc, config=cfg).data[ds]
    except Exception as e:
        print(f"  [로드 실패] {e}"); continue
    allest, allgt, corrs = [], [], []
    for traj in sorted(data.keys()):
        est, gt = infer_traj(data[traj])
        if len(est) < 5:
            continue
        r = np.corrcoef(est, gt)[0, 1]; corrs.append(r)
        lab = (gt > TAU).astype(int)
        a = auc_roc(lab, est) if 0 < lab.sum() < len(lab) else float("nan")
        print(f"  {traj:18s} n={len(est):5d} corr={r:+.3f} AUC={a:.3f} ev={int(lab.sum())}")
        allest.append(est); allgt.append(gt)
    if not allest:
        print("  (유효 traj 없음)"); continue
    E = np.concatenate(allest); G = np.concatenate(allgt); L = (G > TAU).astype(int)
    np.savez(f"{OUTDIR}/deployed_{ds}.npz", est=E, gt=G)
    summary[ds] = {"n": int(len(E)), "events": int(L.sum()),
                   "corr_mean": round(float(np.mean(corrs)), 3),
                   "AUC": round(float(auc_roc(L, E)), 3)}
    print(f"  >>> {ds}: corr평균={summary[ds]['corr_mean']:+.3f}  AUC={summary[ds]['AUC']:.3f}  (ev {L.sum()}/{len(E)})")

print("\n" + "=" * 64)
print("배포 모델(May18) zero-shot 일반화 — 진짜 모델")
print("=" * 64)
print(f"{'dataset':12s} {'corr':>8s} {'AUC':>8s} {'n':>7s} {'events':>7s}")
print(f"{'SenseTime*':12s} {'+0.38':>8s} {'0.80':>8s} {'-':>7s} {'-':>7s}   (*in-domain)")
for ds, s in summary.items():
    print(f"{ds:12s} {s['corr_mean']:+8.3f} {s['AUC']:8.3f} {s['n']:7d} {s['events']:7d}")
print("=" * 64)
json.dump(summary, open(f"{OUTDIR}/deployed_generalization.json", "w"), indent=2)
print("저장:", OUTDIR)
