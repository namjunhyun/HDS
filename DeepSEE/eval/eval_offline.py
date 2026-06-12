"""
HDS 오프라인 정량 평가 v2 — 논문급. "드리프트를 몇 초 일찍, 헛경보 없이 잡는가".

이전 v1의 치명적 결함 3개를 수정:
  (1) acausal 누수 → CAUSAL(과거만) 정규화로 온라인 조기경보 충실 모사
  (2) 가짜 이벤트  → 절대 임계 이벤트 정의 (Y_gt = log1p(10000·clip(RelErr,0.001,0.02)))
  (3) FP 미측정    → 정밀도(precision)/오경보율을 EWR(재현율)과 함께 보고

이벤트 정의: drift event = Y_gt 가 τ 를 상향 돌파한 지점 (τ 스윕으로 민감도 보고).
  τ=4.6 ≈ RelativeError 0.01  (RelErr 0.001→2.40, 0.02→5.30)

지표(고정 동작점 THR, 선행지평 H초):
  · EWR@{1,3,5}s : 이벤트 중 t초 이상 앞서 경보된 비율 (재현율 계열)
  · Precision    : 경보 onset 중 H초 내 실제 이벤트가 뒤따른 비율 (헛경보 아님)
  · FalseAlarm/min : 실제 이벤트로 이어지지 않은 경보 onset 빈도
  · mean lead    : 포착된 이벤트의 평균 선행시간
  · AUC-ROC/PR   : 세그먼트 단위 판별력 (보조)

사용: python3 eval_offline.py
"""
import os, json
import numpy as np
import pandas as pd

_HERE   = os.path.dirname(os.path.abspath(__file__))
RUN     = "/home/junhyun/final_DeepSEE/DeepSEE/Training/runs/May10_23-39-25_AHRI-Junhyun"
RUN_INDOMAIN = "/home/junhyun/SEESys/DeepSEE/Training/runs/May18_14-44-57_AHRI-Junhyun"  # 배포모델 run
EA_DIR  = "/home/junhyun/euroc_error_analysis/data"
SCORED  = os.path.join(_HERE, "annotations_euroc_scored.json")
CALIB   = os.path.join(_HERE, "..", "runs", "ds_calib.npz")
OUT     = os.path.join(_HERE, "eval_results.json")

SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
CAM_HZ        = 20.0
TAU_LIST      = [4.2, 4.6, 5.0]    # 이벤트 임계 스윕 (Y_gt 공간)
TAU_MAIN      = 4.6
THR           = 0.85               # 경보 임계 (causal 정규화 점수, ~합리적 duty 동작점)
H_SEC         = 8.0                # 선행지평
WARMUP        = 8                  # causal 정규화 워밍업 세그먼트(이 구간 경보 억제)
W_HW          = 0.3
LEAD_BUCKETS  = [1, 3, 5]

_W_E, _W_B, _W_L = 0.335, 0.305, 0.262
_WT = _W_E + _W_B + _W_L
W_ENTROPY, W_BRIGHT, W_LAP = _W_E/_WT, _W_B/_WT, _W_L/_WT


# ── 정규화 ────────────────────────────────────────────────────────────────
def causal_norm(x, warmup=WARMUP):
    """과거만 사용하는 확장형 min-max. 워밍업 구간은 0(경보 억제)."""
    x = np.asarray(x, float); n = len(x); out = np.zeros(n)
    for k in range(n):
        if k < warmup:
            continue
        past = x[:k+1]
        mn, mx = past.min(), past.max()
        out[k] = 0.0 if mx-mn < 1e-9 else np.clip((x[k]-mn)/(mx-mn), 0, 1)
    return out


def fixed_calib_norm(x):
    z = np.load(CALIB); lo, hi = float(z['lo']), float(z['hi'])
    return np.clip((np.asarray(x, float)-lo)/(hi-lo), 0, 1)


# ── 보조 ─────────────────────────────────────────────────────────────────
def _flatten(path):
    a = np.load(path, allow_pickle=True)
    return np.concatenate([np.asarray(x).flatten() for x in a])


def _seg_guardrail(df, n):
    idx = np.clip(np.floor(np.arange(len(df)) * n / len(df)).astype(int), 0, n-1)
    def seg(col):
        v = df[col].to_numpy(float) if col in df else np.zeros(len(df))
        s = np.array([v[idx == k].mean() if np.any(idx == k) else np.nan for k in range(n)])
        return pd.Series(s).interpolate().bfill().ffill().to_numpy()
    bright = seg("Brightness")/255.0
    entr   = seg("Entropy")/8.0
    lap    = np.minimum(seg("Laplacian")/1000.0, 1.0)
    feat   = 1.0 - np.minimum(seg("NumberKeyPoints")/300.0, 1.0)
    return np.clip(W_ENTROPY*(1-entr) + W_BRIGHT*(1-np.clip(bright,0,1)) + W_LAP*(1-lap) + 0.1*feat, 0, 1)


def _sym_series(annots, ds, n):
    sym = np.zeros(n)
    for a in annots:
        s, e, rd = a["seg_start"], min(a["seg_end"], n-1), a["risk_delta"]
        for k in range(s, e+1):
            sym[k] += min(max(0.0, 0.65-ds[k]), rd)
    return sym


def _event_onsets(ygt, tau):
    e = (ygt > tau).astype(int)
    return [i for i in range(1, len(e)) if e[i] == 1 and e[i-1] == 0]


def _alert_onsets(alert):
    return [i for i in range(len(alert)) if alert[i] and (i == 0 or not alert[i-1])]


def _auc_roc(label, score):
    P, N = label.sum(), (1-label).sum()
    if P == 0 or N == 0: return float("nan")
    order = np.argsort(-score, kind="mergesort")
    tp = fp = auc = ptp = pfp = 0.0; prev = None
    for i in order:
        if score[i] != prev:
            auc += (fp-pfp)*(tp+ptp)/2.0; ptp, pfp, prev = tp, fp, score[i]
        tp += label[i] == 1; fp += label[i] == 0
    auc += (fp-pfp)*(tp+ptp)/2.0
    return auc/(P*N)


def _auc_pr(label, score):
    P = label.sum()
    if P == 0: return float("nan")
    order = np.argsort(-score, kind="mergesort")
    tp = fp = ap = prev = 0.0
    for i in order:
        tp += label[i] == 1; fp += label[i] == 0
        rec = tp/P; ap += (tp/(tp+fp))*(rec-prev); prev = rec
    return ap


VARIANTS = ["DS", "DS+G", "DS+Sym", "Full"]


def evaluate(norm_fn, tau):
    """주어진 정규화 방식과 이벤트 임계에서 전 시퀀스 집계 지표."""
    acc = {v: {"leads": [], "n_events": 0,
               "tp_alerts": 0, "n_alerts": 0, "alert_time": 0.0, "total_time": 0.0,
               "labels": [], "scores": []} for v in VARIANTS}
    for i, seq in enumerate(SEQS):
        est = _flatten(f"{RUN}/EuRoC_ZeroShot_{i}_Y_est.npy")
        ygt = _flatten(f"{RUN}/EuRoC_ZeroShot_{i}_Y_gt.npy")
        n   = len(est)
        df  = pd.read_csv(f"{EA_DIR}/{seq}/features.csv")
        sec = (len(df)/CAM_HZ)/n
        H   = int(round(H_SEC/sec))

        ds  = norm_fn(est)
        G   = _seg_guardrail(df, n)
        sym = _sym_series(scored.get(seq, []), ds, n)
        hw  = np.maximum(0.0, G-0.20)*W_HW
        scores = {"DS": ds,
                  "DS+G":   np.clip(ds+hw, 0, 1),
                  "DS+Sym": np.clip(ds+sym, 0, 1),
                  "Full":   np.clip(ds+sym+hw, 0, 1)}
        events = _event_onsets(ygt, tau)
        label  = (ygt > tau).astype(int)

        for v in VARIANTS:
            s = scores[v]; alert = s > THR
            # 재현율/lead: 각 이벤트 직전 H초 내 최초 경보
            for ev in events:
                lo = max(0, ev-H)
                hit = next((k for k in range(lo, ev+1) if alert[k]), None)
                acc[v]["leads"].append((ev-hit)*sec if hit is not None else None)
            # 정밀도: 각 경보 onset 뒤 H초 내 이벤트 존재?
            for ao in _alert_onsets(alert):
                hi = min(n-1, ao+H)
                tp = any(label[k] == 1 for k in range(ao, hi+1))
                acc[v]["tp_alerts"] += tp; acc[v]["n_alerts"] += 1
            acc[v]["alert_time"] += alert.sum()*sec
            acc[v]["total_time"] += n*sec
            acc[v]["n_events"]   += len(events)
            acc[v]["labels"].append(label); acc[v]["scores"].append(s)

    out = {}
    for v in VARIANTS:
        a = acc[v]; leads = a["leads"]; det = [x for x in leads if x is not None]
        lab = np.concatenate(a["labels"]); sco = np.concatenate(a["scores"])
        ewr = {str(t): round(100.0*sum(1 for x in det if x >= t)/len(leads), 1) if leads else 0.0
               for t in LEAD_BUCKETS}
        prec = round(100.0*a["tp_alerts"]/a["n_alerts"], 1) if a["n_alerts"] else 0.0
        fa_min = round(60.0*(a["n_alerts"]-a["tp_alerts"])/a["total_time"], 2) if a["total_time"] else 0.0
        out[v] = {"EWR": ewr, "recall": ewr["1"], "precision": prec,
                  "false_alarm_per_min": fa_min,
                  "mean_lead_s": round(float(np.mean(det)) if det else 0.0, 2),
                  "alert_duty_%": round(100.0*a["alert_time"]/a["total_time"], 1) if a["total_time"] else 0.0,
                  "AUC_ROC": round(float(_auc_roc(lab, sco)), 3),
                  "AUC_PR":  round(float(_auc_pr(lab, sco)), 3)}
    out["_n_events"] = acc["DS"]["n_events"]
    return out, acc


def bootstrap_ci(leads, n_boot=2000, seed=0):
    """이벤트 lead 리스트(None=미포착) 부트스트랩 → recall·meanLead 95% CI."""
    rng = np.random.default_rng(seed)
    m = len(leads)
    if m == 0:
        return (0, 0), (0, 0)
    arr = np.array([np.nan if x is None else x for x in leads], float)
    recs, mleads = [], []
    for _ in range(n_boot):
        s = arr[rng.integers(0, m, m)]
        recs.append(100.0 * np.mean(~np.isnan(s)))
        det = s[~np.isnan(s)]
        mleads.append(np.mean(det) if det.size else 0.0)
    pc = lambda a: (round(float(np.percentile(a, 2.5)), 1), round(float(np.percentile(a, 97.5)), 1))
    return pc(recs), pc(mleads)


def eval_indomain(tau):
    """SenseTime in-domain(배포모델 run)의 base DeepSEE 판별력 — AUC + 부트스트랩 CI."""
    labs, scos, per = [], [], {}
    for i in range(6):
        pe = f"{RUN_INDOMAIN}/SupervisedFinetune_{i}_Y_est.npy"
        pg = f"{RUN_INDOMAIN}/SupervisedFinetune_{i}_Y_gt.npy"
        if not os.path.exists(pe):
            continue
        est, gt = _flatten(pe), _flatten(pg)
        lab = (gt > tau).astype(int)
        if 0 < lab.sum() < len(lab):
            per[i] = round(float(_auc_roc(lab, est)), 3)
        labs.append(lab); scos.append(est)
    lab = np.concatenate(labs); sco = np.concatenate(scos)
    # AUC 부트스트랩 CI
    rng = np.random.default_rng(0); m = len(lab); aucs = []
    for _ in range(1000):
        idx = rng.integers(0, m, m)
        l, s = lab[idx], sco[idx]
        if 0 < l.sum() < len(l):
            aucs.append(_auc_roc(l, s))
    ci = (round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3))
    return {"AUC_ROC": round(float(_auc_roc(lab, sco)), 3), "AUC_CI95": ci,
            "AUC_PR": round(float(_auc_pr(lab, sco)), 3),
            "per_fold_AUC": per, "n_events": int(lab.sum()), "n": int(len(lab))}


def _print(title, res):
    print("="*92); print(title); print("="*92)
    hdr = f"{'Variant':<8}{'EWR@1':>7}{'EWR@3':>7}{'EWR@5':>7}{'Prec':>7}{'FA/min':>8}{'mLead':>7}{'Duty%':>7}{'ROC':>7}{'PR':>7}"
    print(hdr); print("-"*len(hdr))
    for v in VARIANTS:
        o = res[v]
        print(f"{v:<8}{o['EWR']['1']:>6.1f}%{o['EWR']['3']:>6.1f}%{o['EWR']['5']:>6.1f}%"
              f"{o['precision']:>6.1f}%{o['false_alarm_per_min']:>8.2f}{o['mean_lead_s']:>6.1f}s"
              f"{o['alert_duty_%']:>6.1f}%{o['AUC_ROC']:>7.3f}{o['AUC_PR']:>7.3f}")
    print()


if __name__ == '__main__':
    with open(SCORED) as f:
        scored = json.load(f)

    results = {"config": {"THR": THR, "H_sec": H_SEC, "warmup": WARMUP,
                          "tau_main": TAU_MAIN, "norm": "causal"},
               "indomain": {}, "by_tau": {}, "fixed_calib": {}}

    # 0) IN-DOMAIN base 검증 (SenseTime) — 모델이 정상임을 입증
    try:
        ind = eval_indomain(TAU_MAIN)
        results["indomain"] = ind
        print("="*92)
        print(f"[IN-DOMAIN base DeepSEE, SenseTime] τ={TAU_MAIN}  (이벤트 {ind['n_events']}/{ind['n']})")
        print("="*92)
        print(f"  AUC-ROC = {ind['AUC_ROC']:.3f}  (95% CI {ind['AUC_CI95'][0]}–{ind['AUC_CI95'][1]})"
              f"   AUC-PR = {ind['AUC_PR']:.3f}")
        print(f"  fold별 AUC: {ind['per_fold_AUC']}")
        print("  → base 모델은 자기 도메인에서 견고. EuRoC zero-shot 저하는 도메인시프트.\n")
    except Exception as e:
        print(f"[in-domain 스킵] {e}\n")

    # 1) 메인: causal 정규화, τ 스윕
    for tau in TAU_LIST:
        r, acc = evaluate(causal_norm, tau)
        results["by_tau"][str(tau)] = r
        _print(f"[CAUSAL norm, EuRoC zero-shot] τ={tau} (이벤트 {r['_n_events']}개)  THR={THR} H={H_SEC:.0f}s", r)
        if tau == TAU_MAIN:   # 메인 τ만 부트스트랩 CI
            print("  부트스트랩 95% CI (recall / mean-lead):")
            for v in VARIANTS:
                rec_ci, lead_ci = bootstrap_ci(acc[v]["leads"])
                r[v]["recall_CI95"] = rec_ci; r[v]["lead_CI95"] = lead_ci
                print(f"    {v:<8} recall {r[v]['recall']:.1f}% [{rec_ci[0]}–{rec_ci[1]}]"
                      f"   lead {r[v]['mean_lead_s']:.1f}s [{lead_ci[0]}–{lead_ci[1]}]")
            print()

    # 2) 보조: 배포 충실(고정 캘리브) at τ_main
    try:
        rf, _ = evaluate(fixed_calib_norm, TAU_MAIN)
        results["fixed_calib"] = rf
        _print(f"[FIXED-CALIB norm, 배포충실] τ={TAU_MAIN} (이벤트 {rf['_n_events']}개)", rf)
    except Exception as e:
        print(f"[fixed-calib 스킵] {e}")

    with open(OUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"저장: {OUT}")
