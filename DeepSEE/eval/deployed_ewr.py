"""
#2 EWR/precision 재산출 — 배포 모델(May18) est로. eval_offline의 정정판.
실제 타임스탬프로 lead-time 계산. DS/DS+G/DS+Sym/Full ablation.

토대: results/deployed_euroc_temporal.npz (배포 시계열 est/gt/ts)
G: features.csv(타임스탬프 정렬). Symbolic: annotations_euroc_scored.json(인덱스 재매핑).

사용: python3 deployed_ewr.py
"""
import os, json
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMP  = os.path.join(_HERE, "results", "deployed_euroc_temporal.npz")
EA    = "/home/junhyun/euroc_error_analysis/data"
SCORED = os.path.join(_HERE, "annotations_euroc_scored.json")
OUT   = os.path.join(_HERE, "deployed_ewr_results.json")

SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
OLD_NEST = {"MH_01_easy":114,"MH_02_easy":92,"MH_03_medium":80,"MH_04_difficult":58,"MH_05_difficult":65}
TAU = 4.6; THR = 0.85; H_SEC = 8.0; WARMUP = 8; W_HW = 0.3; LEADS = [1,3,5]
_W=[0.335,0.305,0.262]; _S=sum(_W); WE,WB,WL=[w/_S for w in _W]
VARIANTS=["DS","DS+G","DS+Sym","Full"]


def causal_norm(x, warm=WARMUP):
    x=np.asarray(x,float);n=len(x);o=np.zeros(n)
    for k in range(n):
        if k<warm: continue
        p=x[:k+1];mn,mx=p.min(),p.max()
        o[k]=0.0 if mx-mn<1e-9 else np.clip((x[k]-mn)/(mx-mn),0,1)
    return o

def auc_roc(label,score):
    label=np.asarray(label);score=np.asarray(score,float)
    P,N=label.sum(),(1-label).sum()
    if P==0 or N==0: return float('nan')
    o=np.argsort(-score,kind='mergesort');tp=fp=a=ptp=pfp=0.0;prev=None
    for i in o:
        if score[i]!=prev: a+=(fp-pfp)*(tp+ptp)/2;ptp,pfp,prev=tp,fp,score[i]
        tp+=label[i]==1;fp+=label[i]==0
    a+=(fp-pfp)*(tp+ptp)/2;return a/(P*N)

def guardrail(ts, seq):
    df=pd.read_csv(f"{EA}/{seq}/features.csv")
    fts=df["TimeStamp"].to_numpy(float)
    def col(c): return df[c].to_numpy(float) if c in df else np.zeros(len(df))
    B,E,L,K=col("Brightness"),col("Entropy"),col("Laplacian"),col("NumberKeyPoints")
    G=[]
    for t in ts:
        j=int(np.argmin(np.abs(fts-t)))
        b=B[j]/255.0; e=E[j]/8.0; l=min(L[j]/1000.0,1.0); f=1.0-min(K[j]/300.0,1.0)
        G.append(np.clip(WE*(1-e)+WB*(1-min(b,1))+WL*(1-l)+0.1*f,0,1))
    return np.array(G)

def sym_series(seq, ds, n):
    old=OLD_NEST[seq]; sym=np.zeros(n)
    for a in scored.get(seq,[]):
        s=int(a["seg_start"]*n/old); e=min(int(a["seg_end"]*n/old),n-1)
        for k in range(max(0,s),e+1):
            sym[k]+=min(max(0.0,0.65-ds[k]),a["risk_delta"])
    return sym


def main():
    Z=np.load(TEMP)
    acc={v:{"leads":[],"tp":0,"na":0,"alert_t":0.0,"tot_t":0.0,"lab":[],"sco":[]} for v in VARIANTS}
    for seq in SEQS:
        ts=Z[seq+"_ts"];est=Z[seq+"_est"];gt=Z[seq+"_gt"];n=len(est)
        ds=causal_norm(est);G=guardrail(ts,seq);hw=np.maximum(0,G-0.2)*W_HW
        sym=sym_series(seq,ds,n)
        scores={"DS":ds,"DS+G":np.clip(ds+hw,0,1),"DS+Sym":np.clip(ds+sym,0,1),"Full":np.clip(ds+sym+hw,0,1)}
        ev=[i for i in range(1,n) if gt[i]>TAU and gt[i-1]<=TAU]
        lab=(gt>TAU).astype(int)
        for v in VARIANTS:
            s=scores[v];al=s>THR
            for e in ev:
                lo=e
                while lo>0 and ts[e]-ts[lo-1]<=H_SEC: lo-=1
                hit=next((k for k in range(lo,e+1) if al[k]),None)
                acc[v]["leads"].append(ts[e]-ts[hit] if hit is not None else None)
            ao=[k for k in range(n) if al[k] and (k==0 or not al[k-1])]
            for a0 in ao:
                hi=a0
                while hi<n-1 and ts[hi+1]-ts[a0]<=H_SEC: hi+=1
                acc[v]["tp"]+=any(lab[k]==1 for k in range(a0,hi+1));acc[v]["na"]+=1
            dt=np.diff(ts,prepend=ts[0])
            acc[v]["alert_t"]+=float(dt[al].sum());acc[v]["tot_t"]+=float(ts[-1]-ts[0])
            acc[v]["lab"].append(lab);acc[v]["sco"].append(s)

    print("="*84)
    print(f"배포 모델(May18) EuRoC EWR ablation  (τ={TAU} THR={THR} H={H_SEC:.0f}s, 실시간 타임스탬프)")
    print("="*84)
    hdr=f"{'Variant':<8}{'EWR@1':>7}{'EWR@3':>7}{'EWR@5':>7}{'Prec':>7}{'FA/min':>8}{'mLead':>7}{'Duty%':>7}{'AUC':>7}"
    print(hdr);print("-"*len(hdr))
    res={}
    for v in VARIANTS:
        a=acc[v];leads=a["leads"];det=[x for x in leads if x is not None]
        ewr={t:100.0*sum(1 for x in det if x>=t)/len(leads) if leads else 0 for t in LEADS}
        prec=100.0*a["tp"]/a["na"] if a["na"] else 0
        fa=60.0*(a["na"]-a["tp"])/a["tot_t"] if a["tot_t"] else 0
        duty=100.0*a["alert_t"]/a["tot_t"] if a["tot_t"] else 0
        lab=np.concatenate(a["lab"]);sco=np.concatenate(a["sco"]);roc=auc_roc(lab,sco)
        ml=np.mean(det) if det else 0
        print(f"{v:<8}{ewr[1]:>6.1f}%{ewr[3]:>6.1f}%{ewr[5]:>6.1f}%{prec:>6.1f}%{fa:>8.2f}{ml:>6.1f}s{duty:>6.1f}%{roc:>7.3f}")
        res[v]={"EWR":{str(t):round(ewr[t],1) for t in LEADS},"precision":round(prec,1),
                "fa_per_min":round(fa,2),"mean_lead":round(float(ml),2),"duty":round(duty,1),"AUC":round(float(roc),3)}
    print("="*84)
    json.dump(res,open(OUT,"w"),ensure_ascii=False,indent=2);print("저장:",OUT)


if __name__=="__main__":
    with open(SCORED) as f: scored=json.load(f)
    main()
