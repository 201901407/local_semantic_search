"""Paired bootstrap: is any variant's nDCG@10 difference vs fp32 real, or noise?"""
import json, numpy as np, os
paths=json.load(open("scifact_paths.json"))
qrels=[json.loads(l) for l in open(paths["qrels/test.jsonl"])]
rel={}
for r in qrels: rel.setdefault(r["query-id"],{})[r["corpus-id"]]=int(r["score"])
qids=json.load(open("qids.json"))
chunk_doc=json.load(open("chunks_meta.json"))["chunk_doc"]
docs=sorted(set(chunk_doc)); didx={d:i for i,d in enumerate(docs)}
cdoc=np.array([didx[d] for d in chunk_doc]); K=10

def per_query_ndcg(Q,D):
    S=Q@D.T
    out=np.full((S.shape[0],len(docs)),-1e9,dtype=np.float32)
    np.maximum.at(out.T,cdoc,S.T)
    top=np.argsort(-out,axis=1)[:,:K]; vals=[]
    for i,qid in enumerate(qids):
        g=rel.get(qid,{})
        gains=[g.get(docs[j],0) for j in top[i]]
        dcg=sum((2**gn-1)/np.log2(r+2) for r,gn in enumerate(gains))
        ideal=sorted(g.values(),reverse=True)[:K]
        idcg=sum((2**gn-1)/np.log2(r+2) for r,gn in enumerate(ideal))
        vals.append(dcg/idcg if idcg else 0.0)
    return np.array(vals)

def q8(M):
    s=np.abs(M).max()/127.0
    R=(np.round(M/s).astype(np.int8).astype(np.float32))*s
    return R/np.clip(np.linalg.norm(R,axis=1,keepdims=True),1e-12,None)

base=per_query_ndcg(np.load("qry_torch_fp32.npy"),np.load("doc_torch_fp32.npy"))
rng=np.random.default_rng(0); N=10000
print(f"baseline nDCG@10 = {base.mean():.4f}  (n={len(base)} queries)\n")
print(f"{'variant':32s} {'nDCG':>7s} {'delta':>8s} {'95% CI of delta':>22s} {'p':>7s}")
print("-"*82)
cands=[("model_quantized (int8 weights)","model_quantized",False),
       ("model_quantized + int8 vectors","model_quantized",True),
       ("model_fp16","model_fp16",False),
       ("model_uint8","model_uint8",False),
       ("model_q4","model_q4",False),
       ("fp32 + int8 vectors ONLY","torch_fp32",True)]
for label,key,qv in cands:
    if not os.path.exists(f"doc_{key}.npy"): continue
    D=np.load(f"doc_{key}.npy"); Q=np.load(f"qry_{key}.npy")
    if qv: D,Q=q8(D),q8(Q)
    v=per_query_ndcg(Q,D); d=v-base
    idx=rng.integers(0,len(d),(N,len(d)))
    boot=d[idx].mean(1)
    lo,hi=np.percentile(boot,[2.5,97.5])
    p=2*min((boot<=0).mean(),(boot>=0).mean())
    print(f"{label:32s} {v.mean():7.4f} {d.mean():+8.4f} [{lo:+.4f}, {hi:+.4f}] {p:7.3f}")
