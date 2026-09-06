import json, glob, os, numpy as np

paths=json.load(open("scifact_paths.json"))
qrels=[json.loads(l) for l in open(paths["qrels/test.jsonl"])]
rel={}
for r in qrels: rel.setdefault(r["query-id"],{})[r["corpus-id"]]=int(r["score"])
qids=json.load(open("qids.json"))
chunk_doc=json.load(open("chunks_meta.json"))["chunk_doc"]
docs=sorted(set(chunk_doc)); didx={d:i for i,d in enumerate(docs)}
cdoc=np.array([didx[d] for d in chunk_doc])
K=10

def doc_scores(Q,D):
    """chunk scores -> per-doc max-pool, the same 'best chunk wins' rule the
    project's own eval uses when it de-duplicates by file."""
    S=Q@D.T                                   # (nq, nchunks)
    out=np.full((S.shape[0],len(docs)),-1e9,dtype=np.float32)
    np.maximum.at(out.T,cdoc,S.T)
    return out

def metrics(out):
    nd=rc=mrr=0.0
    top=np.argsort(-out,axis=1)[:,:K]
    for i,qid in enumerate(qids):
        g=rel.get(qid,{}); 
        if not g: continue
        ranked=[docs[j] for j in top[i]]
        gains=[g.get(d,0) for d in ranked]
        dcg=sum((2**gn-1)/np.log2(r+2) for r,gn in enumerate(gains))
        ideal=sorted(g.values(),reverse=True)[:K]
        idcg=sum((2**gn-1)/np.log2(r+2) for r,gn in enumerate(ideal))
        nd+=dcg/idcg if idcg else 0
        rc+=sum(1 for d in ranked if g.get(d,0)>0)/len(g)
        rr=0
        for r,d in enumerate(ranked):
            if g.get(d,0)>0: rr=1/(r+1); break
        mrr+=rr
    n=len(qids)
    return nd/n, rc/n, mrr/n, top

def quant_int8(M):
    """Symmetric global scalar quantization of stored vectors to int8."""
    s=np.abs(M).max()/127.0
    return (np.round(M/s).astype(np.int8).astype(np.float32))*s

variants=[("torch_fp32","torch fp32 (baseline)")]+[
    (v,v) for v in ["model","model_fp16","model_quantized","model_uint8","model_q4"]
    if os.path.exists(f"doc_{v}.npy")]

base_top=None; rows=[]
for key,label in variants:
    D=np.load(f"doc_{key}.npy"); Q=np.load(f"qry_{key}.npy")
    for qv in [False,True]:
        Dv,Qv=(quant_int8(D),quant_int8(Q)) if qv else (D,Q)
        Dv=Dv/np.clip(np.linalg.norm(Dv,axis=1,keepdims=True),1e-12,None)
        Qv=Qv/np.clip(np.linalg.norm(Qv,axis=1,keepdims=True),1e-12,None)
        nd,rc,mrr,top=metrics(doc_scores(Qv,Dv))
        if base_top is None: base_top=top
        t1=np.mean(top[:,0]==base_top[:,0])
        ov=np.mean([len(set(a)&set(b))/K for a,b in zip(top,base_top)])
        rows.append((label+(" + int8 vectors" if qv else ""),nd,rc,mrr,t1,ov))

print(f"{'variant':34s} {'nDCG@10':>8s} {'Rec@10':>7s} {'MRR@10':>7s} {'top1agr':>8s} {'top10ovl':>9s}")
print("-"*80)
b=rows[0][1]
for label,nd,rc,mrr,t1,ov in rows:
    d=(nd-b)/b*100
    print(f"{label:34s} {nd:8.4f} {rc:7.4f} {mrr:7.4f} {t1:8.1%} {ov:9.1%}   {d:+.2f}%")
