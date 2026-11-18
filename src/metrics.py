"""Development diagnostics. These are NOT the unavailable official evaluator.

EMD uses equal-mass assignment on a capped sample, MMD is unbiased and may
be negative, and FPD uses coordinate moments rather than learned features.
Use the shipped official local_eval.py for competition numbers.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist
from scipy.optimize import linear_sum_assignment
from .common import load_points, crop, save_json, TRACKS, audit_real_manifest

def sample(p, n, rng): return p[rng.choice(len(p),min(n,len(p)),replace=False),:3].astype(np.float64)

def realism(p, q, n=2048, emd_n=512, seed=1234):
    rng = np.random.default_rng(seed); p=sample(p,n,rng); q=sample(q,n,rng)
    if min(len(p),len(q)) < 2: raise ValueError('At least two points per cloud are required')
    cd = np.mean(cKDTree(q).query(p)[0]**2) + np.mean(cKDTree(p).query(q)[0]**2)
    joint = np.concatenate([p,q]); bandwidth = np.median(pdist(sample(joint,1000,rng), metric='sqeuclidean'))
    gamma = 1 / max(2 * bandwidth, 1e-12)
    a=np.exp(-gamma*cdist(p,p,'sqeuclidean')); b=np.exp(-gamma*cdist(q,q,'sqeuclidean'))
    mmd=(a.sum()-np.trace(a))/(len(p)*(len(p)-1)) + (b.sum()-np.trace(b))/(len(q)*(len(q)-1)) - 2*np.exp(-gamma*cdist(p,q,'sqeuclidean')).mean()
    k=min(emd_n,len(p),len(q)); cost=cdist(sample(p,k,rng),sample(q,k,rng))
    rows, cols=linear_sum_assignment(cost); emd=cost[rows,cols].mean()
    cp,cq=np.cov(p.T),np.cov(q.T)
    ev,vec=np.linalg.eigh(cp); sqrtp=(vec*np.sqrt(np.maximum(ev,0)))@vec.T
    evmid=np.linalg.eigvalsh(sqrtp@cq@sqrtp)
    fpd=np.sum((p.mean(0)-q.mean(0))**2)+np.trace(cp)+np.trace(cq)-2*np.sqrt(np.maximum(evmid,0)).sum()
    return {'CD':float(cd),'MMD2':float(mmd),'EMD_W1_proxy':float(emd),'FPD':float(max(fpd,0)),
            'official':False,'points':[len(p),len(q)],'emd_points':k,'seed':seed}

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('diagnostic'); d.add_argument('--synthetic',type=Path,required=True)
    d.add_argument('--real-manifest',type=Path,required=True);d.add_argument('--forbidden',type=Path,required=True)
    d.add_argument('--track',choices=TRACKS,required=True);d.add_argument('--output',type=Path,required=True);d.add_argument('--channels',type=int,default=4)
    o=sub.add_parser('official');o.add_argument('--kit',type=Path,required=True);o.add_argument('args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.command=='official':
        script=a.kit/'local_eval.py'
        if not script.exists(): raise SystemExit('Official scorer unavailable. Obtain the approved starting kit; no substitute is reported as exact.')
        subprocess.run([sys.executable,str(script),*a.args],check=True,cwd=a.kit)
    else:
        rows=audit_real_manifest(a.real_manifest,a.forbidden)
        real=np.concatenate([crop(load_points(r['path'],r.get('channels',4)),TRACKS[a.track]['range']) for r in rows])
        save_json(a.output,realism(crop(load_points(a.synthetic,a.channels),TRACKS[a.track]['range']),real))

if __name__=='__main__': main()
