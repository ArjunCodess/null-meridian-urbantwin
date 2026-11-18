"""Synthetic-only second-stage geometry regression and hard-negative rejection.

The proposal detector is frozen. Point statistics are expressed in proposal
coordinates so the refiner does not learn the absolute intersection location.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
import joblib
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from .common import TRACKS,load_json,load_points,load_labels,save_json,digest,crop_labels,crop
from .geometry import iou3d,nms

def validate_refiner_checkpoint(bundle,checkpoint_path,classes):
    """Accept legacy provenance or an explicitly deployment-bound OOF bundle."""
    if bundle.get('training_domain')!='synthetic' or bundle['classes']!=classes:
        raise ValueError('Refiner synthetic domain/classes mismatch')
    if bundle.get('proposal_training')=='out_of_fold_v1':
        from .train_refiner_oof import reject_in_sample_provenance
        reject_in_sample_provenance(bundle)
        expected=bundle.get('deployment_checkpoint_sha256')
    else:expected=bundle.get('proposal_checkpoint_sha256')
    if expected!=digest(checkpoint_path):raise ValueError('Refiner deployment/proposal checkpoint mismatch')

def features(points,tree,proposal,classes):
    b=np.asarray(proposal['box']);radius=max(2.,float(np.linalg.norm(b[3:5])/2+1.2))
    ids=tree.query_ball_point(b[:2],radius);p=points[ids,:3]-b[:3]
    c,s=np.cos(b[6]),np.sin(b[6]);rotation=np.array([[c,s,0],[-s,c,0],[0,0,1]])
    p=p@rotation.T
    p=p[(abs(p[:,0])<b[3]/2+1.2)&(abs(p[:,1])<b[4]/2+1.2)]
    # Deterministic bounded work; no target-dependent model/statistics updates.
    if len(p)>1500:p=p[np.linspace(0,len(p)-1,1500,dtype=int)]
    quantiles=np.quantile(p,[0,.1,.25,.5,.75,.9,1],axis=0).ravel() if len(p) else np.zeros(21)
    ground=np.quantile(p[:,2],.1) if len(p) else 0.
    foreground=p[p[:,2]>ground+.25]
    fgq=np.quantile(foreground,[.1,.5,.9],axis=0).ravel() if len(foreground) else np.zeros(9)
    cov=np.cov(foreground.T) if len(foreground)>2 else np.zeros((3,3))
    eig,vec=np.linalg.eigh(cov[:2,:2]);axis=vec[:,-1];angle=np.arctan2(axis[1],axis[0])
    return np.r_[np.log1p([len(p),len(foreground)]),np.log(np.maximum(b[3:6],.05)),quantiles,fgq,
                 cov[np.triu_indices(3)],eig[-1]-eig[0],np.sin(2*angle),np.cos(2*angle),
                 [float(proposal['class']==name) for name in classes]].astype(np.float32)

def encode(proposal,target):
    a=np.asarray(proposal);b=np.asarray(target);d=b[:3]-a[:3];c,s=np.cos(a[6]),np.sin(a[6])
    return np.r_[d[0]*c+d[1]*s,-d[0]*s+d[1]*c,d[2],np.log(b[3:6]/a[3:6]),np.sin(2*(b[6]-a[6])),np.cos(2*(b[6]-a[6]))]

def decode(proposal,residual):
    a=np.asarray(proposal,dtype=float).copy();r=np.asarray(residual);c,s=np.cos(a[6]),np.sin(a[6])
    a[:3]+=[r[0]*c-r[1]*s,r[0]*s+r[1]*c,r[2]]
    a[3:6]*=np.exp(np.clip(r[3:6],-.7,.7));a[6]+=.5*np.arctan2(r[6],r[7])
    a[6]=(a[6]+np.pi)%(2*np.pi)-np.pi
    return a.tolist()

def apply_refiner(points,predictions,bundle,*,objectness_threshold=.5,detector_weight=0.,nms_threshold=.15,max_boxes=100,class_limits=None):
    if not 0<=detector_weight<=1 or not 0<=objectness_threshold<=1:
        raise ValueError('Score fusion weight and objectness threshold must lie in [0,1]')
    if not predictions:return []
    tree=cKDTree(points[:,:2]);x=np.stack([features(points,tree,p,bundle['classes']) for p in predictions])
    prob=bundle['classifier'].predict_proba(x)[:,1];residual=bundle['regressor'].predict(x);result=[]
    for p,score,delta in zip(predictions,prob,residual):
        if score<objectness_threshold:continue
        combined=float(score)**(1-detector_weight)*float(p['score'])**detector_weight
        result.append({'box':decode(p['box'],delta),'class':p['class'],'score':combined})
    return nms(result,threshold=nms_threshold,limit=max_boxes,class_limits=class_limits)

def main():
    from .detector import CompactPillar,predict
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--track',choices=TRACKS,required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--detector',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu',help='Device for the frozen proposal detector');args=parser.parse_args()
    torch.set_num_threads(1);rng=np.random.default_rng(1234);classes=TRACKS[args.track]['classes']
    rows=sorted(load_json(args.manifest),key=lambda r:r['frame_id']);cut=int(.8*len(rows))
    for row in rows:
        if row['domain']!='synthetic' or digest(row['path'])!=row['sha256'] or digest(row['labels'])!=row['labels_sha256']:
            raise ValueError('Refiner accepts only verified synthetic inputs')
    checkpoint=torch.load(args.detector,map_location='cpu',weights_only=True)
    if checkpoint.get('training_domain')!='synthetic' or checkpoint['manifest_sha256']!=digest(args.manifest):
        raise ValueError('Proposal checkpoint does not match synthetic provenance')
    model=CompactPillar(len(classes),int(checkpoint['architecture'][-1])).to(args.device);model.load_state_dict(checkpoint['model']);model.eval()
    x=[];labels=[];reg_x=[];reg_y=[]
    for row in rows[:cut]:
        points=crop(load_points(row['path']),TRACKS[args.track]['range']);gt=crop_labels(load_labels(row['labels'],args.track),TRACKS[args.track]['range']);tree=cKDTree(points[:,:2])
        for obj in gt:
            for _ in range(5):
                p=copy.deepcopy(obj);b=np.array(p['box']);b[:2]+=rng.normal(0,.65,2);b[2]+=rng.normal(0,.55)
                b[3:6]*=rng.uniform(.85,1.15,3);b[6]+=rng.normal(0,.35);p['box']=b.tolist()
                f=features(points,tree,p,classes);x.append(f);labels.append(1);reg_x.append(f);reg_y.append(encode(b,obj['box']))
        proposals=predict(model,points,args.track,device=args.device,threshold=.08)
        for _ in range(10):
            k=int(rng.integers(len(classes)));same=[o for o in gt if o['class']==classes[k]]
            if not same:continue
            p=copy.deepcopy(same[int(rng.integers(len(same)))]);p['box'][:2]=rng.uniform(-38,38,2).tolist();proposals.append(p)
        for p in proposals:
            candidates=[(iou3d(p['box'],g['box']),g) for g in gt if g['class']==p['class']]
            overlap,match=max(candidates,key=lambda pair:pair[0],default=(0,None))
            if .02<overlap<.1:continue
            f=features(points,tree,p,classes);x.append(f);labels.append(int(overlap>=.1))
            if overlap>=.1:reg_x.append(f);reg_y.append(encode(p['box'],match['box']))
    classifier=ExtraTreesClassifier(n_estimators=64,max_depth=18,min_samples_leaf=3,class_weight='balanced',n_jobs=1,random_state=1234)
    regressor=ExtraTreesRegressor(n_estimators=64,max_depth=18,min_samples_leaf=3,n_jobs=1,random_state=1234)
    classifier.fit(np.asarray(x),labels);regressor.fit(np.asarray(reg_x),np.asarray(reg_y))
    args.output.mkdir(parents=True,exist_ok=True)
    bundle={'classifier':classifier,'regressor':regressor,'classes':classes,'training_domain':'synthetic',
            'manifest_sha256':digest(args.manifest),'proposal_checkpoint_sha256':digest(args.detector)}
    joblib.dump(bundle,args.output/'refiner.joblib',compress=3)
    refined={};unrefined={};totals={name:{'TP':0,'FP':0,'GT':0} for name in ['proposal','refined']}
    for row in rows[cut:]:
        points=crop(load_points(row['path']),TRACKS[args.track]['range']);gt=crop_labels(load_labels(row['labels'],args.track),TRACKS[args.track]['range'])
        proposal=predict(model,points,args.track,device=args.device,threshold=.08);ref=apply_refiner(points,proposal,bundle)
        unrefined[row['frame_id']]=nms(proposal);refined[row['frame_id']]=ref
        for name,predictions in [('proposal',unrefined[row['frame_id']]),('refined',ref)]:
            used=set();totals[name]['GT']+=len(gt)
            for p in sorted(predictions,key=lambda p:-p['score']):
                matches=[(iou3d(p['box'],g['box']),i) for i,g in enumerate(gt) if i not in used and p['class']==g['class']]
                overlap,index=max(matches,default=(0,-1))
                if overlap>=.5:totals[name]['TP']+=1;used.add(index)
                else:totals[name]['FP']+=1
    for stats in totals.values():
        stats['precision']=stats['TP']/max(stats['TP']+stats['FP'],1);stats['recall']=stats['TP']/max(stats['GT'],1)
    save_json(args.output/'refined_predictions.json',refined);save_json(args.output/'proposal_predictions.json',unrefined)
    save_json(args.output/'comparison.json',{'domain':'synthetic_validation','official':False,'training_samples':len(x),
              'regression_samples':len(reg_x),'held_out_real_used':False,'results':totals,
              'proposal_device':args.device,'proposal_threshold':.08,'refiner_objectness_threshold':.5,
              'note':'Fixed thresholds; this is a source-only development comparison, not KITTI mAP or evidence of transfer.'})
    print(totals,flush=True)

if __name__=='__main__':main()
