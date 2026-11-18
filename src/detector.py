"""Portable compact pillar detector and frozen geometric test-time ensemble.

This CPU-compatible baseline uses aggregated pillar features and a center
heatmap head. It is not a reproduction of PointPillars/SECOND. The original
LiGuard PointPillars backend is retained in vendor/, but requires CUDA ops.
No real-frame BatchNorm updates, pretrained real-data weights, or real labels.
"""
from __future__ import annotations
import argparse
import copy
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import TRACKS,load_points,load_labels,load_json,save_json,crop,digest,crop_labels
from .geometry import transform,nms,inside_box,iou3d
from .training_splits import resolve_training_splits

CELL=.5
LAST_STATE_NAME='training_last_state.pt'
CHECKPOINT_INTERVAL_DEFAULT=50

def pillar_stride(cell=CELL):
    return float(cell)

def output_stride(version,pillar=CELL):
    return pillar if version>=5 else pillar*2

def architecture_metadata(version,classes):
    """Train/infer geometry contract; same-class objects still share one regression cell."""
    return {'pillar_stride_m':pillar_stride(),'output_stride_m':output_stride(version),
            'heading_period':float(np.pi if version>=2 else 2*np.pi),
            'class_specific_box_regression':version>=5,'class_groups':[[c] for c in classes] if version>=5 else None,
            'neighborhood_regression_offsets':version>=5,
            'same_class_single_cell_limit':True,
            'collision_note':'Finer stride and class-specific heads reduce cross-class overwrite; '
                             'two same-class objects in one output cell still collide.'}

def pillar_features(points,bounds,cell=CELL):
    p=crop(points,bounds);size=np.rint((np.asarray(bounds[3:5])-bounds[:2])/cell).astype(int)
    w,h=map(int,size);feat=np.zeros((7,h*w),dtype=np.float32)
    if not len(p):return feat.reshape(7,h,w)
    ij=np.floor((p[:,:2]-bounds[:2])/cell).astype(int);idx=ij[:,1]*w+ij[:,0]
    count=np.bincount(idx,minlength=h*w);valid=count>0
    z=p[:,2];maximum=np.full(h*w,-np.inf);minimum=np.full(h*w,np.inf)
    np.maximum.at(maximum,idx,z);np.minimum.at(minimum,idx,z)
    feat[0]=np.log1p(count)/np.log(65)
    feat[1,valid]=np.bincount(idx,weights=z,minlength=h*w)[valid]/count[valid]/10
    feat[2,valid]=maximum[valid]/10;feat[3,valid]=minimum[valid]/10
    for channel,axis in [(4,0),(5,1)]:
        offsets=(p[:,axis]-bounds[axis])/cell-ij[:,axis]-.5
        feat[channel,valid]=np.bincount(idx,weights=offsets,minlength=h*w)[valid]/count[valid]
    if p.shape[1]==4:feat[6,valid]=np.bincount(idx,weights=p[:,3],minlength=h*w)[valid]/count[valid]
    return feat.reshape(7,h,w)

class CompactPillar(nn.Module):
    def __init__(self,classes,version=1):
        super().__init__()
        self.version=version
        self.num_classes=classes
        if version==4:
            self.point_encoder=nn.Sequential(nn.Linear(9,32),nn.LayerNorm(32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        elif version==5:
            self.point_encoder=nn.Sequential(nn.Linear(9,48),nn.LayerNorm(48),nn.ReLU(),nn.Linear(48,48),nn.ReLU())
            pdim=48
            self.body=nn.Sequential(nn.Conv2d(7+pdim,36,3,padding=1,bias=False),nn.GroupNorm(6,36),nn.ReLU(),
                                    nn.Conv2d(36,pdim,3,padding=1,stride=1,bias=False),nn.GroupNorm(8,pdim),nn.ReLU(),
                                    nn.Conv2d(pdim,pdim,3,padding=1,bias=False),nn.GroupNorm(8,pdim),nn.ReLU())
            self.context=nn.Sequential(*[layer for stride in (2,2,1) for layer in
                  (nn.Conv2d(pdim,pdim,3,padding=1,stride=stride,bias=False),nn.GroupNorm(8,pdim),nn.ReLU())])
            self.heat=nn.Conv2d(pdim,classes,1);self.box=nn.Conv2d(pdim,8*classes,1)
            nn.init.constant_(self.heat.bias,-2.19)
            return
        self.body=nn.Sequential(nn.Conv2d(39 if version==4 else 7,24,3,padding=1,bias=False),nn.GroupNorm(6,24),nn.ReLU(),
                                nn.Conv2d(24,32,3,padding=1,stride=2,bias=False),nn.GroupNorm(8,32),nn.ReLU(),
                                nn.Conv2d(32,32,3,padding=1,bias=False),nn.GroupNorm(8,32),nn.ReLU())
        self.context=nn.Sequential(*[layer for stride in (2,2,1) for layer in
              (nn.Conv2d(32,32,3,padding=1,stride=stride,bias=False),nn.GroupNorm(8,32),nn.ReLU())]) if version in (2,4) else nn.Identity()
        self.heat=nn.Conv2d(32,classes,1);self.box=nn.Conv2d(32,8,1)
        nn.init.constant_(self.heat.bias,-2.19)
    def encode_pillars(self,points,height,width):
        indices=torch.floor((points[:,:2]+40)/CELL).long()
        valid=(indices[:,0]>=0)&(indices[:,0]<width)&(indices[:,1]>=0)&(indices[:,1]<height)
        points=points[valid];indices=indices[valid];index=indices[:,1]*width+indices[:,0]
        sums=points.new_zeros((height*width,3));counts=points.new_zeros((height*width,1))
        sums.index_add_(0,index,points[:,:3]);counts.index_add_(0,index,points.new_ones((len(points),1)))
        means=sums/counts.clamp(min=1)
        intensity=points[:,3:4] if points.shape[1]==4 else points.new_zeros((len(points),1))
        decorated=torch.cat([points[:,:3]/points.new_tensor([40,40,10]),intensity,
                             (points[:,:2]+40)/CELL-indices-.5,points[:,:3]-means[index]],dim=1)
        encoded=self.point_encoder(decorated)
        pdim=48 if self.version==5 else 32
        grid=encoded.new_zeros((height*width,pdim))
        grid.scatter_reduce_(0,index[:,None].expand(-1,pdim),encoded,reduce='amax',include_self=True)
        return grid.T.reshape(1,pdim,height,width)

    def forward(self,x,points=None):
        if self.version in (4,5):
            x=torch.cat([x,self.encode_pillars(points,x.shape[-2],x.shape[-1])],dim=1)
        f=self.body(x)
        f=f+F.interpolate(self.context(f),size=f.shape[-2:],mode='bilinear',align_corners=False) if self.version in (2,4,5) else f
        return self.heat(f),self.box(f)

def _box_target_slice(gx,gy,ix,iy,b,heading_multiplier,k,n_classes):
    vec=[gx-ix,gy-iy,b[2]/10,*np.log(b[3:6]),np.sin(heading_multiplier*b[6]),np.cos(heading_multiplier*b[6])]
    if n_classes<=1:
        return np.asarray(vec,np.float32)
    out=np.zeros((8*n_classes,),np.float32);out[k*8:(k+1)*8]=vec
    return out

def targets(objects,bounds,classes,shape,heading_multiplier=1,version=None):
    h,w=shape;cell=output_stride(version if version is not None else 1)
    if version is None or version<5:cell=CELL*2
    classwise=version is not None and version>=5
    channels=8*len(classes) if classwise else 8
    heat=np.zeros((len(classes),h,w),np.float32);reg=np.zeros((channels,h,w),np.float32)
    mask=np.zeros((len(classes),h,w) if classwise else (h,w),bool)
    nearest=np.full((len(classes),h,w),np.inf) if classwise else None
    for obj in objects:
        b=np.asarray(obj['box']);gx,gy=(b[:2]-bounds[:2])/cell;ix,iy=int(np.floor(gx)),int(np.floor(gy))
        if not (0<=ix<w and 0<=iy<h and bounds[2]<=b[2]<bounds[5]):continue
        k=classes.index(obj['class'])
        for dy in range(-2,3):
            for dx in range(-2,3):
                if 0<=iy+dy<h and 0<=ix+dx<w:
                    heat[k,iy+dy,ix+dx]=max(heat[k,iy+dy,ix+dx],math.exp(-(dx*dx+dy*dy)/2))
                    if classwise and dx*dx+dy*dy<nearest[k,iy+dy,ix+dx]:
                        cx,cy=ix+dx,iy+dy
                        slice_target=_box_target_slice(gx,gy,cx,cy,b,heading_multiplier,k,len(classes))
                        reg[k*8:(k+1)*8,iy+dy,ix+dx]=slice_target[k*8:(k+1)*8]
                        mask[k,iy+dy,ix+dx]=True
                        nearest[k,iy+dy,ix+dx]=dx*dx+dy*dy
        ng=len(classes) if classwise else 1
        target=_box_target_slice(gx,gy,ix,iy,b,heading_multiplier,k,ng)
        if not classwise:
            reg[:,iy,ix]=target
            mask[iy,ix]=True
    return torch.from_numpy(heat),torch.from_numpy(reg),torch.from_numpy(mask)

def loss_fn(outputs,target,metric_regression=False):
    heat,reg=outputs;y,box,mask=target
    prob=heat.sigmoid().clamp(1e-5,1-1e-5);pos=y.eq(1);neg=~pos
    focal=-(torch.log(prob)*(1-prob)**2*pos + torch.log(1-prob)*prob**2*(1-y)**4*neg).sum()/pos.sum().clamp(min=1)
    if mask.any():
        if mask.ndim==4:
            batch,channels,height,width=reg.shape
            predicted=reg.reshape(batch,channels//8,8,height,width).permute(0,1,3,4,2)[mask]
            expected=box.reshape(batch,channels//8,8,height,width).permute(0,1,3,4,2)[mask]
        else:
            predicted=reg.permute(0,2,3,1)[mask];expected=box.permute(0,2,3,1)[mask]
        if metric_regression:
            unit=reg.new_tensor([2.,2.,10.,1.,1.,1.,1.,1.])
            repeats=max(1,predicted.shape[-1]//8)
            weights=unit.repeat(repeats) if repeats>1 else unit
            predicted=predicted*weights;expected=expected*weights
        regloss=F.smooth_l1_loss(predicted,expected)
    else:regloss=reg.sum()*0
    return focal+2*regloss,{'heat':float(focal.detach()),'reg':float(regloss.detach())}

def augment(points,objects,rng,donor=None):
    objects=copy.deepcopy(objects);boxes=np.array([o['box'] for o in objects]).reshape(-1,7)
    points,boxes=transform(points,boxes,rng.uniform(-np.pi,np.pi),rng.uniform(.95,1.05))
    if rng.random()<.5:points[:,1]*=-1;boxes[:,1]*=-1;boxes[:,6]*=-1
    # Small pitch/roll perturbation. Convert tilted cuboids to enclosing yaw-only boxes.
    pitch,roll=rng.normal(0,.006,2)
    ry=np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    rx=np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    rotation=(rx@ry).astype(np.float32);points[:,:3]=points[:,:3]@rotation.T
    for i,b in enumerate(boxes):
        b[:3]=b[:3]@rotation.T;c,s=np.cos(b[6]),np.sin(b[6]);rz=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        b[3:6]=np.abs(rz.T@rotation@rz)@b[3:6]
    for o,b in zip(objects,boxes):o['box']=b.tolist()
    if donor is not None and rng.random()<.5:
        donor_p,donor_obj=donor
        if donor_obj:
            o=copy.deepcopy(donor_obj[rng.integers(len(donor_obj))]);b=np.array(o['box'])
            extracted=donor_p[inside_box(donor_p,b)].copy()
            shift=np.r_[rng.uniform(-8,8,2),0];candidate=b.copy();candidate[:3]+=shift
            if len(extracted)>=5 and all(iou3d(candidate,q['box'])<.01 for q in objects):
                points=points[~inside_box(points,candidate)];extracted[:,:3]+=shift
                if extracted.shape[1]==points.shape[1]:
                    points=np.concatenate([points,extracted]);o['box']=candidate.tolist();objects.append(o)
    return points,objects

def _atomic_torch_save(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        torch.save(payload,tmp);os.replace(tmp,path)
    except BaseException:
        if tmp.is_file():tmp.unlink(missing_ok=True)
        raise

def _numpy_rng_to_dict(rng):
    st=rng.bit_generator.state
    inner=st['state']
    if isinstance(inner,dict):inner_payload={k:int(v) for k,v in inner.items()}
    else:inner_payload={'values':[int(x) for x in inner]}
    out={'bit_generator':st['bit_generator'],'inner_state':inner_payload,'has_uint32':bool(st['has_uint32'])}
    if st.get('uinteger') is not None:out['uinteger']=int(st['uinteger'])
    return out

def _numpy_rng_from_dict(rng,payload):
    inner=payload['inner_state']
    if 'values' in inner:inner_state=np.array(inner['values'],dtype=np.uint32)
    else:inner_state={k:int(v) for k,v in inner.items()}
    state={'bit_generator':payload['bit_generator'],'state':inner_state,'has_uint32':payload['has_uint32']}
    if 'uinteger' in payload:state['uinteger']=int(payload['uinteger'])
    rng.bit_generator.state=state

def _training_provenance(args,manifest_sha,version):
    return {'manifest_sha256':manifest_sha,'track':args.track,'architecture':f'CompactPillar-v{version}',
            'selection_metric':getattr(args,'selection_metric','loss'),'seed':int(args.seed),
            'epochs':int(args.epochs),'threads':int(args.threads),'device':str(args.device),
            'runtime_sources':{name:digest(Path(__file__).parent/name) for name in
                               ('detector.py','common.py','geometry.py','training_splits.py','evaluate_source.py')},
            'initialize_from_sha256':digest(args.initialize_from) if args.initialize_from else None}

def _validate_training_provenance(saved,expected):
    for key,value in expected.items():
        if saved.get(key)!=value:
            raise ValueError(f'Training recovery provenance mismatch on {key!r}: '
                             f'checkpoint has {saved.get(key)!r}, run requests {value!r}')

def _collect_rng_bundle(rng):
    out={'numpy':_numpy_rng_to_dict(rng),'torch':torch.get_rng_state().tolist()}
    if getattr(torch,'cuda',None) and torch.cuda.is_available():
        out['cuda_all']=[s.tolist() for s in torch.cuda.get_rng_state_all()]
    return out

def _restore_rng_bundle(rng,bundle):
    _numpy_rng_from_dict(rng,bundle['numpy'])
    torch.set_rng_state(torch.as_tensor(bundle['torch'],dtype=torch.uint8))
    if 'cuda_all' in bundle and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.as_tensor(s,dtype=torch.uint8) for s in bundle['cuda_all']])

def _best_checkpoint_payload(model,args,version,manifest_sha,epoch_one_based):
    return {'model':model.state_dict(),'track':args.track,'architecture':f'CompactPillar-v{version}',
            'training_domain':'synthetic','manifest_sha256':manifest_sha,'epoch':epoch_one_based}

def _save_training_last_state(path,*,model,optimizer,rng_bundle,epoch_index,permutation,step_in_epoch,
                               epoch_losses,history,best,prior_seconds,provenance,run_started):
    perm=None if permutation is None else [int(x) for x in permutation]
    payload={'schema_version':1,'kind':'training_last_state','provenance':provenance,
             'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_bundle,
             'epoch_index':int(epoch_index),'permutation':perm,'step_in_epoch':int(step_in_epoch),
             'epoch_losses':[float(x) for x in epoch_losses],'history':history,'best':float(best),
             'prior_seconds':float(prior_seconds),'run_started_monotonic':float(run_started)}
    _atomic_torch_save(payload,path)

def _load_training_last_state(path,expected_provenance):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('kind')!='training_last_state':
        raise ValueError('Training recovery artifact has unexpected kind')
    _validate_training_provenance(state.get('provenance') or {},expected_provenance)
    return state

@torch.no_grad()
def predict(model,points,track,device='cpu',threshold=.15):
    bounds=TRACKS[track]['range'];classes=TRACKS[track]['classes']
    x=torch.from_numpy(pillar_features(points,bounds)[None]).to(device)
    raw=torch.from_numpy(crop(points,bounds)).to(device) if model.version in (4,5) else None
    heat,reg=model(x,raw);prob=heat.sigmoid();prob=prob*(prob==F.max_pool2d(prob,3,1,1))
    scores,ids=prob.flatten().topk(min(150,prob.numel()));h,w=prob.shape[-2:];result=[]
    stride=output_stride(model.version)
    for score,index in zip(scores.cpu().numpy(),ids.cpu().numpy()):
        if score<threshold:continue
        k=index//(h*w);iy,ix=divmod(index%(h*w),w)
        if model.version>=5:b=reg[0,k*8:(k+1)*8,iy,ix].cpu().numpy()
        else:b=reg[0,:,iy,ix].cpu().numpy()
        box=[float((ix+b[0])*stride+bounds[0]),float((iy+b[1])*stride+bounds[1]),float(b[2]*10),
             *np.exp(np.clip(b[3:6],-2,3)).tolist(),float(np.arctan2(b[6],b[7])/(2 if model.version>=2 else 1))]
        if not np.isfinite(box).all() or not bounds[2]<=box[2]<bounds[5]:continue
        result.append({'box':box,'score':float(score),'class':classes[int(k)]})
    return result

def train(args):
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed);torch.set_num_threads(args.threads)
    records=load_json(args.manifest)
    if len(records)<4:raise ValueError('At least four synthetic frames required')
    for row in records:
        if row.get('domain')!='synthetic':raise ValueError('Detector training accepts synthetic records only')
        if row.get('sha256')!=digest(row['path']):raise ValueError('Training cloud hash mismatch')
        if row.get('labels_sha256')!=digest(row['labels']):raise ValueError('Training label hash mismatch')
    split_plan=resolve_training_splits(records)
    train_rows,val_rows=split_plan.fit_rows,split_plan.selection_rows
    bounds=TRACKS[args.track]['range'];classes=TRACKS[args.track]['classes']
    class SyntheticScenes:
        def __init__(self,rows):
            self.rows=rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self,index):
            if isinstance(index,slice):return (self[i] for i in range(*index.indices(len(self.rows))))
            row=self.rows[index]
            return crop(load_points(row['path']),bounds),crop_labels(load_labels(row['labels'],args.track),bounds)
    train_data=SyntheticScenes(train_rows)
    val_data=SyntheticScenes(val_rows)
    version=int(args.architecture[-1])
    if args.evaluate_only or args.resume:
        initial_checkpoint=torch.load(args.output/'best.pt',map_location=args.device,weights_only=True)
        version=int(initial_checkpoint['architecture'][-1])
    # Keep the context experiment on the native CPU convolution path: the local
    # oneDNN build failed under this machine's fluctuating commit limit.
    if version==2 and args.device=='cpu':torch.backends.mkldnn.enabled=False
    model=CompactPillar(len(classes),version).to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    args.output.mkdir(parents=True,exist_ok=True)
    manifest_sha=digest(args.manifest);provenance=_training_provenance(args,manifest_sha,version)
    last_state_path=args.output/LAST_STATE_NAME
    checkpoint_interval=getattr(args,'checkpoint_interval',CHECKPOINT_INTERVAL_DEFAULT)
    max_train_steps=getattr(args,'max_train_steps',None)
    auto_recover=not getattr(args,'no_auto_recover',False)
    history=[];best=float('inf');start=time.monotonic();start_epoch=0;prior_seconds=0
    recovered=False;epoch_index=0;step_in_epoch=0;permutation=None;epoch_losses=[];train_steps=0
    recovery_mode=None
    if args.resume:
        checkpoint=torch.load(args.output/'best.pt',map_location=args.device,weights_only=True)
        if checkpoint.get('training_domain')!='synthetic' or checkpoint.get('manifest_sha256')!=manifest_sha or checkpoint['track']!=args.track:
            raise ValueError('Resume checkpoint provenance mismatch')
        model.load_state_dict(checkpoint['model']);start_epoch=checkpoint['epoch']
        previous=load_json(args.output/'training_report.json');history=[h for h in previous['history'] if h['epoch']<=start_epoch]
        if getattr(args,'selection_metric','loss')=='map':
            best=min((-h['synthetic_validation_mAP_R40'] for h in history if 'synthetic_validation_mAP_R40' in h),default=float('inf'))
        else:best=min(h['synthetic_validation_loss'] for h in history)
        prior_seconds=previous['seconds'];recovery_mode='legacy_best_checkpoint'
    elif not args.evaluate_only and auto_recover and last_state_path.is_file():
        state=_load_training_last_state(last_state_path,provenance)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        _restore_rng_bundle(rng,state['rng'])
        epoch_index=int(state['epoch_index']);step_in_epoch=int(state['step_in_epoch'])
        permutation=None if state['permutation'] is None else np.asarray(state['permutation'],dtype=np.int64)
        epoch_losses=list(state['epoch_losses']);history=list(state['history']);best=float(state['best'])
        prior_seconds=float(state['prior_seconds']);start_epoch=epoch_index;recovered=True;recovery_mode='auto_last_state'
    elif args.initialize_from and not args.evaluate_only:
        initial=torch.load(args.initialize_from,map_location=args.device,weights_only=True)
        if initial.get('training_domain')!='synthetic' or initial.get('manifest_sha256')!=manifest_sha or initial['track']!=args.track:
            raise ValueError('Initialization checkpoint must match synthetic data provenance')
        model.load_state_dict(initial['model'],strict=True)
        if int(initial['architecture'][-1])==1 and version>=2:
            with torch.no_grad():
                nn.init.normal_(model.box.weight[6:],std=.01);model.box.bias[6:].zero_()
    def persist_last_state(ep_idx,perm,step,losses):
        _save_training_last_state(last_state_path,model=model,optimizer=optimizer,
            rng_bundle=_collect_rng_bundle(rng),epoch_index=ep_idx,permutation=perm,step_in_epoch=step,
            epoch_losses=losses,history=history,best=best,prior_seconds=prior_seconds+time.monotonic()-start,
            provenance=provenance,run_started=start)
    def maybe_stop_after_limit():
        nonlocal train_steps
        if max_train_steps is not None and train_steps>=max_train_steps:
            persist_last_state(epoch,permutation,step_in_epoch,epoch_losses);return True
        return False
    def write_training_report():
        save_json(args.output/'training_report.json',{'history':history,'train_ids':[r['frame_id'] for r in train_rows],
                  'validation_ids':[r['frame_id'] for r in val_rows],'split_plan':split_plan.report,'split_mode':split_plan.mode,
                  'device':args.device,'seconds':prior_seconds+time.monotonic()-start,
                  'resume_optimizer_reset':bool(args.resume),
                  'training_recovery':recovery_mode,
                  'checkpoint_interval':checkpoint_interval,
                  'selection_metric':getattr(args,'selection_metric','loss'),
                  'architecture':f'CompactPillar-v{version}','heading_period':float(np.pi if version>=2 else 2*np.pi),
                  'geometry':architecture_metadata(version,classes),
                  'initialization_checkpoint_sha256':digest(args.initialize_from) if args.initialize_from else None,
                  'parameters':sum(p.numel() for p in model.parameters()),'real_detection_score':None,
                  'purpose':'Source-only baseline smoke experiment, not a tuned competitive detector'})
    end_epoch=0 if args.evaluate_only else args.epochs
    for epoch in range(start_epoch,end_epoch):
        if not (recovered and epoch==epoch_index and permutation is not None):
            permutation=rng.permutation(len(train_rows));step_in_epoch=0;epoch_losses=[]
        else:recovered=False
        model.train()
        while step_in_epoch<len(permutation):
            i=int(permutation[step_in_epoch])
            points,objects=train_data[i];donor=train_data[int(rng.integers(len(train_rows)))]
            p,o=augment(points,objects,rng,donor)
            x=torch.from_numpy(pillar_features(p,bounds)[None]).to(args.device)
            raw=torch.from_numpy(crop(p,bounds)).to(args.device) if version in (4,5) else None
            out=model(x,raw);target=targets(o,bounds,classes,out[0].shape[-2:],2 if version>=2 else 1,version=version)
            target=tuple(t[None].to(args.device) for t in target)
            loss,_=loss_fn(out,target,version>=2);optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
            epoch_losses.append(float(loss.detach()));train_steps+=1;step_in_epoch+=1
            if train_steps%checkpoint_interval==0:
                persist_last_state(epoch,permutation,step_in_epoch,epoch_losses)
            if maybe_stop_after_limit():return
        model.eval();val=[]
        with torch.no_grad():
            for points,objects in val_data:
                raw=torch.from_numpy(crop(points,bounds)).to(args.device) if version in (4,5) else None
                out=model(torch.from_numpy(pillar_features(points,bounds)[None]).to(args.device),raw)
                target=tuple(t[None].to(args.device) for t in targets(objects,bounds,classes,out[0].shape[-2:],2 if version>=2 else 1,version=version))
                loss,_=loss_fn(out,target,version>=2);val.append(float(loss))
        entry={'epoch':epoch+1,'train_loss':float(np.mean(epoch_losses)),'synthetic_validation_loss':float(np.mean(val))}
        if getattr(args,'selection_metric','loss')=='map':
            from .evaluate_source import official_synthetic_evaluation
            validation_predictions={row['frame_id']:nms(predict(model,points,args.track,args.device,threshold=.01))
                                    for row,(points,objects) in zip(val_rows,val_data)}
            entry['synthetic_validation_mAP_R40']=official_synthetic_evaluation(
                args.manifest,validation_predictions,args.track,evaluation_split='development')['detection_3d']['mAP_0.5_R40']
        history.append(entry);print(entry,flush=True)
        selection=-entry['synthetic_validation_mAP_R40'] if getattr(args,'selection_metric','loss')=='map' else entry['synthetic_validation_loss']
        if selection<best:
            best=selection
            _atomic_torch_save(_best_checkpoint_payload(model,args,version,manifest_sha,epoch+1),args.output/'best.pt')
        step_in_epoch=0;permutation=None
        persist_last_state(epoch+1,None,0,[])
        write_training_report()
    if last_state_path.is_file() and not args.evaluate_only and end_epoch>start_epoch:
        last_state_path.unlink(missing_ok=True)
    # Source-domain qualitative predictions and simple matched detection counts.
    checkpoint=torch.load(args.output/'best.pt',map_location=args.device,weights_only=True);model.load_state_dict(checkpoint['model']);model.eval()
    preds={};tp=fp=total=0
    for row,(points,objects) in zip(val_rows,val_data):
        pred=nms(predict(model,points,args.track,args.device,threshold=.15));preds[row['frame_id']]=pred;matched=set();total+=len(objects)
        for p in sorted(pred,key=lambda o:-o['score']):
            choices=[(iou3d(p['box'],o['box']),j) for j,o in enumerate(objects) if o['class']==p['class'] and j not in matched]
            score,j=max(choices,default=(0,-1))
            if score>=.5:tp+=1;matched.add(j)
            else:fp+=1
    save_json(args.output/'synthetic_validation_predictions.json',preds)
    save_json(args.output/'synthetic_validation_detection.json',{'official':False,'domain':'synthetic',
              'iou_threshold':.5,'confidence_threshold':.15,'TP':tp,'FP':fp,'GT':total,
              'precision':tp/max(tp+fp,1),'recall':tp/max(total,1),'KITTI_mAP':None})

def infer(args):
    ckpt=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
    if ckpt.get('training_domain')!='synthetic' or ckpt['track']!=args.track:raise ValueError('Checkpoint provenance/track mismatch')
    model=CompactPillar(len(TRACKS[args.track]['classes']),int(ckpt['architecture'][-1])).to(args.device);model.load_state_dict(ckpt['model']);model.eval()
    refiner=None
    if args.refiner:
        import joblib
        refiner=joblib.load(args.refiner)
        from .box_refiner import validate_refiner_checkpoint
        validate_refiner_checkpoint(refiner,args.checkpoint,TRACKS[args.track]['classes'])
    ids=load_json(args.frame_ids)
    if not isinstance(ids,list) or len(ids)!=50 or len(set(ids))!=50:raise ValueError('Expected exactly 50 unique canonical test IDs')
    proposal_threshold=.08 if refiner else args.threshold
    result={}
    for frame in ids:
        candidates=[args.points/(str(frame)+ext) for ext in ['.npy','.bin','.pcd']]
        path=next((p for p in candidates if p.exists()),None)
        if path is None:raise FileNotFoundError(frame)
        points=load_points(path,args.channels);all_preds=[]
        variants=[(0,1)] if not args.tta else [(0,1),(-.05,1),(.05,1),(0,.97),(0,1.03)]
        for angle,scale in variants:
            p,_=transform(points,[],angle,scale);pred=predict(model,p,args.track,args.device,proposal_threshold)
            for o in pred:
                _,boxes=transform(np.zeros((0,3)),[o['box']],angle,scale,inverse=True);o['box']=boxes[0].tolist()
            all_preds.extend(pred)
        if refiner:
            from .box_refiner import apply_refiner
            result[frame]=apply_refiner(points,all_preds,refiner)
        else:result[frame]=nms(all_preds)
    save_json(args.output,result)
    save_json(args.output.with_suffix('.provenance.json'),{'checkpoint_sha256':digest(args.checkpoint),
              'training_domain':'synthetic','batchnorm_adaptation':False,'frozen_geometric_tta':args.tta,
              'threshold':proposal_threshold,'proposal_threshold':proposal_threshold,
              'refiner_objectness_threshold':.5 if refiner else None,
              'refiner_sha256':digest(args.refiner) if refiner else None,
              'frame_ids_sha256':digest(args.frame_ids)})

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--manifest',type=Path,required=True);t.add_argument('--epochs',type=int,default=8)
    t.add_argument('--output',type=Path,required=True);t.add_argument('--seed',type=int,default=1234);t.add_argument('--threads',type=int,default=4)
    t.add_argument('--evaluate-only',action='store_true',help='Evaluate the existing best.pt without retraining or replacing reports')
    t.add_argument('--resume',action='store_true',help='Continue best synthetic checkpoint to --epochs total; optimizer is reset')
    t.add_argument('--architecture',choices=['v1','v2','v3','v4','v5'],default='v3')
    t.add_argument('--selection-metric',choices=['loss','map'],default='map',help='Select checkpoints using organizer R40 AP on synthetic validation')
    t.add_argument('--initialize-from',type=Path,help='Warm start from a checkpoint trained on this exact synthetic manifest')
    t.add_argument('--checkpoint-interval',type=int,default=CHECKPOINT_INTERVAL_DEFAULT,
                   help='Save restart-safe training_last_state.pt every N minibatch steps and after each epoch')
    t.add_argument('--no-auto-recover',action='store_true',help='Do not resume from training_last_state.pt on startup')
    t.add_argument('--max-train-steps',type=int,help='Stop after N minibatch steps (for recovery smoke tests)')
    i=sub.add_parser('infer');i.add_argument('--checkpoint',type=Path,required=True);i.add_argument('--frame-ids',type=Path,required=True)
    i.add_argument('--points',type=Path,required=True);i.add_argument('--output',type=Path,required=True);i.add_argument('--channels',type=int,choices=[3,4],default=3)
    i.add_argument('--tta',action='store_true');i.add_argument('--threshold',type=float,default=.15)
    i.add_argument('--refiner',type=Path,help='Optional trusted local synthetic-only refiner.joblib')
    for parser in [t,i]:parser.add_argument('--track',choices=TRACKS,required=True);parser.add_argument('--device',default='cpu')
    args=p.parse_args();train(args) if args.command=='train' else infer(args)

if __name__=='__main__':main()
