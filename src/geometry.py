"""Portable oriented 3D IoU and class-aware NMS, independent of CUDA extensions."""
import numpy as np

def corners(box):
    b=np.asarray(box);x,y=b[:2];dx,dy=b[3:5]/2;a=b[6]
    p=np.array([[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy]])
    r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    return p@r.T+[x,y]

def area(poly):
    if len(poly)<3:return 0.
    p=np.asarray(poly);return abs(float(np.dot(p[:,0],np.roll(p[:,1],-1))-np.dot(p[:,1],np.roll(p[:,0],-1))))/2

def cross(a,b):return a[0]*b[1]-a[1]*b[0]

def intersection(subject,clip):
    output=list(subject)
    for a,b in zip(clip,np.roll(clip,-1,axis=0)):
        inp=output;output=[]
        if not inp:break
        s=inp[-1]
        for e in inp:
            inside_e=cross(b-a,e-a)>=-1e-9;inside_s=cross(b-a,s-a)>=-1e-9
            if inside_e!=inside_s:
                den=cross(e-s,b-a)
                if abs(den)>1e-12:output.append(s+(e-s)*(cross(a-s,b-a)/den))
            if inside_e:output.append(e)
            s=e
    return output

def iou3d(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if np.any(np.abs(a[:2]-b[:2])>(np.linalg.norm(a[3:5])+np.linalg.norm(b[3:5]))/2):return 0.
    h=max(0,min(a[2]+a[5]/2,b[2]+b[5]/2)-max(a[2]-a[5]/2,b[2]-b[5]/2))
    if h<=0:return 0.
    volume=area(intersection(corners(a),corners(b)))*h
    return float(volume/max(np.prod(a[3:6])+np.prod(b[3:6])-volume,1e-12))

def nms(predictions,threshold=.15,limit=100,class_limits=None):
    remaining=sorted(predictions,key=lambda p:p['score'],reverse=True);kept=[]
    while remaining and len(kept)<limit:
        p=remaining.pop(0)
        if class_limits and sum(q['class']==p['class'] for q in kept)>=class_limits.get(p['class'],limit):continue
        kept.append(p)
        overlap_threshold=threshold[p['class']] if isinstance(threshold,dict) else threshold
        remaining=[q for q in remaining if q['class']!=p['class'] or iou3d(p['box'],q['box'])<=overlap_threshold]
    return kept

def transform(points,boxes,angle=0.,scale=1.,inverse=False):
    """Similarity transform with an exact inverse for frozen geometric TTA."""
    if scale<=0:raise ValueError('Scale must be positive')
    p=points.copy();b=np.asarray(boxes,dtype=np.float32).reshape(-1,7).copy()
    if inverse:angle=-angle;scale=1/scale
    c,s=np.cos(angle),np.sin(angle);r=np.array([[c,-s],[s,c]])
    p[:,:2]=p[:,:2]@r.T*scale;p[:,2]*=scale
    if len(b):
        b[:,:2]=b[:,:2]@r.T*scale;b[:,2:6]*=scale;b[:,6]+=angle
        b[:,6]=(b[:,6]+np.pi)%(2*np.pi)-np.pi
    return p,b

def inside_box(points,box):
    p=points[:,:3]-np.asarray(box[:3]);c,s=np.cos(box[6]),np.sin(box[6])
    x=p[:,0]*c+p[:,1]*s;y=-p[:,0]*s+p[:,1]*c
    return (abs(x)<=box[3]/2)&(abs(y)<=box[4]/2)&(abs(p[:,2])<=box[5]/2)
