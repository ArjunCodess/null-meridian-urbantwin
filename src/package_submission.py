"""Validate and package the published Codabench layout; never sign for the user."""
from __future__ import annotations
import argparse
import json
import math
import zipfile
from pathlib import Path
import numpy as np
from .common import TRACKS,load_points,load_json,save_json,digest

STATEMENT='I confirm that I did not train on any frame listed in forbidden_frames.txt, and that all submitted detections were produced by a model trained only on synthetic data I generated.'

def validate(synthetic,predictions,frame_ids,declaration,track):
    files=sorted(Path(synthetic).glob('*.bin'))
    if len(files)!=50:raise ValueError(f'Expected 50 synthetic files; found {len(files)}')
    if {p.name for p in files}!={f'{i:04d}.bin' for i in range(1,51)}:raise ValueError('Expected 0001.bin through 0050.bin')
    for file in files:
        if file.stat().st_size>1_500_000:raise ValueError(f'Synthetic file exceeds conservative 1.5 MB cap: {file}')
        points=load_points(file,channels=3)
        if len(points)==0:raise ValueError('Empty synthetic frame')
    ids=load_json(frame_ids);pred=load_json(predictions)
    if not isinstance(ids,list) or len(ids)!=50 or len(set(ids))!=50:raise ValueError('Canonical IDs must be a list of 50 unique strings')
    if set(pred)!=set(ids):raise ValueError('Prediction keys must match all 50 canonical IDs exactly')
    valid={c.lower() for c in TRACKS[track]['classes']}
    for frame,objects in pred.items():
        if not isinstance(objects,list):raise ValueError(f'Detection list required for {frame}')
        for obj in objects:
            b=np.asarray(obj['box'],dtype=float)
            if b.shape!=(7,) or not np.isfinite(b).all() or (b[3:6]<=0).any():raise ValueError('Invalid box')
            if isinstance(obj['score'],bool) or not math.isfinite(obj['score']) or not 0<=obj['score']<=1:raise ValueError('Invalid confidence')
            if obj['class'].lower() not in valid:raise ValueError('Class not in track taxonomy')
    declaration=Path(declaration)
    if declaration.suffix.lower() not in ['.txt','.pdf']:raise ValueError('Declaration must be TXT or PDF')
    if declaration.suffix.lower()=='.txt':
        text=declaration.read_text(encoding='utf-8')
        normalized=' '.join(text.replace('`','').split()).lower()
        if STATEMENT.lower() not in normalized:raise ValueError('Required declaration statement missing')
        if '[unsigned]' in normalized or '[signature]' in normalized:raise ValueError('Declaration template is unsigned')
        if not any(line.lower().startswith('signed by:') and line.split(':',1)[1].strip() for line in text.splitlines()):
            raise ValueError('TXT declaration needs a completed Signed by: line')
    if declaration.stat().st_size==0:raise ValueError('Empty declaration')
    return files

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--synthetic',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--frame-ids',type=Path,required=True)
    p.add_argument('--declaration',type=Path,required=True);p.add_argument('--track',choices=TRACKS,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();files=validate(a.synthetic,a.predictions,a.frame_ids,a.declaration,a.track)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,'synthetic/'+f.name)
        z.write(a.predictions,'predictions.json');z.write(a.declaration,'declaration'+a.declaration.suffix.lower())
    save_json(a.output.with_suffix('.manifest.json'),{'track':a.track,'zip_sha256':digest(a.output),'official_scorer_validated':False,
              'files':{f'synthetic/{f.name}':digest(f) for f in files},'predictions_sha256':digest(a.predictions),
              'canonical_ids_sha256':digest(a.frame_ids),'declaration_sha256':digest(a.declaration)})
    print(a.output)

if __name__=='__main__':main()
