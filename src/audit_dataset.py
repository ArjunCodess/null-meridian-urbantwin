"""Geometry and class-coverage audit for synthetic manifests.

Official test clouds, realism references, and real labels are refused before
they are opened. Boxes are read as OpenPCDet centers: z is the cuboid center
and the bottom face is z - dz/2. The point-z comparison reports which of those
two origins the returns support. It does not rewrite boxes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from .common import (
    TRACKS, _DIGITS, _optional_key, assert_not_held_out, digest, label_class,
    load_labels, load_points, crop, parse_frame_key,
)
from .geometry import inside_box, iou3d

SCORED_CLASSES = {
    'v2x': ('vehicle', 'pedestrian', 'truck'),
    # Unknown has no held-out ground truth and is excluded from the LUMPI mean.
    'lumpi': ('Person', 'Car', 'Bicycle', 'Motorcycle', 'Bus', 'Truck', 'Van'),
}
WALKERS = frozenset({'child', 'roadworker', 'pedestrian'})
RIDERS = frozenset({'scooter', 'scooterrider', 'motorcycle', 'motorcylerider', 'bicyclerider'})
MINIMUM_INSTANCES = {'train': 20, 'locked_validation': 5}
SUPPORT_MIN_POINTS = 5
_FRAME_ERROR_PREFIXES = ('Invalid synthetic box', 'Invalid cloud', 'Nonfinite', 'Unsupported cloud')


def bind_synthetic_row(row, track):
    """Validate one synthetic record. This is the only identity parse for that row."""
    if track not in TRACKS: raise ValueError(f'Unrecognized track: {track!r}')
    if not isinstance(row, dict) or any(field not in row for field in ('frame_id', 'path', 'labels')):
        raise ValueError('Unrecognized manifest row')
    assert_not_held_out(row['path'])
    assert_not_held_out(row['labels'])
    if row.get('domain') != 'synthetic':
        raise ValueError('Only synthetic rows are readable here')
    if Path(row['path']).suffix.lower() not in {'.npy', '.bin', '.pcd'}:
        raise ValueError(f'Unrecognized synthetic cloud: {row["path"]}')
    if Path(row['labels']).suffix.lower() not in {'.txt', '.json'}:
        raise ValueError(f'Unrecognized synthetic labels: {row["labels"]}')
    full = parse_frame_key(row['frame_id'], track, sensor=row.get('sensor'), sequence=row.get('sequence'))
    stem = Path(row['path']).stem
    if _DIGITS.fullmatch(stem):
        if int(stem) != full.frame:
            raise ValueError(f'Frame identity mismatch: {row["frame_id"]} vs {stem}')
    else:
        stem_key = _optional_key(stem, track, sensor=row.get('sensor'), sequence=row.get('sequence'))
        if stem_key is not None and stem_key != full:
            raise ValueError(f'Frame identity mismatch: {row["frame_id"]} vs {stem}')
    return {**row, 'canonical_id': full.canonical(), 'sensor': full.sensor, 'sequence': full.sequence,
            'frame_index': full.frame, 'track': track, 'domain': 'synthetic'}


def _subtype(name):
    key = name.lower()
    if key in WALKERS: return 'walker'
    if key in RIDERS: return 'rider'
    return 'other'


def _raw_label_rows(path):
    assert_not_held_out(path)
    suffix = Path(path).suffix.lower()
    if suffix == '.txt':
        rows = []
        for line in Path(path).read_text().splitlines():
            if not line.strip(): continue
            fields = line.split()
            if len(fields) < 8: raise ValueError(f'Unrecognized synthetic labels: {path}')
            rows.append({'class': fields[7]})
        return rows
    if suffix == '.json':
        from .common import load_json
        rows = load_json(path)
        return [{'class': row.get('obj_type', row.get('class', ''))} for row in rows]
    raise ValueError(f'Unrecognized synthetic labels: {path}')


def _points_in_box(points, box):
    if len(points) == 0: return np.zeros(0, dtype=bool)
    center = np.asarray(box[:3], dtype=np.float64)
    radius = 0.5 * float(np.linalg.norm(np.asarray(box[3:6], dtype=np.float64))) + 1e-6
    near = np.linalg.norm(points[:, :3] - center, axis=1) <= radius
    mask = np.zeros(len(points), dtype=bool)
    chosen = np.flatnonzero(near)
    if len(chosen): mask[chosen] = inside_box(points[chosen], box)
    return mask


def _overlap_count(boxes):
    count = 0
    arrays = [np.asarray(box, dtype=np.float64) for box in boxes]
    for i, left in enumerate(arrays):
        for right in arrays[i + 1:]:
            reach = 0.5 * (np.linalg.norm(left[3:6]) + np.linalg.norm(right[3:6]))
            if np.linalg.norm(left[:3] - right[:3]) > reach: continue
            if iou3d(left, right) > 0: count += 1
    return count


def box_center_in_range(box, track):
    bounds = np.asarray(TRACKS[track]['range'], dtype=np.float64)
    center = np.asarray(box[:3], dtype=np.float64)
    return bool((center >= bounds[:3]).all() and (center < bounds[3:]).all())


def diagnose_frame(points, objects, track):
    """Point support, crop coverage, and center-versus-bottom z for one cloud."""
    if track not in TRACKS: raise ValueError(f'Unrecognized track: {track!r}')
    bounds = np.asarray(TRACKS[track]['range'], dtype=np.float64)
    cropped = crop(np.asarray(points, dtype=np.float32), bounds)
    per_class = {}
    dims = defaultdict(list)
    center_hits = bottom_hits = ambiguous = supported = in_range = yaw_outside = 0
    for obj in objects:
        box = np.asarray(obj['box'], dtype=np.float64)
        name = obj['class']
        stats = per_class.setdefault(name, {'boxes': 0, 'in_range': 0, 'supported': 0})
        stats['boxes'] += 1
        dims[name].append(box[3:6].tolist())
        if box[6] < -np.pi or box[6] > np.pi: yaw_outside += 1
        inside_range = box_center_in_range(box, track)
        if inside_range:
            in_range += 1
            stats['in_range'] += 1
        mask = _points_in_box(cropped, box) if len(cropped) else np.zeros(0, dtype=bool)
        count = int(mask.sum()) if len(cropped) else 0
        if count >= SUPPORT_MIN_POINTS:
            supported += 1
            stats['supported'] += 1
            median_z = float(np.median(cropped[mask, 2]))
            center_z = float(box[2])
            bottom_z = float(box[2] - box[5] / 2)
            if abs(median_z - center_z) + 0.05 < abs(median_z - bottom_z): center_hits += 1
            elif abs(median_z - bottom_z) + 0.05 < abs(median_z - center_z): bottom_hits += 1
            else: ambiguous += 1
    return {
        'points': int(len(points)), 'points_cropped': int(len(cropped)), 'boxes': len(objects),
        'boxes_in_range': in_range, 'boxes_supported': supported, 'z_closer_to_center': center_hits,
        'z_closer_to_bottom': bottom_hits, 'z_ambiguous': ambiguous, 'yaw_outside_pi': yaw_outside,
        'overlapping_pairs': _overlap_count([obj['box'] for obj in objects]),
        'per_class': per_class, 'class_dims': dict(dims),
    }


def _coverage(class_counts, track):
    coverage = {}
    missing = {}
    below = {}
    for split, minimum in MINIMUM_INSTANCES.items():
        counts = class_counts[split]
        coverage[split] = {name: int(counts[name]) for name in TRACKS[track]['classes']}
        missing[split] = [name for name in SCORED_CLASSES[track] if counts[name] == 0]
        below[split] = {name: int(counts[name]) for name in SCORED_CLASSES[track] if counts[name] < minimum}
    return coverage, missing, below


def _dimension_medians(samples):
    medians = {}
    for name, axes in samples.items():
        medians[name] = {axis: (None if not values else float(np.median(np.asarray(values, dtype=np.float64))))
                         for axis, values in axes.items()}
    return medians


def audit_records(records, track, verify_hash=True):
    """Audit a synthetic manifest. Identity checks finish before any cloud or label is opened."""
    if not isinstance(records, list): raise ValueError('Unrecognized synthetic manifest')
    bound_rows = []
    seen = {}
    for row in records:
        bound = bind_synthetic_row(row, track)
        if bound['canonical_id'] in seen:
            raise ValueError(f'Duplicate frame identity: {bound["canonical_id"]}')
        seen[bound['canonical_id']] = bound['path']
        bound_rows.append(bound)
    class_counts = {split: Counter() for split in ('train', 'locked_validation', 'purged_gap', 'unspecified')}
    in_range_counts = {split: Counter() for split in class_counts}
    raw_counts, unmapped, mapped, subtypes = Counter(), Counter(), Counter(), Counter()
    dim_samples = defaultdict(lambda: {'dx': [], 'dy': [], 'dz': []})
    frames, errors = [], []
    totals = Counter()
    for bound in bound_rows:
        split = bound.get('split') or 'unspecified'
        if split not in class_counts: split = 'unspecified'
        try:
            if verify_hash and bound.get('sha256') and digest(bound['path']) != bound['sha256']:
                raise ValueError(f'Content hash mismatch: {bound["path"]}')
            if verify_hash and bound.get('labels_sha256') and digest(bound['labels']) != bound['labels_sha256']:
                raise ValueError(f'Label hash mismatch: {bound["labels"]}')
            for item in _raw_label_rows(bound['labels']):
                raw_counts[item['class']] += 1
                mapped_name = label_class(item['class'], track)
                if mapped_name is None: unmapped[item['class']] += 1
                else:
                    mapped[mapped_name] += 1
                    if mapped_name == 'pedestrian': subtypes[_subtype(item['class'])] += 1
            points = load_points(bound['path'])
            objects = load_labels(bound['labels'], track)
            stats = diagnose_frame(points, objects, track)
        except ValueError as exc:
            if not str(exc).startswith(_FRAME_ERROR_PREFIXES): raise
            errors.append({'frame_id': bound['canonical_id'], 'error': str(exc)})
            continue
        for obj in objects:
            class_counts[split][obj['class']] += 1
            if box_center_in_range(obj['box'], track): in_range_counts[split][obj['class']] += 1
        for name, samples in stats.pop('class_dims').items():
            for dx, dy, dz in samples:
                dim_samples[name]['dx'].append(dx)
                dim_samples[name]['dy'].append(dy)
                dim_samples[name]['dz'].append(dz)
        for key in ('points', 'points_cropped', 'boxes', 'boxes_in_range', 'boxes_supported',
                    'z_closer_to_center', 'z_closer_to_bottom', 'z_ambiguous', 'yaw_outside_pi',
                    'overlapping_pairs'):
            totals[key] += stats[key]
        frames.append({'frame_id': bound['canonical_id'], 'sequence': bound['sequence'],
                       'frame_index': bound['frame_index'], 'sensor': bound['sensor'], 'split': split,
                       'points': stats['points'], 'points_cropped': stats['points_cropped'],
                       'boxes': stats['boxes'], 'boxes_in_range': stats['boxes_in_range'],
                       'boxes_supported': stats['boxes_supported'],
                       'z_closer_to_center': stats['z_closer_to_center'],
                       'z_closer_to_bottom': stats['z_closer_to_bottom'],
                       'overlapping_pairs': stats['overlapping_pairs']})
        if len(frames) % 100 == 0:
            print(f'audited {len(frames)} {track} frames', flush=True)
    coverage, missing, below = _coverage(class_counts, track)
    in_range_coverage, _, _ = _coverage(in_range_counts, track)
    points = int(totals['points'])
    if totals['z_closer_to_center'] > totals['z_closer_to_bottom']:
        interpretation = 'Interior point z is closer to the cuboid center than to the bottom face.'
    elif totals['z_closer_to_bottom'] > totals['z_closer_to_center']:
        interpretation = 'Interior point z is closer to the bottom face than to the cuboid center.'
    else:
        interpretation = 'Interior point z does not favor the center or the bottom face.'
    return {
        'track': track, 'official_inputs_read': False, 'frames_audited': len(frames),
        'frame_errors': errors, 'frames': frames,
        'class_coverage': coverage, 'in_range_class_coverage': in_range_coverage,
        'missing_classes': missing, 'below_minimum': below,
        'minimum_instances': MINIMUM_INSTANCES,
        'raw_class_counts': dict(raw_counts), 'mapped_class_counts': dict(mapped),
        'unmapped_class_counts': dict(unmapped), 'pedestrian_subtypes': dict(subtypes),
        'crop': {'points': points, 'points_cropped': int(totals['points_cropped']),
                 'fraction': None if not points else float(totals['points_cropped'] / points)},
        'geometry': {
            'box_origin': 'openpcdet_center', 'support_min_points': SUPPORT_MIN_POINTS,
            'boxes': int(totals['boxes']), 'boxes_in_range': int(totals['boxes_in_range']),
            'boxes_supported': int(totals['boxes_supported']),
            'z_closer_to_center': int(totals['z_closer_to_center']),
            'z_closer_to_bottom': int(totals['z_closer_to_bottom']),
            'z_ambiguous': int(totals['z_ambiguous']), 'z_interpretation': interpretation,
            'yaw_outside_pi': int(totals['yaw_outside_pi']),
            'overlapping_pairs': int(totals['overlapping_pairs']),
            'dimension_medians': _dimension_medians(dim_samples),
        },
    }
