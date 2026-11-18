"""Physics-guided refinement and joint subset search on permitted development data.

No held-out target input is used for fitting. Without a real development
manifest, --source-only exports a reproducible source-domain baseline and
--no-adapter preserves raw cropped point clouds without ray-binning or noise.
Neither mode claims a Sim2Real improvement.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from .common import (
    TRACKS, assert_not_held_out, audit_real_manifest, crop, digest,
    load_json, load_labels, load_points, save_json,
)
from .geometry import inside_box
from .metrics import realism, sample

IDENTITY_ORIGIN: tuple[float, float, float] = (0.0, 0.0, 0.0)
IDENTITY_ROTATION: tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
EXTRINSICS_MODES = frozenset({'sensor_to_cloud', 'cloud_to_sensor'})
SUPPORT_MIN_POINTS = 5
RANGE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ('0_20m', 0.0, 20.0),
    ('20_40m', 20.0, 40.0),
    ('40m_plus', 40.0, float('inf')),
)


def _normalize_mode(mode: str | None) -> str:
    if mode is None:
        return 'sensor_to_cloud'
    aliases = {
        'sensor_to_cloud': 'sensor_to_cloud',
        'sensor_to_world': 'sensor_to_cloud',
        'cloud_to_sensor': 'cloud_to_sensor',
        'world_to_sensor': 'cloud_to_sensor',
    }
    if not isinstance(mode, str) or mode.strip().lower() not in aliases:
        raise ValueError(f'Unsupported extrinsics_mode: {mode!r}')
    return aliases[mode.strip().lower()]


def _normalize_origin(origin) -> tuple[float, float, float]:
    arr = np.asarray(origin, dtype=np.float64)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError('sensor_origin must be a finite 3-element vector')
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _validate_rotation(matrix) -> np.ndarray:
    rot = np.asarray(matrix, dtype=np.float64)
    if rot.shape != (3, 3) or not np.isfinite(rot).all():
        raise ValueError('Extrinsics rotation must be a finite 3x3 matrix')
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rot), 1.0, atol=1e-4):
        raise ValueError('Extrinsics rotation must be a proper orthogonal 3x3 matrix (R^T R = I, det = +1)')
    return rot


def _normalize_extrinsics_matrix(extrinsics) -> tuple[tuple[float, ...], ...]:
    arr = np.asarray(extrinsics, dtype=np.float64)
    if arr.ndim != 2 or arr.shape not in ((3, 3), (3, 4), (4, 4)) or not np.isfinite(arr).all():
        raise ValueError('extrinsics must be a finite (3, 3), (3, 4), or (4, 4) matrix')
    if arr.shape == (4, 4) and not np.allclose(arr[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError('4x4 extrinsics must have bottom row [0, 0, 0, 1]')
    _validate_rotation(arr[:3, :3])
    return tuple(tuple(float(v) for v in row) for row in arr)


@dataclass(frozen=True)
class SensorPose:
    """Explicit sensor pose in the point cloud frame.

    ``origin`` is the 3D sensor position in cloud coordinates, and ``rotation``
    maps sensor-frame vectors into cloud-frame vectors:
    ``p_cloud = origin + p_sensor @ rotation.T`` and
    ``p_sensor = (p_cloud - origin) @ rotation``.
    """
    origin: tuple[float, float, float] = IDENTITY_ORIGIN
    rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = IDENTITY_ROTATION
    source: str = 'identity_default'
    mode: str = 'sensor_to_cloud'

    @property
    def is_identity(self) -> bool:
        return (
            np.allclose(self.origin, IDENTITY_ORIGIN, atol=1e-12)
            and np.allclose(self.rotation, IDENTITY_ROTATION, atol=1e-12)
        )

    def to_dict(self) -> dict:
        return {
            'origin': [float(v) for v in self.origin],
            'rotation': [[float(v) for v in row] for row in self.rotation],
            'source': self.source,
            'mode': self.mode,
            'is_identity': bool(self.is_identity),
        }


def resolve_sensor_pose(
    sensor_origin=None,
    extrinsics=None,
    extrinsics_mode: str | None = None,
    *,
    source: str | None = None,
) -> SensorPose:
    """Resolve and validate explicit sensor origin/extrinsics, or return identity default."""
    if isinstance(extrinsics, SensorPose):
        if sensor_origin is not None and not np.allclose(_normalize_origin(sensor_origin), extrinsics.origin, atol=1e-5):
            raise ValueError('Contradictory sensor_origin and SensorPose.origin')
        return extrinsics
    if isinstance(extrinsics, dict):
        dict_origin = extrinsics.get('sensor_origin', extrinsics.get('origin', extrinsics.get('translation')))
        if sensor_origin is not None and dict_origin is not None:
            if not np.allclose(_normalize_origin(sensor_origin), _normalize_origin(dict_origin), atol=1e-5):
                raise ValueError('Contradictory sensor_origin and extrinsics dict origin')
        merged_origin = sensor_origin if sensor_origin is not None else dict_origin
        matrix = extrinsics.get('extrinsics', extrinsics.get('matrix', extrinsics.get('pose', extrinsics.get('rotation'))))
        dict_mode = extrinsics.get('extrinsics_mode', extrinsics.get('mode', extrinsics.get('convention', extrinsics_mode)))
        dict_source = extrinsics.get('source', source or 'explicit_metadata')
        if merged_origin is None and matrix is None:
            raise ValueError('Extrinsics metadata dict must specify sensor_origin or extrinsics/rotation')
        return resolve_sensor_pose(merged_origin, matrix, dict_mode, source=dict_source)

    mode = _normalize_mode(extrinsics_mode)
    if sensor_origin is None and extrinsics is None:
        return SensorPose(IDENTITY_ORIGIN, IDENTITY_ROTATION, source or 'identity_default', mode)

    origin = None if sensor_origin is None else _normalize_origin(sensor_origin)
    rot_cloud = np.eye(3, dtype=np.float64)
    if extrinsics is not None:
        mat = np.asarray(_normalize_extrinsics_matrix(extrinsics), dtype=np.float64)
        rot_raw = mat[:3, :3]
        t_raw = mat[:3, 3] if mat.shape in ((3, 4), (4, 4)) else None
        if mode == 'sensor_to_cloud':
            rot_cloud = rot_raw
            ext_origin = None if t_raw is None else _normalize_origin(t_raw)
        else:
            rot_cloud = rot_raw.T
            ext_origin = None if t_raw is None else _normalize_origin(-rot_raw.T @ t_raw)
        if ext_origin is not None:
            if origin is not None and not np.allclose(origin, ext_origin, atol=1e-5):
                raise ValueError('Contradictory sensor_origin and extrinsics translation')
            origin = ext_origin

    if origin is None:
        origin = IDENTITY_ORIGIN
    rot_tuple = tuple(tuple(float(v) for v in row) for row in rot_cloud)
    resolved_source = source or ('explicit_extrinsics' if extrinsics is not None else 'explicit_sensor_origin')
    return SensorPose(origin, rot_tuple, resolved_source, mode)


@dataclass
class Physics:
    drop: float = 0.05
    range_drop: float = 0.10
    jitter: float = 0.005
    range_jitter: float = 0.0003
    azimuth_deg: float = 0.2
    ring_deg: float = 0.4
    z_shift: float = 0.0
    ground_jitter: float = 0.005
    max_points: int = 90_000
    no_adapter: bool = False
    sensor_origin: tuple[float, float, float] | None = None
    extrinsics: tuple[tuple[float, ...], ...] | None = None
    extrinsics_mode: str = 'sensor_to_cloud'

    def __post_init__(self):
        self.no_adapter = bool(self.no_adapter)
        self.extrinsics_mode = _normalize_mode(self.extrinsics_mode)
        if self.sensor_origin is not None:
            self.sensor_origin = _normalize_origin(self.sensor_origin)
        if self.extrinsics is not None:
            self.extrinsics = _normalize_extrinsics_matrix(self.extrinsics)
        if self.sensor_origin is not None or self.extrinsics is not None:
            resolve_sensor_pose(self.sensor_origin, self.extrinsics, self.extrinsics_mode)


def _active_pose(params: Physics, sensor_origin=None, extrinsics=None, extrinsics_mode: str | None = None, pose_source: str | None = None) -> SensorPose:
    origin = params.sensor_origin if sensor_origin is None else sensor_origin
    matrix = params.extrinsics if extrinsics is None else extrinsics
    mode = params.extrinsics_mode if extrinsics_mode is None else extrinsics_mode
    return resolve_sensor_pose(origin, matrix, mode, source=pose_source)


def profile(points, *, sensor_origin=None, extrinsics=None, extrinsics_mode: str | None = None):
    pose = resolve_sensor_pose(sensor_origin, extrinsics, extrinsics_mode)
    xyz = points[:, :3]
    if pose.is_identity:
        sensor_xyz = xyz
    else:
        origin = np.asarray(pose.origin, dtype=xyz.dtype)
        rot = np.asarray(pose.rotation, dtype=xyz.dtype)
        sensor_xyz = (xyz - origin) @ rot
    r = np.linalg.norm(sensor_xyz, axis=1)
    elev = np.degrees(np.arctan2(sensor_xyz[:, 2], np.linalg.norm(sensor_xyz[:, :2], axis=1)))
    result = {
        'points': len(points),
        'range_hist': np.histogram(r, np.linspace(0, 80, 17))[0].tolist(),
        'elevation_hist': np.histogram(elev, np.linspace(-90, 90, 361))[0].tolist(),
        'xyz_mean': xyz.mean(0).tolist(),
        'xyz_cov': np.cov(xyz.T).tolist(),
        'z_quantiles': np.quantile(xyz[:, 2], [.01, .1, .5, .9, .99]).tolist(),
    }
    if points.shape[1] == 4:
        result['intensity_quantiles'] = np.quantile(points[:, 3], np.linspace(0, 1, 101)).tolist()
    if not pose.is_identity:
        result['sensor_pose'] = pose.to_dict()
    return result


def refine(
    points,
    params: Physics | None = None,
    seed: int = 1234,
    intensity_reference=None,
    *,
    sensor_origin=None,
    extrinsics=None,
    extrinsics_mode: str | None = None,
    no_adapter: bool | None = None,
):
    if params is None:
        params = Physics(no_adapter=True if no_adapter is None else bool(no_adapter))
    p = np.asarray(points).copy()
    if p.ndim != 2 or p.shape[1] not in (3, 4) or not len(p) or not np.isfinite(p).all():
        raise ValueError('Point cloud must be a non-empty finite Nx3 or Nx4 array')
    if params.azimuth_deg <= 0 or params.ring_deg <= 0:
        raise ValueError('Angular bins must be positive')
    if params.max_points <= 0:
        raise ValueError('max_points must be positive')
    pose = _active_pose(params, sensor_origin=sensor_origin, extrinsics=extrinsics, extrinsics_mode=extrinsics_mode)
    skip_adapter = params.no_adapter if no_adapter is None else bool(no_adapter)
    if skip_adapter:
        return p.astype(np.float32)

    rng = np.random.default_rng(seed)
    xyz = p[:, :3]
    if pose.is_identity:
        sensor_xyz = xyz
        r = np.linalg.norm(xyz, axis=1)
    else:
        origin = np.asarray(pose.origin, dtype=xyz.dtype)
        rot = np.asarray(pose.rotation, dtype=xyz.dtype)
        rel_xyz = xyz - origin
        sensor_xyz = rel_xyz @ rot
        r = np.linalg.norm(sensor_xyz, axis=1)
    az = np.arctan2(sensor_xyz[:, 1], sensor_xyz[:, 0])
    el = np.arctan2(sensor_xyz[:, 2], np.linalg.norm(sensor_xyz[:, :2], axis=1))
    # First-return visibility, one nearest hit per angular ray cell in the active sensor frame.
    bins = np.column_stack([
        np.floor((az + np.pi) / np.radians(params.azimuth_deg)),
        np.round(el / np.radians(params.ring_deg)),
    ]).astype(np.int64)
    order = np.argsort(r, kind='stable')
    _, first = np.unique(bins[order], axis=0, return_index=True)
    keep = order[first]
    keep = keep[rng.random(len(keep)) >= np.clip(params.drop + params.range_drop * (r[keep] / 60) ** 2, 0, .95)]
    if not len(keep):
        raise ValueError('Adapter removed every point')
    p = p[keep]
    xyz = p[:, :3]
    rr = r[keep]
    # Radial range noise respects physical beam direction from sensor origin; small ground noise models roughness.
    if pose.is_identity:
        beam = xyz / np.maximum(rr[:, None], 1e-6)
    else:
        origin = np.asarray(pose.origin, dtype=xyz.dtype)
        beam = (xyz - origin) / np.maximum(rr[:, None], 1e-6)
    xyz += beam * rng.normal(0, params.jitter + params.range_jitter * rr)[:, None]
    ground = xyz[:, 2] < np.quantile(xyz[:, 2], .25)
    xyz[ground, 2] += rng.normal(0, params.ground_jitter, ground.sum())
    xyz[:, 2] += params.z_shift
    if p.shape[1] == 4 and intensity_reference is not None:
        q = np.linspace(0, 1, len(intensity_reference))
        source = np.quantile(p[:, 3], q)
        unique, indices = np.unique(source, return_index=True)
        p[:, 3] = np.interp(p[:, 3], unique, np.asarray(intensity_reference)[indices])
    if len(p) > params.max_points:
        p = p[rng.choice(len(p), params.max_points, replace=False)]
    return p.astype(np.float32)


def _points_in_box(points: np.ndarray, box) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    b = np.asarray(box, dtype=np.float64)
    center = b[:3]
    radius = 0.5 * float(np.linalg.norm(b[3:6])) + 1e-6
    near = np.linalg.norm(points[:, :3] - center, axis=1) <= radius
    mask = np.zeros(len(points), dtype=bool)
    chosen = np.flatnonzero(near)
    if len(chosen):
        mask[chosen] = inside_box(points[chosen], b)
    return mask


def _range_bucket_name(distance: float) -> str:
    for name, low, high in RANGE_BUCKETS:
        if low <= distance < high:
            return name
    return RANGE_BUCKETS[-1][0]


def _z_alignment_status(points_xyz: np.ndarray, box) -> str:
    median_z = float(np.median(points_xyz[:, 2]))
    center_z = float(box[2])
    bottom_z = float(box[2] - box[5] / 2.0)
    if abs(median_z - center_z) + 0.05 < abs(median_z - bottom_z):
        return 'center'
    if abs(median_z - bottom_z) + 0.05 < abs(median_z - center_z):
        return 'bottom'
    return 'ambiguous'


def diagnose_adaptation(
    raw_points,
    adapted_points,
    objects_before: list[dict],
    objects_after: list[dict] | None = None,
    track: str = 'v2x',
    *,
    sensor_pose: SensorPose | None = None,
    z_shift: float = 0.0,
    support_min_points: int = SUPPORT_MIN_POINTS,
) -> dict:
    """Compute per-object point support and geometric consistency diagnostics for one frame."""
    if track not in TRACKS:
        raise ValueError(f'Unrecognized track: {track!r}')
    if support_min_points <= 0:
        raise ValueError('support_min_points must be positive')
    pose = sensor_pose or SensorPose()
    raw = np.asarray(raw_points, dtype=np.float32)
    adapted = np.asarray(adapted_points, dtype=np.float32)
    bounds = np.asarray(TRACKS[track]['range'], dtype=np.float64)
    if objects_after is None:
        objects_after = [
            {'class': obj['class'], 'box': [
                float(obj['box'][0]), float(obj['box'][1]), float(obj['box'][2]) + float(z_shift),
                float(obj['box'][3]), float(obj['box'][4]), float(obj['box'][5]), float(obj['box'][6]),
            ]}
            for obj in objects_before
        ]
    if len(objects_before) != len(objects_after):
        raise ValueError('objects_before and objects_after must have the same length prior to bounds filtering')

    origin = np.asarray(pose.origin, dtype=np.float64)
    rot = np.asarray(pose.rotation, dtype=np.float64)
    raw_sensor = raw[:, :3] if pose.is_identity else (raw[:, :3] - origin) @ rot
    adapted_unshifted = adapted[:, :3].copy()
    if z_shift != 0.0:
        adapted_unshifted[:, 2] -= float(z_shift)
    adapted_sensor = adapted_unshifted if pose.is_identity else (adapted_unshifted - origin) @ rot

    bins = np.linspace(0.0, 80.0, 17)
    raw_hist = np.histogram(np.linalg.norm(raw_sensor, axis=1), bins)[0].astype(np.float64)
    adapted_hist = np.histogram(np.linalg.norm(adapted_sensor, axis=1), bins)[0].astype(np.float64)
    raw_density = raw_hist / max(raw_hist.sum(), 1.0)
    adapted_density = adapted_hist / max(adapted_hist.sum(), 1.0)
    range_hist_l1 = float(np.abs(raw_density - adapted_density).sum())

    exact_raw_preserved = bool(
        raw.shape == adapted.shape and np.array_equal(raw[:, :3], adapted[:, :3])
    )
    raw_in_bounds = float(((raw[:, :3] >= bounds[:3]).all(1) & (raw[:, :3] < bounds[3:]).all(1)).mean()) if len(raw) else 0.0
    adapted_in_bounds = float(((adapted[:, :3] >= bounds[:3]).all(1) & (adapted[:, :3] < bounds[3:]).all(1)).mean()) if len(adapted) else 0.0
    ground_q10_shift = (
        float(np.quantile(adapted_unshifted[:, 2], 0.10) - np.quantile(raw[:, :3][:, 2], 0.10))
        if len(raw) and len(adapted) else 0.0
    )

    per_object = []
    per_class: dict[str, dict] = {}
    per_range: dict[str, dict] = {
        name: {
            'boxes': 0, 'supported_before': 0, 'supported_after': 0,
            'support_retained': 0, 'support_lost': 0, 'zero_point_lost': 0,
            'points_before': 0, 'points_after': 0,
        }
        for name, _, _ in RANGE_BUCKETS
    }
    centroid_shifts = []
    z_residuals_before = []
    z_residuals_after = []
    z_status_before = {'center': 0, 'bottom': 0, 'ambiguous': 0}
    z_status_after = {'center': 0, 'bottom': 0, 'ambiguous': 0}

    supported_before_total = supported_after_total = support_retained_total = 0
    support_lost_total = zero_point_lost_total = nonempty_before_total = nonempty_after_total = 0
    in_range_before_total = in_range_after_total = 0

    for idx, (obj_b, obj_a) in enumerate(zip(objects_before, objects_after)):
        cls_name = obj_b['class']
        box_b = np.asarray(obj_b['box'], dtype=np.float64)
        box_a = np.asarray(obj_a['box'], dtype=np.float64)
        in_range_b = bool((box_b[:3] >= bounds[:3]).all() and (box_b[:3] < bounds[3:]).all())
        in_range_a = bool((box_a[:3] >= bounds[:3]).all() and (box_a[:3] < bounds[3:]).all())
        in_range_before_total += int(in_range_b)
        in_range_after_total += int(in_range_a)

        mask_b = _points_in_box(raw, box_b)
        mask_a = _points_in_box(adapted, box_a)
        cnt_b = int(mask_b.sum())
        cnt_a = int(mask_a.sum())
        sup_b = cnt_b >= support_min_points
        sup_a = cnt_a >= support_min_points
        retained = bool(sup_b and sup_a)
        lost = bool(sup_b and not sup_a)
        zero_lost = bool(cnt_b > 0 and cnt_a == 0)
        ratio = float(cnt_a / cnt_b) if cnt_b > 0 else None

        nonempty_before_total += int(cnt_b > 0)
        nonempty_after_total += int(cnt_a > 0)
        supported_before_total += int(sup_b)
        supported_after_total += int(sup_a)
        support_retained_total += int(retained)
        support_lost_total += int(lost)
        zero_point_lost_total += int(zero_lost)

        dist = float(np.linalg.norm(box_b[:3] - origin))
        bucket = _range_bucket_name(dist)
        rb = per_range[bucket]
        rb['boxes'] += 1
        rb['supported_before'] += int(sup_b)
        rb['supported_after'] += int(sup_a)
        rb['support_retained'] += int(retained)
        rb['support_lost'] += int(lost)
        rb['zero_point_lost'] += int(zero_lost)
        rb['points_before'] += cnt_b
        rb['points_after'] += cnt_a

        cs = per_class.setdefault(cls_name, {
            'boxes': 0, 'in_range_before': 0, 'in_range_after': 0,
            'nonempty_before': 0, 'nonempty_after': 0,
            'supported_before': 0, 'supported_after': 0,
            'support_retained': 0, 'support_lost': 0, 'zero_point_lost': 0,
            'points_before': 0, 'points_after': 0, '_ratios': [],
        })
        cs['boxes'] += 1
        cs['in_range_before'] += int(in_range_b)
        cs['in_range_after'] += int(in_range_a)
        cs['nonempty_before'] += int(cnt_b > 0)
        cs['nonempty_after'] += int(cnt_a > 0)
        cs['supported_before'] += int(sup_b)
        cs['supported_after'] += int(sup_a)
        cs['support_retained'] += int(retained)
        cs['support_lost'] += int(lost)
        cs['zero_point_lost'] += int(zero_lost)
        cs['points_before'] += cnt_b
        cs['points_after'] += cnt_a
        if ratio is not None:
            cs['_ratios'].append(ratio)

        centroid_shift = None
        if cnt_b > 0 and cnt_a > 0:
            pts_b = raw[mask_b, :3].astype(np.float64)
            pts_a_unshifted = adapted_unshifted[mask_a, :3].astype(np.float64)
            centroid_shift = float(np.linalg.norm(pts_a_unshifted.mean(0) - pts_b.mean(0)))
            centroid_shifts.append(centroid_shift)

        if sup_b:
            z_status_before[_z_alignment_status(raw[mask_b, :3], box_b)] += 1
            z_residuals_before.append(float(np.median(raw[mask_b, 2]) - box_b[2]))
        if sup_a:
            z_status_after[_z_alignment_status(adapted[mask_a, :3], box_a)] += 1
            z_residuals_after.append(float(np.median(adapted[mask_a, 2]) - box_a[2]))

        per_object.append({
            'index': idx,
            'class': cls_name,
            'range_m': dist,
            'range_bucket': bucket,
            'in_range_before': in_range_b,
            'in_range_after': in_range_a,
            'points_before': cnt_b,
            'points_after': cnt_a,
            'supported_before': sup_b,
            'supported_after': sup_a,
            'support_retained': retained,
            'support_lost': lost,
            'zero_point_lost': zero_lost,
            'point_retention_ratio': ratio,
            'centroid_shift_m': centroid_shift,
        })

    for stats in per_class.values():
        ratios = stats.pop('_ratios')
        stats['mean_retention_ratio'] = float(np.mean(ratios)) if ratios else None

    return {
        'track': track,
        'support_min_points': support_min_points,
        'points_before': int(len(raw)),
        'points_after': int(len(adapted)),
        'point_retention_fraction': float(len(adapted) / len(raw)) if len(raw) else 0.0,
        'boxes': len(objects_before),
        'boxes_in_range_before': in_range_before_total,
        'boxes_in_range_after': in_range_after_total,
        'boxes_nonempty_before': nonempty_before_total,
        'boxes_nonempty_after': nonempty_after_total,
        'boxes_supported_before': supported_before_total,
        'boxes_supported_after': supported_after_total,
        'boxes_support_retained': support_retained_total,
        'boxes_support_lost': support_lost_total,
        'boxes_zero_point_lost': zero_point_lost_total,
        'support_retention_rate': (
            float(support_retained_total / supported_before_total) if supported_before_total else None
        ),
        'per_class': per_class,
        'per_range_bucket': per_range,
        'per_object': per_object,
        'geometric_consistency': {
            'exact_raw_crop_preserved': exact_raw_preserved,
            'box_z_shift_applied': float(z_shift),
            'in_bounds_fraction_before': raw_in_bounds,
            'in_bounds_fraction_after': adapted_in_bounds,
            'ground_z_q10_shift_m': ground_q10_shift,
            'range_histogram_l1': range_hist_l1,
            'mean_interior_centroid_shift_m': float(np.mean(centroid_shifts)) if centroid_shifts else 0.0,
            'max_interior_centroid_shift_m': float(np.max(centroid_shifts)) if centroid_shifts else 0.0,
            'median_z_residual_before_m': float(np.median(z_residuals_before)) if z_residuals_before else None,
            'median_z_residual_after_m': float(np.median(z_residuals_after)) if z_residuals_after else None,
            'z_alignment_before': z_status_before,
            'z_alignment_after': z_status_after,
            'sensor_pose': pose.to_dict(),
        },
    }


def summarize_adaptation(frame_diagnostics: list[dict], track: str = 'v2x', support_min_points: int = SUPPORT_MIN_POINTS) -> dict:
    """Aggregate frame-level adaptation diagnostics across a synthetic split or corpus."""
    if track not in TRACKS:
        raise ValueError(f'Unrecognized track: {track!r}')
    points_before = sum(int(d['points_before']) for d in frame_diagnostics)
    points_after = sum(int(d['points_after']) for d in frame_diagnostics)
    boxes = sum(int(d['boxes']) for d in frame_diagnostics)
    in_range_before = sum(int(d['boxes_in_range_before']) for d in frame_diagnostics)
    in_range_after = sum(int(d['boxes_in_range_after']) for d in frame_diagnostics)
    nonempty_before = sum(int(d['boxes_nonempty_before']) for d in frame_diagnostics)
    nonempty_after = sum(int(d['boxes_nonempty_after']) for d in frame_diagnostics)
    sup_before = sum(int(d['boxes_supported_before']) for d in frame_diagnostics)
    sup_after = sum(int(d['boxes_supported_after']) for d in frame_diagnostics)
    retained = sum(int(d['boxes_support_retained']) for d in frame_diagnostics)
    lost = sum(int(d['boxes_support_lost']) for d in frame_diagnostics)
    zero_lost = sum(int(d['boxes_zero_point_lost']) for d in frame_diagnostics)

    per_class: dict[str, dict] = {}
    for cls_name in TRACKS[track]['classes']:
        per_class[cls_name] = {
            'boxes': 0, 'in_range_before': 0, 'in_range_after': 0,
            'nonempty_before': 0, 'nonempty_after': 0,
            'supported_before': 0, 'supported_after': 0,
            'support_retained': 0, 'support_lost': 0, 'zero_point_lost': 0,
            'points_before': 0, 'points_after': 0,
        }
    per_range: dict[str, dict] = {
        name: {
            'boxes': 0, 'supported_before': 0, 'supported_after': 0,
            'support_retained': 0, 'support_lost': 0, 'zero_point_lost': 0,
            'points_before': 0, 'points_after': 0,
        }
        for name, _, _ in RANGE_BUCKETS
    }
    z_before = {'center': 0, 'bottom': 0, 'ambiguous': 0}
    z_after = {'center': 0, 'bottom': 0, 'ambiguous': 0}
    pose_sources = sorted({d['geometric_consistency']['sensor_pose']['source'] for d in frame_diagnostics})
    all_raw_preserved = bool(frame_diagnostics) and all(
        d['geometric_consistency']['exact_raw_crop_preserved'] for d in frame_diagnostics
    )

    for d in frame_diagnostics:
        for cls_name, stats in d['per_class'].items():
            target = per_class.setdefault(cls_name, {
                'boxes': 0, 'in_range_before': 0, 'in_range_after': 0,
                'nonempty_before': 0, 'nonempty_after': 0,
                'supported_before': 0, 'supported_after': 0,
                'support_retained': 0, 'support_lost': 0, 'zero_point_lost': 0,
                'points_before': 0, 'points_after': 0,
            })
            for k in target:
                target[k] += int(stats[k])
        for bucket_name, stats in d['per_range_bucket'].items():
            for k in per_range[bucket_name]:
                per_range[bucket_name][k] += int(stats[k])
        for k in ('center', 'bottom', 'ambiguous'):
            z_before[k] += int(d['geometric_consistency']['z_alignment_before'][k])
            z_after[k] += int(d['geometric_consistency']['z_alignment_after'][k])

    for stats in per_class.values():
        stats['support_retention_rate'] = (
            float(stats['support_retained'] / stats['supported_before']) if stats['supported_before'] else None
        )
        stats['point_retention_fraction'] = (
            float(stats['points_after'] / stats['points_before']) if stats['points_before'] else None
        )
    for stats in per_range.values():
        stats['support_retention_rate'] = (
            float(stats['support_retained'] / stats['supported_before']) if stats['supported_before'] else None
        )
        stats['point_retention_fraction'] = (
            float(stats['points_after'] / stats['points_before']) if stats['points_before'] else None
        )

    mean_l1 = (
        float(np.mean([d['geometric_consistency']['range_histogram_l1'] for d in frame_diagnostics]))
        if frame_diagnostics else 0.0
    )
    mean_centroid_shift = (
        float(np.mean([d['geometric_consistency']['mean_interior_centroid_shift_m'] for d in frame_diagnostics]))
        if frame_diagnostics else 0.0
    )
    max_centroid_shift = (
        float(np.max([d['geometric_consistency']['max_interior_centroid_shift_m'] for d in frame_diagnostics]))
        if frame_diagnostics else 0.0
    )
    mean_ground_shift = (
        float(np.mean([d['geometric_consistency']['ground_z_q10_shift_m'] for d in frame_diagnostics]))
        if frame_diagnostics else 0.0
    )

    return {
        'track': track,
        'frames': len(frame_diagnostics),
        'support_min_points': support_min_points,
        'points_before': points_before,
        'points_after': points_after,
        'point_retention_fraction': float(points_after / points_before) if points_before else 0.0,
        'boxes': boxes,
        'boxes_in_range_before': in_range_before,
        'boxes_in_range_after': in_range_after,
        'boxes_nonempty_before': nonempty_before,
        'boxes_nonempty_after': nonempty_after,
        'boxes_supported_before': sup_before,
        'boxes_supported_after': sup_after,
        'boxes_support_retained': retained,
        'boxes_support_lost': lost,
        'boxes_zero_point_lost': zero_lost,
        'support_retention_rate': float(retained / sup_before) if sup_before else None,
        'per_class': per_class,
        'per_range_bucket': per_range,
        'geometric_consistency': {
            'exact_raw_crop_preserved': all_raw_preserved,
            'sensor_pose_sources': pose_sources,
            'mean_range_histogram_l1': mean_l1,
            'mean_interior_centroid_shift_m': mean_centroid_shift,
            'max_interior_centroid_shift_m': max_centroid_shift,
            'mean_ground_z_q10_shift_m': mean_ground_shift,
            'z_alignment_before': z_before,
            'z_alignment_after': z_after,
        },
    }


def load_sensor_metadata(path: Path) -> dict:
    """Load explicit sensor origin/extrinsics metadata JSON without inventing values."""
    assert_not_held_out(path)
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError('Sensor extrinsics metadata must be a JSON object')
    return data


def _pose_for_record(
    row: dict,
    params: Physics,
    cli_origin=None,
    cli_extrinsics_map: dict | None = None,
    cli_extrinsics_mode: str | None = None,
) -> SensorPose:
    mode = cli_extrinsics_mode or row.get('extrinsics_mode') or params.extrinsics_mode
    if cli_extrinsics_map is not None:
        if any(k in cli_extrinsics_map for k in ('sensor_origin', 'origin', 'translation', 'extrinsics', 'matrix', 'pose', 'rotation')):
            return resolve_sensor_pose(cli_origin, cli_extrinsics_map, mode, source='explicit_metadata_file')
        frames_map = cli_extrinsics_map.get('frames', cli_extrinsics_map)
        sensors_map = cli_extrinsics_map.get('sensors', {})
        sequences_map = cli_extrinsics_map.get('sequences', {})
        entry = None
        for key in (row.get('canonical_id'), row.get('frame_id')):
            if key is not None and key in frames_map:
                entry = frames_map[key]
                break
        if entry is None and row.get('sensor') is not None and row['sensor'] in sensors_map:
            entry = sensors_map[row['sensor']]
        if entry is None and row.get('sequence') is not None and str(row['sequence']) in sequences_map:
            entry = sequences_map[str(row['sequence'])]
        if entry is None and 'default' in cli_extrinsics_map:
            entry = cli_extrinsics_map['default']
        if entry is None:
            raise ValueError(f"Missing explicit sensor extrinsics entry for frame {row.get('frame_id')!r}")
        return resolve_sensor_pose(cli_origin, entry, mode, source='explicit_metadata_file')

    if cli_origin is not None:
        return resolve_sensor_pose(cli_origin, params.extrinsics, mode, source='explicit_cli')
    row_origin = row.get('sensor_origin')
    row_ext = row.get('extrinsics', row.get('sensor_extrinsics'))
    if row_origin is not None or row_ext is not None:
        return resolve_sensor_pose(row_origin, row_ext, mode, source='manifest_row')
    if params.sensor_origin is not None or params.extrinsics is not None:
        return resolve_sensor_pose(params.sensor_origin, params.extrinsics, mode, source='physics_params')
    return SensorPose(IDENTITY_ORIGIN, IDENTITY_ROTATION, 'identity_default', _normalize_mode(mode))


def source_records(root, track):
    if track not in TRACKS:
        raise ValueError(f'Unrecognized track: {track!r}')
    root_path = Path(root)
    assert_not_held_out(root_path)
    if root_path.is_file() and root_path.suffix.lower() == '.json':
        rows = load_json(root_path)
        if not isinstance(rows, list) or not rows:
            raise ValueError('Expected non-empty synthetic manifest JSON list')
        result = []
        for row in rows:
            if not isinstance(row, dict) or 'frame_id' not in row or 'path' not in row or 'labels' not in row:
                raise ValueError('Unrecognized synthetic manifest row')
            if row.get('domain') != 'synthetic':
                raise ValueError('Only synthetic domain records are permitted in source_records')
            cloud_path = Path(row['path'])
            label_path = Path(row['labels'])
            assert_not_held_out(cloud_path)
            assert_not_held_out(label_path)
            if not cloud_path.exists():
                raise ValueError(f'Missing synthetic cloud: {cloud_path}')
            if not label_path.exists():
                raise ValueError(f'Missing synthetic labels: {label_path}')
            if row.get('sha256') and row['sha256']!=digest(cloud_path):
                raise ValueError(f"Synthetic cloud hash mismatch: {row['frame_id']}")
            if row.get('labels_sha256') and row['labels_sha256']!=digest(label_path):
                raise ValueError(f"Synthetic label hash mismatch: {row['frame_id']}")
            rec = {
                **row,
                'frame_id': str(row['frame_id']),
                'path': str(cloud_path.resolve()),
                'labels': str(label_path.resolve()),
                'domain': 'synthetic',
                'sha256': row.get('sha256') or digest(cloud_path),
            }
            result.append(rec)
        return result

    files = sorted(root_path.rglob('pcd/*.npy'))
    if not files:
        raise ValueError('Expected extracted UrbanTwin pcd/*.npy files')
    result = []
    for p in files:
        assert_not_held_out(p)
        label = p.parent.parent / 'lbl' / f'{p.stem}.txt'
        assert_not_held_out(label)
        if not label.exists():
            raise ValueError(f'Missing synthetic labels: {label}')
        result.append({
            'frame_id': p.stem,
            'path': str(p.resolve()),
            'labels': str(label.resolve()),
            'domain': 'synthetic',
            'sha256': digest(p),
        })
        import re
        sequences=[re.fullmatch(r'.*-Seq-(\d+)',part,re.IGNORECASE) for part in p.parts]
        sequence=next((int(match.group(1)) for match in sequences if match),None)
        if sequence is not None:result[-1]['sequence']=sequence
    return result


def run_synthetic_ablation(
    records: list[dict],
    track: str,
    candidates: dict[str, Physics] | None = None,
    *,
    seed: int = 1234,
    support_min_points: int = SUPPORT_MIN_POINTS,
    max_frames: int | None = None,
    sensor_metadata: dict | None = None,
) -> dict:
    """Compare no-adapter, source-only ray-binning, and physics/pose candidates on synthetic frames."""
    if track not in TRACKS:
        raise ValueError(f'Unrecognized track: {track!r}')
    if not records:
        raise ValueError('Synthetic ablation requires at least one synthetic record')
    if candidates is None:
        candidates = {
            'no_adapter': Physics(no_adapter=True),
            'source_only': Physics(drop=0, range_drop=0, jitter=0, range_jitter=0, ground_jitter=0),
            'default_physics': Physics(),
        }
    selected_records = records[:max_frames] if max_frames is not None else list(records)
    bounds = TRACKS[track]['range']
    prepared = []
    for row in selected_records:
        if row.get('domain') != 'synthetic':
            raise ValueError('Synthetic ablation accepts synthetic records only')
        assert_not_held_out(row['path'])
        assert_not_held_out(row['labels'])
        raw_cloud = crop(load_points(row['path']), bounds)
        raw_labels = load_labels(row['labels'], track)
        prepared.append((row, raw_cloud, raw_labels))

    results = {}
    for name, params in candidates.items():
        frame_diags = []
        for i, (row, raw_cloud, raw_labels) in enumerate(prepared):
            pose = _pose_for_record(row, params, cli_extrinsics_map=sensor_metadata)
            adapted = refine(
                raw_cloud,
                params,
                seed + i,
                sensor_origin=pose.origin,
                extrinsics=pose.rotation,
                extrinsics_mode='sensor_to_cloud',
                no_adapter=params.no_adapter,
            )
            effective_z_shift = 0.0 if params.no_adapter else float(params.z_shift)
            shifted_labels = [
                {
                    'class': obj['class'],
                    'box': [
                        obj['box'][0], obj['box'][1], obj['box'][2] + effective_z_shift,
                        obj['box'][3], obj['box'][4], obj['box'][5], obj['box'][6],
                    ],
                }
                for obj in raw_labels
            ]
            diag = diagnose_adaptation(
                raw_cloud,
                adapted,
                raw_labels,
                shifted_labels,
                track,
                sensor_pose=pose,
                z_shift=effective_z_shift,
                support_min_points=support_min_points,
            )
            diag['frame_id'] = row['frame_id']
            frame_diags.append(diag)
        summary = summarize_adaptation(frame_diags, track, support_min_points=support_min_points)
        results[name] = {
            'parameters': asdict(params),
            'summary': summary,
            'frames': [
                {k: v for k, v in d.items() if k != 'per_object'}
                for d in frame_diags
            ],
        }
    return {
        'domain': 'synthetic_adapter_ablation',
        'official': False,
        'track': track,
        'frames_evaluated': len(prepared),
        'support_min_points': support_min_points,
        'seed': seed,
        'candidates': results,
        'warning': 'Synthetic geometry and object-support diagnostic only; no real target data or server score is claimed.',
    }


def objective(p, q, baseline):
    m = realism(p, q, n=768, emd_n=192)
    # Development-only dimensionless loss, not organizer normalization endpoints.
    keys = ['CD', 'MMD2', 'EMD_W1_proxy', 'FPD']
    return float(np.mean([m[k] / max(abs(baseline[k]), 1e-5) for k in keys])), m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--track', choices=TRACKS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--real-manifest', type=Path)
    parser.add_argument('--forbidden', type=Path)
    parser.add_argument('--source-only', action='store_true')
    parser.add_argument('--no-adapter', action='store_true', help='Preserve raw cropped clouds without ray-binning, drop, or noise')
    parser.add_argument('--sensor-origin', nargs=3, type=float, metavar=('X', 'Y', 'Z'), help='Explicit sensor origin in cloud coordinates')
    parser.add_argument('--sensor-extrinsics', type=Path, help='Optional JSON file with explicit sensor origin/extrinsics metadata')
    parser.add_argument('--extrinsics-mode', choices=sorted(EXTRINSICS_MODES), default='sensor_to_cloud')
    parser.add_argument('--ablation-output', type=Path, help='Optional JSON path to write multi-candidate synthetic ablation report')
    parser.add_argument('--ablation-only', action='store_true', help='Run synthetic ablation diagnostics only without exporting clouds')
    parser.add_argument('--ablation-max-frames', type=int, help='Optional frame cap for --ablation-output / --ablation-only')
    parser.add_argument('--trials', type=int, default=12)
    args = parser.parse_args()
    assert_not_held_out(args.source)
    assert_not_held_out(args.output)
    rng = np.random.default_rng(1234)
    records = source_records(args.source, args.track)
    if args.count<1 or (not args.ablation_only and len(records)<args.count):
        raise ValueError('Requested synthetic count must be positive and cannot exceed distinct source frames')
    sensor_meta = load_sensor_metadata(args.sensor_extrinsics) if args.sensor_extrinsics else None
    cli_origin = _normalize_origin(args.sensor_origin) if args.sensor_origin is not None else None

    if args.ablation_output or args.ablation_only:
        ablation_path = args.ablation_output or (args.output / 'adapter_ablation_report.json')
        assert_not_held_out(ablation_path)
        ablation_candidates = {
            'no_adapter': Physics(
                drop=0, range_drop=0, jitter=0, range_jitter=0, ground_jitter=0,
                no_adapter=True, sensor_origin=cli_origin, extrinsics_mode=args.extrinsics_mode,
            ),
            'source_only': Physics(
                drop=0, range_drop=0, jitter=0, range_jitter=0, ground_jitter=0,
                no_adapter=False, sensor_origin=cli_origin, extrinsics_mode=args.extrinsics_mode,
            ),
            'default_physics': Physics(
                no_adapter=False, sensor_origin=cli_origin, extrinsics_mode=args.extrinsics_mode,
            ),
        }
        ablation = run_synthetic_ablation(
            records,
            args.track,
            ablation_candidates,
            seed=1234,
            max_frames=args.ablation_max_frames,
            sensor_metadata=sensor_meta,
        )
        save_json(ablation_path, ablation)
        if args.ablation_only:
            print(f'Saved synthetic adapter ablation report for {ablation["frames_evaluated"]} frames to {ablation_path}')
            return

    if len(records) < args.count:
        raise ValueError('Not enough unique source frames')
    bounds = TRACKS[args.track]['range']
    clouds = [crop(load_points(r['path']), bounds) for r in records]
    pooled = np.concatenate([sample(p, 2000, rng) for p in clouds])
    save_json(args.output / 'source_profile.json', profile(pooled, sensor_origin=cli_origin, extrinsics_mode=args.extrinsics_mode))
    target = None
    intensity = None
    if args.no_adapter:
        params = Physics(
            drop=0, range_drop=0, jitter=0, range_jitter=0, ground_jitter=0,
            no_adapter=True, sensor_origin=cli_origin, extrinsics_mode=args.extrinsics_mode,
        )
    elif args.source_only:
        params = Physics(
            drop=0, range_drop=0, jitter=0, range_jitter=0, ground_jitter=0,
            sensor_origin=cli_origin, extrinsics_mode=args.extrinsics_mode,
        )
    else:
        if not args.real_manifest or not args.forbidden:
            raise ValueError('Supply permitted real development manifest and official forbidden list, or use --source-only / --no-adapter')
        real_rows = audit_real_manifest(args.real_manifest, args.forbidden)
        real = [crop(load_points(r['path'], r.get('channels', 4)), bounds) for r in real_rows]
        target = np.concatenate([sample(p, 2000, rng) for p in real])
        real_profile = profile(np.concatenate(real))
        save_json(args.output / 'real_development_profile.json', real_profile)
        intensity = real_profile.get('intensity_quantiles')
        params = Physics(sensor_origin=cli_origin, extrinsics_mode=args.extrinsics_mode)
    poses = [
        _pose_for_record(row, params, cli_origin=cli_origin, cli_extrinsics_map=sensor_meta, cli_extrinsics_mode=args.extrinsics_mode)
        for row in records
    ]
    selected = np.linspace(0, len(records) - 1, args.count, dtype=int)
    report = {
        'official': False,
        'source_only': bool(args.source_only or args.no_adapter),
        'no_adapter': bool(params.no_adapter),
        'selected_ids': [records[i]['frame_id'] for i in selected],
        'development_metrics': None,
        'trials': [],
        'warning': 'No claim about held-out scores. Target frames and target labels remain unused.',
    }
    if target is not None and not params.no_adapter:
        baseline = realism(pooled, target, n=768, emd_n=192)
        report['before'] = baseline
        best = float('inf')
        best_params = params
        best_selection = selected
        for trial in range(args.trials):
            candidate = Physics(
                drop=float(rng.uniform(0, .15)),
                range_drop=float(rng.uniform(0, .3)),
                jitter=float(rng.uniform(0, .02)),
                range_jitter=float(rng.uniform(0, .0008)),
                azimuth_deg=float(rng.choice([.1, .2, .3])),
                ring_deg=float(rng.choice([.2, .4, .6])),
                sensor_origin=cli_origin,
                extrinsics_mode=args.extrinsics_mode,
            )
            indices = selected.copy()
            if trial and len(records) > args.count:
                available = np.setdiff1d(np.arange(len(records)), indices)
                indices[rng.integers(len(indices))] = rng.choice(available)
            adapted = [
                refine(
                    clouds[i],
                    candidate,
                    1234 + i,
                    sensor_origin=poses[i].origin,
                    extrinsics=poses[i].rotation,
                    extrinsics_mode='sensor_to_cloud',
                )
                for i in indices
            ]
            merged = np.concatenate([sample(p, 1000, rng) for p in adapted])
            loss, metrics = objective(merged, target, baseline)
            report['trials'].append({'trial': trial, 'loss': loss, 'parameters': asdict(candidate), 'metrics': metrics})
            if loss < best:
                best = loss
                best_params = candidate
                best_selection = indices.copy()
                selected = indices.copy()
        params = best_params
        selected = best_selection
    # Export all adapted synthetic training frames, not only the 50 submitted clouds.
    manifest = []
    selected_set = set(map(int, selected))
    syn = args.output / 'synthetic'
    syn.mkdir(parents=True, exist_ok=True)
    merged_after = []
    frame_diagnostics = []
    effective_z_shift = 0.0 if params.no_adapter else float(params.z_shift)
    for i, (row, p, pose) in enumerate(zip(records, clouds, poses)):
        adapted = refine(
            p,
            params,
            1234 + i,
            intensity,
            sensor_origin=pose.origin,
            extrinsics=pose.rotation,
            extrinsics_mode='sensor_to_cloud',
            no_adapter=params.no_adapter,
        )
        raw_labels = load_labels(row['labels'], args.track)
        labels = [
            {
                'class': obj['class'],
                'box': [
                    obj['box'][0], obj['box'][1], obj['box'][2] + effective_z_shift,
                    obj['box'][3], obj['box'][4], obj['box'][5], obj['box'][6],
                ],
            }
            for obj in raw_labels
        ]
        diag = diagnose_adaptation(
            p,
            adapted,
            raw_labels,
            labels,
            args.track,
            sensor_pose=pose,
            z_shift=effective_z_shift,
        )
        diag['frame_id'] = row['frame_id']
        frame_diagnostics.append(diag)
        labels = [o for o in labels if all(bounds[j] <= o['box'][j] < bounds[j + 3] for j in range(3))]
        dst = args.output / 'training' / f'{row["frame_id"]}.npy'
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, adapted)
        lab = dst.with_suffix('.json')
        save_json(lab, labels)
        entry = {
            'frame_id': row['frame_id'],
            'path': str(dst.resolve()),
            'labels': str(lab.resolve()),
            'domain': 'synthetic',
            'source_sha256': row['sha256'],
            'sha256': digest(dst),
            'labels_sha256': digest(lab),
            'track': args.track,
            'no_adapter': bool(params.no_adapter),
            'sensor_pose_source': pose.source,
        }
        for passthrough in ('canonical_id', 'sequence', 'frame_index', 'sensor', 'split', 'split_rule', 'split_cut', 'split_spacing', 'provenance'):
            if passthrough in row:
                entry[passthrough] = row[passthrough]
        manifest.append(entry)
        if i in selected_set:
            out = syn / f'{list(map(int, selected)).index(i) + 1:04d}.bin'
            adapted[:, :3].astype('<f4').tofile(out)
            merged_after.append(sample(adapted, 1000, rng))
    report['parameters'] = asdict(params)
    report['selected_ids'] = [records[i]['frame_id'] for i in selected]
    report['sensor_pose_summary'] = {
        'sources': sorted({pose.source for pose in poses}),
        'all_identity': all(pose.is_identity for pose in poses),
        'first_pose': poses[0].to_dict() if poses else SensorPose().to_dict(),
    }
    report['adaptation_diagnostics'] = summarize_adaptation(frame_diagnostics, args.track)
    if target is not None:
        report['development_metrics'] = realism(np.concatenate(merged_after), target)
    save_json(args.output / 'manifest.json', manifest)
    save_json(args.output / 'refinement_report.json', report)
    print(f'Exported {len(manifest)} adapted training frames and {len(selected)} submission clouds. source_only={args.source_only} no_adapter={params.no_adapter}')


if __name__ == '__main__':
    main()

