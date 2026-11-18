from __future__ import annotations
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np

TRACKS = {
    'lumpi': {'range': [-40,-40,-5,40,40,5], 'classes': ['Person','Car','Bicycle','Motorcycle','Bus','Truck','Van','Unknown']},
    'v2x': {'range': [-40,-40,-8,40,40,2], 'classes': ['vehicle','pedestrian','truck']},
}

def digest(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def crop(points, bounds):
    b = np.asarray(bounds)
    return points[np.isfinite(points).all(1) & (points[:,:3] >= b[:3]).all(1) & (points[:,:3] < b[3:]).all(1)]

def crop_labels(objects,bounds):
    """Keep typed boxes whose centers lie in the detector's half-open track crop."""
    return [obj for obj in objects if all(bounds[i]<=obj['box'][i]<bounds[i+3] for i in range(3))]

def load_points(path, channels=4):
    """Require explicit channel count for raw binaries, avoiding ambiguous N x 3/4 inference."""
    path = Path(path)
    if path.suffix == '.npy': p = np.load(path, allow_pickle=False)
    elif path.suffix == '.bin': p = np.fromfile(path, dtype='<f4').reshape(-1, channels)
    elif path.suffix == '.pcd': p = load_pcd(path)
    else: raise ValueError(f'Unsupported cloud format: {path}')
    if p.ndim != 2 or p.shape[1] not in (3,4) or not len(p): raise ValueError(f'Invalid cloud: {path}')
    if not np.isfinite(p).all(): raise ValueError(f'Nonfinite point values: {path}')
    return np.asarray(p, dtype=np.float32)

def load_pcd(path):
    with open(path,'rb') as f:
        header = {}
        while True:
            line = f.readline()
            if not line: raise ValueError('PCD DATA header missing')
            line = line.decode('ascii').strip()
            if not line or line.startswith('#'): continue
            k, *v = line.split(); header[k.upper()] = v
            if k.upper() == 'DATA': break
        fields = header['FIELDS']; count = list(map(int,header.get('COUNT',['1']*len(fields))))
        if any(c != 1 for c in count): raise ValueError('PCD vector fields need explicit conversion')
        mode = header['DATA'][0]
        if mode == 'ascii': values = np.loadtxt(f, ndmin=2)
        elif mode == 'binary':
            kinds = {'F':'f','U':'u','I':'i'}
            dtype = np.dtype([(n,'<'+kinds[t]+s) for n,t,s in zip(fields,header['TYPE'],header['SIZE'])])
            a = np.frombuffer(f.read(),dtype=dtype)
            values = np.column_stack([a[n] for n in fields])
        else: raise ValueError('Compressed PCD is unsupported; convert with LiGuard/Open3D first')
    selected = [fields.index(n) for n in ['x','y','z']]
    if 'intensity' in fields: selected.append(fields.index('intensity'))
    return values[:,selected].astype(np.float32)

# Directory and file names that training, calibration, and selection must not open.
# Official inference is a separate path and does not use this guard.
HELD_OUT_PATH_PARTS = frozenset({
    'detection_test_frames', 'realism_reference', 'reference_data', 'reference_data_dummy',
    'v2x_real_starting_kit', 'lumpi_starting_kit',
})
HELD_OUT_FILENAMES = frozenset({'detection_gt.json'})
_LABEL_SUFFIXES = frozenset({'.json', '.txt', '.label', '.csv'})
_SENSOR = re.compile(r'[A-Za-z][A-Za-z0-9_-]{0,31}')
_DIGITS = re.compile(r'\d{1,9}')

class FrameIdError(ValueError):
    """The token is not a frame identity. Callers must not try a second parsing."""

class FrameContextError(FrameIdError):
    """Sensor or sequence context contradicts the frame token."""

@dataclass(frozen=True)
class FrameKey:
    """One real or synthetic frame. Equality includes sensor and sequence, so a
    different sensor or a longer numeric token is not the same identity."""
    track: str
    frame: int
    sequence: int | None = None
    sensor: str | None = None

    def canonical(self):
        if self.track == 'v2x':
            if self.sequence is None: raise FrameContextError('V2X identity requires a sequence')
            body = f'{self.sequence}_{self.frame:06d}'
        elif self.sequence is None:
            body = f'{self.frame:06d}'
        else:
            body = f'{self.sequence}_{self.frame:06d}'
        return f'{self.sensor}:{body}' if self.sensor else body

class AuditedRows(list):
    """Rows that have already passed the real-data boundary. Do not rebuild them."""
    def __init__(self, rows, track, forbidden_digest):
        super().__init__(rows)
        self.track = track
        self.forbidden_digest = forbidden_digest

def held_out_input(path):
    raw = Path(path)
    if raw.name.lower() in HELD_OUT_FILENAMES: return True
    return any(part.lower() in HELD_OUT_PATH_PARTS for part in raw.parts)

def assert_not_held_out(path):
    if held_out_input(path): raise ValueError(f'Held-out cloud path: {path}')

def require_audited(rows):
    if not isinstance(rows, AuditedRows):
        raise ValueError('Real input must be validated once at the dataset boundary')
    return rows

def _as_text(value):
    if isinstance(value, bool) or isinstance(value, float) or value is None:
        raise FrameIdError(f'Unrecognized frame id: {value!r}')
    if isinstance(value, int): return str(value)
    if isinstance(value, str):
        text = value.strip()
        if text and not any(ch.isspace() for ch in text): return text
    raise FrameIdError(f'Unrecognized frame id: {value!r}')

def _optional_int(value, name):
    if value is None or value == '': return None
    if isinstance(value, bool) or isinstance(value, float):
        raise FrameContextError(f'Unrecognized {name} context: {value!r}')
    if isinstance(value, int): number = value
    elif isinstance(value, str) and _DIGITS.fullmatch(value.strip()): number = int(value.strip())
    else: raise FrameContextError(f'Unrecognized {name} context: {value!r}')
    if number < 0: raise FrameContextError(f'Unrecognized {name} context: {value!r}')
    return number

def _optional_sensor(value):
    if value is None or value == '': return None
    if not isinstance(value, str) or not _SENSOR.fullmatch(value.strip()):
        raise FrameContextError(f'Unrecognized sensor context: {value!r}')
    return value.strip().lower()

def _split_sensor(text):
    if ':' not in text: return None, text
    sensor, body = text.split(':', 1)
    if not _SENSOR.fullmatch(sensor) or ':' in body or not body:
        raise FrameIdError(f'Unrecognized frame id: {text!r}')
    return sensor.lower(), body

def _numeric_parts(body):
    parts = re.split(r'[_-]', body)
    if not parts or any(not _DIGITS.fullmatch(part) for part in parts):
        raise FrameIdError(f'Unrecognized frame id: {body!r}')
    return [int(part) for part in parts]

def parse_frame_key(value, track, sensor=None, sequence=None):
    """Parse one identity. Formatting aliases collapse; extra digits and other sensors do not."""
    if track not in TRACKS: raise FrameIdError(f'Unrecognized track: {track!r}')
    text = _as_text(value)
    embedded, body = _split_sensor(text)
    field_sensor = _optional_sensor(sensor)
    if embedded and field_sensor and embedded != field_sensor:
        raise FrameContextError(f'Contradictory sensor context: {value!r} vs {sensor!r}')
    sensor = embedded or field_sensor
    field_sequence = _optional_int(sequence, 'sequence')
    parts = _numeric_parts(body)
    if track == 'v2x':
        if len(parts) != 2: raise FrameIdError(f'Unrecognized frame id: {value!r}')
        parsed_sequence, frame = parts
    elif len(parts) == 1:
        parsed_sequence, frame = None, parts[0]
    elif len(parts) == 2:
        parsed_sequence, frame = parts
    else:
        raise FrameIdError(f'Unrecognized frame id: {value!r}')
    if field_sequence is not None and parsed_sequence is not None and field_sequence != parsed_sequence:
        raise FrameContextError(f'Contradictory sequence context: {value!r} vs {sequence!r}')
    if parsed_sequence is None: parsed_sequence = field_sequence
    return FrameKey(track, frame, parsed_sequence, sensor)

def _forbidden_lines(path):
    p = Path(path)
    if not p.is_file(): raise ValueError('Official forbidden_frames.txt is required for real-data use')
    return [line.strip() for line in p.read_text().splitlines() if line.strip() and not line.startswith('#')]

def infer_forbidden_track(lines):
    kinds = set()
    for line in lines:
        _, body = _split_sensor(_as_text(line))
        parts = _numeric_parts(body)
        if len(parts) == 2: kinds.add('v2x')
        elif len(parts) == 1: kinds.add('lumpi')
        else: raise FrameIdError(f'Unrecognized frame id: {line!r}')
    if len(kinds) != 1: raise ValueError('Forbidden list mixes sequence-qualified and bare frame identities')
    return kinds.pop()

def forbidden_ids(path):
    lines = _forbidden_lines(path)
    if len(lines) != 100 or len(set(lines)) != 100:
        raise ValueError(f'Expected 100 official forbidden IDs, got {len(set(lines))}')
    track = infer_forbidden_track(lines)
    canonical = {parse_frame_key(line, track).canonical() for line in lines}
    if len(canonical) != 100:
        raise ValueError(f'Expected 100 official forbidden IDs, got {len(canonical)}')
    return canonical

def _optional_key(value, track, sensor=None, sequence=None):
    try: return parse_frame_key(value, track, sensor=sensor, sequence=sequence)
    except FrameContextError: raise
    except FrameIdError: return None

def audit_real_manifest(manifest, forbidden_path, track=None):
    """Validate real rows once. Forbidden aliases are rejected before any cloud is opened.

    Official IDs stay byte-compatible: ``0_000036`` and ``006497`` canonicalize to themselves.
    ``10_34`` matches ``10_000034``. ``1_000034`` does not match ``10_000034``, and an explicit
    sensor token does not collapse onto the unsensored official ID.
    """
    assert_not_held_out(manifest)
    forbidden = forbidden_ids(forbidden_path)
    inferred = infer_forbidden_track(_forbidden_lines(forbidden_path))
    if track is None: track = inferred
    elif track != inferred:
        raise ValueError(f'Forbidden list track is {inferred}, not {track}')
    rows = load_json(manifest)
    if not isinstance(rows, list): raise ValueError('Unrecognized real manifest')
    seen = {}
    audited = []
    for row in rows:
        if not isinstance(row, dict) or 'frame_id' not in row or 'path' not in row:
            raise ValueError('Unrecognized manifest row')
        path = row['path']
        assert_not_held_out(path)
        if Path(path).suffix.lower() in _LABEL_SUFFIXES:
            raise ValueError('Real labels are not accepted at this boundary')
        for field in ('labels', 'label', 'annotations'):
            if row.get(field):
                raise ValueError('Real labels are not accepted at this boundary')
        if row.get('domain') != 'real_development' or row.get('split') != 'train':
            raise ValueError('Real calibration requires domain=real_development and split=train')
        sensor, sequence = row.get('sensor'), row.get('sequence')
        full = parse_frame_key(row['frame_id'], track, sensor=sensor, sequence=sequence)
        bare = parse_frame_key(row['frame_id'], track)
        stem = Path(path).stem
        stem_full = _optional_key(stem, track, sensor=sensor, sequence=sequence)
        if stem_full is not None and stem_full != full:
            raise ValueError(f'Frame identity mismatch: {row["frame_id"]} vs {stem}')
        stem_bare = _optional_key(stem, track)
        identities = {full.canonical(), bare.canonical()}
        if stem_bare is not None: identities.add(stem_bare.canonical())
        if identities & forbidden:
            raise ValueError(f'Forbidden frame: {row["frame_id"]}')
        if full.canonical() in seen:
            raise ValueError(f'Duplicate frame identity: {full.canonical()}')
        seen[full.canonical()] = path
        audited.append({**row, 'canonical_id': full.canonical(), 'sensor': full.sensor,
                        'sequence': full.sequence, 'frame_index': full.frame, 'track': track})
    return AuditedRows(audited, track, digest(forbidden_path))

def label_class(name, track):
    if track == 'lumpi':
        alias = {'pedestrian':'Person','person':'Person','cyclist':'Bicycle','motorbike':'Motorcycle'}
        canonical = {x.lower():x for x in TRACKS[track]['classes']}
        return alias.get(name.lower(),canonical.get(name.lower()))
    # Executable SUPER_CLASS_MAP in UCLA V2X-Real, opencood/data_utils/__init__.py.
    # Bare Bicycle is deliberately absent from the active upstream dictionary.
    groups={'vehicle':['LongVehicle','Car','PoliceCar','vehicle'],
            'pedestrian':['Child','RoadWorker','Pedestrian','Scooter','ScooterRider','Motorcycle','MotorcyleRider','BicycleRider','pedestrian'],
            'truck':['Truck','Van','TrashCan','ConcreteTruck','Bus']}
    mapping={source.lower():target for target,sources in groups.items() for source in sources}
    return mapping.get(name.lower())

def load_labels(path, track):
    """Read SUSTechPOINTS annotations supplied by UrbanTwin, or canonical JSON boxes."""
    if Path(path).suffix == '.txt':
        rows = []
        for line in Path(path).read_text().splitlines():
            if not line.strip(): continue
            fields = line.split()
            rows.append({'box':list(map(float,fields[:7])),'class':fields[7]})
    else: rows = load_json(path)
    result = []
    for row in rows:
        name = label_class(row.get('obj_type', row.get('class','')), track)
        if name is None: continue
        if 'box' in row: box = row['box']
        else:
            psr = row['psr']; pos, size, rot = psr['position'], psr['scale'], psr['rotation']
            box = [pos['x'],pos['y'],pos['z'],size['x'],size['y'],size['z'],rot['z']]
        box = np.asarray(box,dtype=np.float32)
        if box.shape != (7,) or not np.isfinite(box).all() or (box[3:6]<=0).any():
            raise ValueError('Invalid synthetic box')
        result.append({'box':box.tolist(),'class':name})
    return result
