"""Synthetic-only out-of-fold proposal generation and second-stage refiner fitting.

The legacy ``box_refiner`` path scores proposals from a detector trained on the
same frames. Here each fold trains (or loads) a proposal model only on other
train-split frames, then collects proposals on held-out fold frames. Locked
validation, purged gap, official held-out paths, and real domains never enter
fitting.

Saved bundles work with ``apply_refiner``. Multi-checkpoint provenance is stored
explicitly; ``proposal_checkpoint_sha256`` is intentionally absent so nothing
falsely matches a single detector file.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from collections import defaultdict

from .augmentation_lineage import is_augmented_variant_row, validate_augmentation_lineage
from .box_refiner import apply_refiner, encode, features
from .common import TRACKS, assert_not_held_out, digest, held_out_input, load_json, load_labels, load_points, save_json, crop_labels, crop
from .detector import CompactPillar, predict, train as train_detector
from .geometry import iou3d, nms
from .training_splits import resolve_training_splits

DEFAULT_CACHE = Path("G:/BigData2026/challenge-04-urbantwin/artifacts/cache/refiner_oof")

REFINER_FIT_SPLITS = frozenset({"train"})
REFINER_DEV_SPLITS = frozenset({"development", "val"})
REFINER_EXCLUDED_SPLITS = frozenset({"locked_validation", "purged_gap", "test"})
MIN_FIT_FRAMES = 4
MIN_FOLD_TRAIN_FRAMES = 4


class OofLeakageError(ValueError):
    """Proposal-model training frames overlap refiner proposal frames."""


def verify_synthetic_row(row: dict, *, track: str) -> None:
    """Hash and domain checks shared by fitting and reporting."""
    if row.get("domain") != "synthetic":
        raise ValueError("Refiner OOF accepts verified synthetic manifest rows only")
    assert_not_held_out(row["path"])
    assert_not_held_out(row["labels"])
    if held_out_input(row["path"]) or held_out_input(row["labels"]):
        raise ValueError("Held-out cloud or label path in manifest")
    if row.get("sha256") != digest(row["path"]):
        raise ValueError(f"Cloud hash mismatch: {row['frame_id']}")
    if row.get("labels_sha256") != digest(row["labels"]):
        raise ValueError(f"Label hash mismatch: {row['frame_id']}")
    load_labels(row["labels"], track)


def partition_manifest_rows(rows: list[dict]):
    """Return fit, development, and excluded rows respecting explicit split metadata."""
    if rows and all(r.get('split') in REFINER_EXCLUDED_SPLITS for r in rows):
        return [], [], sorted(rows,key=lambda r:r['frame_id'])
    plan=resolve_training_splits(rows,min_fit=1)
    return plan.fit_rows,plan.selection_rows,plan.excluded_rows


def validate_oof_manifest_lineage(rows: list[dict], track: str) -> None:
    """Reject invalid parent/donor provenance before any fold fitting."""
    plan = resolve_training_splits(rows, min_fit=1)
    validate_augmentation_lineage(rows, track, plan, verify_hash=True)


def assert_fit_pool_train_only(fit_rows: list[dict]) -> None:
    for row in fit_rows:
        split = row.get("split")
        if split is not None and split not in REFINER_FIT_SPLITS:
            raise ValueError(
                f"OOF refiner fitting excludes split={split!r} frame {row.get('frame_id')!r}"
            )


def lineage_group_root_id(row: dict) -> str:
    if is_augmented_variant_row(row):
        parent = str(row.get("parent_frame_id", ""))
        if not parent:
            raise ValueError(f"Augmented variant missing parent_frame_id: {row.get('frame_id')!r}")
        return parent
    return str(row["frame_id"])


def build_lineage_groups(fit_rows: list[dict]) -> list[list[dict]]:
    """Parent bases and their augmented descendants share one OOF assignment unit."""
    variants_by_parent: dict[str, list[dict]] = defaultdict(list)
    bases: list[dict] = []
    for row in fit_rows:
        if is_augmented_variant_row(row):
            variants_by_parent[str(row["parent_frame_id"])].append(row)
        else:
            bases.append(row)
    base_ids = {str(r["frame_id"]) for r in bases}
    for parent, variant_rows in variants_by_parent.items():
        if parent not in base_ids:
            raise ValueError(f"OOF fit pool variant parent not in train split: {parent!r}")
    groups: list[list[dict]] = []
    for base in sorted(bases, key=lambda r: str(r["frame_id"])):
        parent_id = str(base["frame_id"])
        group = [base] + sorted(variants_by_parent.pop(parent_id, []), key=lambda r: str(r["frame_id"]))
        groups.append(group)
    if variants_by_parent:
        orphan = next(iter(variants_by_parent))
        raise ValueError(f"OOF fit pool variant parent not in train split: {orphan!r}")
    return groups


def assert_oof_lineage_groups_unsplit(
    trained_frame_ids: list[str],
    proposal_frame_ids: list[str],
    fit_rows: list[dict],
) -> None:
    trained = set(trained_frame_ids)
    proposal = set(proposal_frame_ids)
    for group in build_lineage_groups(fit_rows):
        members = {str(r["frame_id"]) for r in group}
        in_train = members & trained
        in_proposal = members & proposal
        if in_train and in_proposal:
            raise OofLeakageError(
                "OOF fold leak: parent/variant lineage group split between detector "
                f"training and proposals ({sorted(members)[:5]}...)"
            )
        if in_train | in_proposal != members:
            raise OofLeakageError(
                f"OOF fold plan omits lineage group members: {sorted(members - in_train - in_proposal)[:5]}..."
            )


def assert_oof_frame_disjoint(trained_frame_ids: list[str], proposal_frame_ids: list[str]) -> None:
    overlap = set(trained_frame_ids) & set(proposal_frame_ids)
    if overlap:
        raise OofLeakageError(
            f"OOF fold leak: proposal frames overlap detector training ({sorted(overlap)[:5]}...)"
        )


def plan_oof_folds(fit_rows: list[dict], n_folds: int, seed: int = 1234) -> list[dict]:
    """Deterministic fold plan over the train-split pool with lineage-isolated groups."""
    if n_folds < 2:
        raise ValueError("OOF refiner fitting requires at least two folds")
    groups = build_lineage_groups(fit_rows)
    if len(groups) < n_folds:
        raise ValueError(f"Need at least {n_folds} lineage groups, got {len(groups)}")
    if len(fit_rows) < n_folds:
        raise ValueError(f"Need at least {n_folds} train-split frames, got {len(fit_rows)}")
    ordered_groups = sorted(groups, key=lambda g: str(g[0]["frame_id"]))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ordered_groups))
    folds: list[dict] = []
    for fold in range(n_folds):
        proposal_group_idx = set(perm[fold::n_folds].tolist())
        proposal_rows = [row for gi, group in enumerate(ordered_groups) if gi in proposal_group_idx for row in group]
        trained_rows = [row for gi, group in enumerate(ordered_groups) if gi not in proposal_group_idx for row in group]
        trained_ids = [r["frame_id"] for r in trained_rows]
        proposal_ids = [r["frame_id"] for r in proposal_rows]
        assert_oof_frame_disjoint(trained_ids, proposal_ids)
        assert_oof_lineage_groups_unsplit(trained_ids, proposal_ids, fit_rows)
        if len(trained_rows) < MIN_FOLD_TRAIN_FRAMES:
            raise ValueError(
                f"Fold {fold} trains on {len(trained_rows)} frames; "
                f"detector training requires at least {MIN_FOLD_TRAIN_FRAMES}"
            )
        folds.append(
            {
                "fold": fold,
                "trained_rows": trained_rows,
                "proposal_rows": proposal_rows,
                "trained_frame_ids": trained_ids,
                "proposal_frame_ids": proposal_ids,
                "lineage_group_roots": sorted({lineage_group_root_id(r) for r in proposal_rows}),
            }
        )
    return folds


def _manifest_digest(rows: list[dict]) -> str:
    payload = json.dumps([(r['frame_id'],r['sha256'],r['labels_sha256']) for r in sorted(rows,key=lambda r:r['frame_id'])])
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_fold_manifest(path: Path, rows: list[dict]) -> None:
    save_json(path, rows)


def _verify_fold_checkpoint(checkpoint_path: Path, manifest_path: Path, track: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if ckpt.get("training_domain") != "synthetic":
        raise ValueError("Fold checkpoint must be synthetic-only")
    if ckpt.get("track") != track:
        raise ValueError("Fold checkpoint track mismatch")
    manifest_sha = digest(manifest_path)
    if ckpt.get("manifest_sha256") != manifest_sha:
        raise ValueError("Fold checkpoint manifest hash does not match fold training manifest")
    return ckpt


def train_fold_detector(
    fold_rows: list[dict],
    track: str,
    output: Path,
    *,
    epochs: int = 2,
    seed: int,
    architecture: str = "v1",
) -> Path:
    """Train a compact detector on one fold's training subset (CPU-safe small runs)."""
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "fold_manifest.json"
    _write_fold_manifest(manifest, fold_rows)
    args = SimpleNamespace(
        manifest=manifest,
        track=track,
        output=output,
        epochs=epochs,
        seed=seed,
        threads=1,
        device="cpu",
        evaluate_only=False,
        resume=False,
        architecture=architecture,
        selection_metric="loss",
        initialize_from=None,
    )
    train_detector(args)
    checkpoint = output / "best.pt"
    _verify_fold_checkpoint(checkpoint, manifest, track)
    return checkpoint


def _collect_proposal_rows(
    rows: list[dict],
    model: CompactPillar,
    track: str,
    classes: list[str],
    rng: np.random.Generator,
    device: str,
    *,
    proposal_threshold: float = 0.08,
):
    clf_x: list[np.ndarray] = []
    clf_y: list[int] = []
    reg_x: list[np.ndarray] = []
    reg_y: list[np.ndarray] = []
    for row in rows:
        points = crop(load_points(row["path"]),TRACKS[track]["range"])
        gt = crop_labels(load_labels(row["labels"], track),TRACKS[track]["range"])
        tree = cKDTree(points[:, :2])
        for obj in gt:
            for _ in range(5):
                perturbed = copy.deepcopy(obj)
                b = np.array(perturbed["box"])
                b[:2] += rng.normal(0, 0.65, 2)
                b[2] += rng.normal(0, 0.55)
                b[3:6] *= rng.uniform(0.85, 1.15, 3)
                b[6] += rng.normal(0, 0.35)
                perturbed["box"] = b.tolist()
                feat = features(points, tree, perturbed, classes)
                clf_x.append(feat)
                clf_y.append(1)
                reg_x.append(feat)
                reg_y.append(encode(b, obj["box"]))
        proposals = predict(model, points, track, device=device, threshold=proposal_threshold)
        for _ in range(10):
            k = int(rng.integers(len(classes)))
            same = [o for o in gt if o["class"] == classes[k]]
            if not same:
                continue
            decoy = copy.deepcopy(same[int(rng.integers(len(same)))])
            decoy["box"][:2] = rng.uniform(-38, 38, 2).tolist()
            proposals.append(decoy)
        for proposal in proposals:
            candidates = [(iou3d(proposal["box"], g["box"]), g) for g in gt if g["class"] == proposal["class"]]
            overlap, match = max(candidates, key=lambda pair: pair[0], default=(0, None))
            if 0.02 < overlap < 0.1:
                continue
            feat = features(points, tree, proposal, classes)
            clf_x.append(feat)
            clf_y.append(int(overlap >= 0.1))
            if overlap >= 0.1:
                reg_x.append(feat)
                reg_y.append(encode(proposal["box"], match["box"]))
    return clf_x, clf_y, reg_x, reg_y


def fit_refiner_models(clf_x, clf_y, reg_x, reg_y):
    classifier = ExtraTreesClassifier(
        n_estimators=64,
        max_depth=18,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=1,
        random_state=1234,
    )
    regressor = ExtraTreesRegressor(
        n_estimators=64,
        max_depth=18,
        min_samples_leaf=3,
        n_jobs=1,
        random_state=1234,
    )
    classifier.fit(np.asarray(clf_x), np.asarray(clf_y))
    regressor.fit(np.asarray(reg_x), np.asarray(reg_y))
    return classifier, regressor


def build_bundle(
    classifier,
    regressor,
    classes: list[str],
    manifest_sha256: str,
    fold_provenance: list[dict],
):
    return {
        "classifier": classifier,
        "regressor": regressor,
        "classes": classes,
        "training_domain": "synthetic",
        "manifest_sha256": manifest_sha256,
        "proposal_training": "out_of_fold_v1",
        "proposal_checkpoints": fold_provenance,
    }


def reject_in_sample_provenance(bundle: dict) -> None:
    """Guardrail: never ship a bundle that looks like single-checkpoint in-sample training."""
    if bundle.get("proposal_checkpoint_sha256"):
        raise ValueError("OOF bundle must not set proposal_checkpoint_sha256")
    checkpoints = bundle.get("proposal_checkpoints")
    if not checkpoints:
        raise ValueError("OOF bundle missing proposal_checkpoints provenance")
    for entry in checkpoints:
        assert_oof_frame_disjoint(entry["trained_frame_ids"], entry["proposal_frame_ids"])


def run_oof_training(
    manifest_path: Path,
    track: str,
    output: Path,
    *,
    n_folds: int = 5,
    epochs: int = 2,
    seed: int = 1234,
    device: str = "cpu",
    cache: Path = DEFAULT_CACHE,
    architecture: str = "v1",
    fold_checkpoints: dict[int, Path] | None = None,
) -> dict:
    torch.set_num_threads(1)
    rows = sorted(load_json(manifest_path), key=lambda r: r["frame_id"])
    for row in rows:
        verify_synthetic_row(row, track=track)
    validate_oof_manifest_lineage(rows, track)
    if len({r['frame_id'] for r in rows})!=len(rows) or len({r['sha256'] for r in rows})!=len(rows):
        raise OofLeakageError('Duplicate frame identity or cloud content across OOF pool')
    fit_rows, dev_rows, excluded = partition_manifest_rows(rows)
    assert_fit_pool_train_only(fit_rows)
    if not fit_rows:
        raise ValueError("No train-split frames available for OOF refiner fitting")
    folds = plan_oof_folds(fit_rows, n_folds, seed=seed)
    classes = TRACKS[track]["classes"]
    manifest_sha = digest(manifest_path)
    rng = np.random.default_rng(seed)
    cache = Path(cache)
    clf_x: list[np.ndarray] = []
    clf_y: list[int] = []
    reg_x: list[np.ndarray] = []
    reg_y: list[np.ndarray] = []
    fold_provenance: list[dict] = []

    for spec in folds:
        fold = spec["fold"]
        fold_dir = cache / f"{track}_oof_fold{fold}"
        if fold_checkpoints and fold in fold_checkpoints:
            checkpoint = Path(fold_checkpoints[fold])
            fold_manifest = checkpoint.parent / "fold_manifest.json"
            if not fold_manifest.is_file():
                raise ValueError(f"Missing fold manifest beside checkpoint: {fold_manifest}")
            if _manifest_digest(spec["trained_rows"]) != _manifest_digest(load_json(fold_manifest)):
                raise OofLeakageError("Fold checkpoint manifest does not match planned training frames")
            ckpt = _verify_fold_checkpoint(checkpoint, fold_manifest, track)
        else:
            checkpoint = train_fold_detector(
                spec["trained_rows"], track, fold_dir, epochs=epochs, seed=seed + fold, architecture=architecture
            )
            ckpt = _verify_fold_checkpoint(checkpoint, fold_dir / "fold_manifest.json", track)
        version = int(ckpt["architecture"][-1])
        model = CompactPillar(len(classes), version).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        f_clf, f_y, f_rx, f_ry = _collect_proposal_rows(
            spec["proposal_rows"], model, track, classes, rng, device
        )
        clf_x.extend(f_clf)
        clf_y.extend(f_y)
        reg_x.extend(f_rx)
        reg_y.extend(f_ry)
        fold_provenance.append(
            {
                "fold": fold,
                "checkpoint_sha256": digest(checkpoint),
                "fold_manifest_sha256": digest(checkpoint.parent / "fold_manifest.json"),
                "trained_frame_ids": spec["trained_frame_ids"],
                "proposal_frame_ids": spec["proposal_frame_ids"],
                "architecture": ckpt.get("architecture"),
            }
        )

    if not clf_x:
        raise ValueError("OOF proposal collection produced no refiner training features")
    classifier, regressor = fit_refiner_models(clf_x, clf_y, reg_x, reg_y)
    bundle = build_bundle(classifier, regressor, classes, manifest_sha, fold_provenance)
    reject_in_sample_provenance(bundle)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / "refiner.joblib", compress=3)

    report = {
        "domain": "synthetic_oof_refiner",
        "official": False,
        "held_out_real_used": False,
        "manifest_sha256": manifest_sha,
        "manifest_path": str(manifest_path.resolve()),
        "track": track,
        "n_folds": n_folds,
        "training_samples": len(clf_x),
        "regression_samples": len(reg_x),
        "fit_frame_ids": [r["frame_id"] for r in fit_rows],
        "development_frame_ids": [r["frame_id"] for r in dev_rows],
        "excluded_frame_ids": [r["frame_id"] for r in excluded],
        "folds": fold_provenance,
        "proposal_training": bundle["proposal_training"],
        "note": "OOF proposals only; locked_validation and purged_gap never used.",
    }
    save_json(output / "oof_provenance.json", report)

    if dev_rows:
        primary_ckpt = cache / f"{track}_oof_fold0" / "best.pt"
        if primary_ckpt.is_file():
            ckpt = torch.load(primary_ckpt, map_location="cpu", weights_only=True)
            eval_model = CompactPillar(len(classes), int(ckpt["architecture"][-1])).to(device)
            eval_model.load_state_dict(ckpt["model"])
            eval_model.eval()
            dev_stats = {"proposal": {"TP": 0, "FP": 0, "GT": 0}, "refined": {"TP": 0, "FP": 0, "GT": 0}}
            for row in dev_rows:
                points = crop(load_points(row["path"]),TRACKS[track]["range"])
                gt = crop_labels(load_labels(row["labels"], track),TRACKS[track]["range"])
                proposals = predict(eval_model, points, track, device=device, threshold=0.08)
                refined = apply_refiner(points, proposals, bundle)
                for name, predictions in (
                    ("proposal", nms(proposals)),
                    ("refined", refined),
                ):
                    used: set[int] = set()
                    dev_stats[name]["GT"] += len(gt)
                    for pred in sorted(predictions, key=lambda p: -p["score"]):
                        matches = [
                            (iou3d(pred["box"], g["box"]), i)
                            for i, g in enumerate(gt)
                            if i not in used and pred["class"] == g["class"]
                        ]
                        overlap, index = max(matches, default=(0, -1))
                        if overlap >= 0.5:
                            dev_stats[name]["TP"] += 1
                            used.add(index)
                        else:
                            dev_stats[name]["FP"] += 1
            for stats in dev_stats.values():
                stats["precision"] = stats["TP"] / max(stats["TP"] + stats["FP"], 1)
                stats["recall"] = stats["TP"] / max(stats["GT"], 1)
            report["development_comparison"] = dev_stats

    save_json(output / "oof_provenance.json", report)
    return report


def bind_deployment_checkpoint(bundle_path, checkpoint_path, manifest_path, output_path):
    """Bind a verified OOF refiner to one separately trained production detector."""
    bundle=joblib.load(bundle_path)
    reject_in_sample_provenance(bundle)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint.get('training_domain')!='synthetic' or checkpoint.get('manifest_sha256')!=digest(manifest_path):
        raise ValueError('Deployment checkpoint synthetic manifest mismatch')
    if bundle['manifest_sha256']!=digest(manifest_path) or bundle['classes']!=TRACKS[checkpoint['track']]['classes']:
        raise ValueError('OOF bundle and deployment detector corpus/classes mismatch')
    rows=load_json(manifest_path)
    for row in rows:verify_synthetic_row(row,track=checkpoint['track'])
    validate_oof_manifest_lineage(rows,checkpoint['track'])
    fit,dev,excluded=partition_manifest_rows(rows)
    assert_fit_pool_train_only(fit)
    expected={r['frame_id'] for r in fit}
    proposals=[fid for fold in bundle['proposal_checkpoints'] for fid in fold['proposal_frame_ids']]
    if len(proposals)!=len(set(proposals)) or set(proposals)!=expected:
        raise ValueError('OOF proposal coverage does not match fitting pool')
    for fold in bundle['proposal_checkpoints']:
        if not set(fold['trained_frame_ids'])<=expected:
            raise ValueError('OOF detector training escapes fitting pool')
        if fold['architecture']!=checkpoint['architecture']:
            raise ValueError('OOF proposal architecture differs from deployment detector')
    bundle['deployment_checkpoint_sha256']=digest(checkpoint_path)
    save_path=Path(output_path);save_path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,save_path,compress=3)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=TRACKS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2, help="Per-fold detector epochs (CPU pilot default)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--architecture", choices=["v1", "v2", "v3", "v4", "v5"], default="v1")
    args = parser.parse_args()
    report = run_oof_training(
        args.manifest,
        args.track,
        args.output,
        n_folds=args.folds,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        cache=args.cache,
        architecture=args.architecture,
    )
    print(json.dumps({"training_samples": report["training_samples"], "folds": len(report["folds"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
