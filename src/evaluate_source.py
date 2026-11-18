"""Class-balanced synthetic validation diagnostics, never official KITTI scores."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from .common import TRACKS, digest, load_json, load_labels, save_json, crop_labels
from .geometry import iou3d
from .training_splits import legacy_selection_rows, resolve_training_splits, rows_for_evaluation


def _evaluation_manifest_rows(
    manifest: list[dict] | Path,
    evaluation_split: str = "development",
) -> list[dict]:
    rows = load_json(manifest) if isinstance(manifest, (str, Path)) else list(manifest)
    rows = sorted(rows, key=lambda r: r["frame_id"])
    has_explicit = any("split" in r and r.get("split") is not None for r in rows)
    if not has_explicit:
        if evaluation_split.strip().lower() in {"locked_validation", "locked"}:
            raise ValueError("locked_validation evaluation requires explicit split metadata on the manifest")
        return legacy_selection_rows(rows)
    plan = resolve_training_splits(rows, min_fit=1)
    return rows_for_evaluation(plan, evaluation_split)


def official_synthetic_evaluation(
    manifest,
    predictions,
    track,
    *,
    evaluation_split: str = "development",
):
    """Run unmodified organizer AP code on synthetic validation labels only."""
    root = Path(__file__).resolve().parents[1]
    kit = "v2x_real_starting_kit" if track == "v2x" else "lumpi_starting_kit"
    source = root / kit / "scoring_program" / "detection_eval.py"
    spec = importlib.util.spec_from_file_location("organizer_detection_eval", source)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    rows = _evaluation_manifest_rows(manifest, evaluation_split)
    if not rows or set(predictions) != {r["frame_id"] for r in rows}:
        raise ValueError("Predictions must cover the exact synthetic validation split")
    truth = {}
    for row in rows:
        if row["domain"] != "synthetic" or digest(row["labels"]) != row["labels_sha256"]:
            raise ValueError("Only verified synthetic labels are allowed")
        truth[row["frame_id"]] = crop_labels(load_labels(row["labels"], track),TRACKS[track]["range"])
    result = {
        "domain": "synthetic_validation",
        "evaluation_split": evaluation_split,
        "server_score": False,
        "ground_truth_policy": "track_range_box_center_v1",
        "scorer_sha256": digest(source),
        "detection_3d": scorer.evaluate_3d_detection(predictions, truth, TRACKS[track]["classes"]),
        "detection_bev": scorer.evaluate_3d_detection(
            predictions, truth, TRACKS[track]["classes"], iou_type="bev"
        ),
    }
    return json.loads(json.dumps(result), parse_constant=lambda value: None)


def evaluate(manifest, predictions, track, *, evaluation_split: str = "development"):
    rows = _evaluation_manifest_rows(manifest, evaluation_split)
    if set(predictions) != {r["frame_id"] for r in rows}:
        raise ValueError("Predictions must cover the exact synthetic validation split")
    truth = {}
    for row in rows:
        if row["domain"] != "synthetic" or digest(row["labels"]) != row["labels_sha256"]:
            raise ValueError("Only verified synthetic labels are allowed")
        truth[row["frame_id"]] = crop_labels(load_labels(row["labels"], track),TRACKS[track]["range"])
    per_class = {}
    for name in TRACKS[track]["classes"]:
        count = sum(o["class"] == name for frame in truth.values() for o in frame)
        detections = sorted(
            [(p["score"], frame, p["box"]) for frame, values in predictions.items() for p in values if p["class"] == name],
            key=lambda p: -p[0],
        )
        matched = {frame: set() for frame in truth}
        flags = []
        for score, frame, box in detections:
            choices = [
                (iou3d(box, g["box"]), i)
                for i, g in enumerate(truth[frame])
                if g["class"] == name and i not in matched[frame]
            ]
            overlap, index = max(choices, default=(0, -1))
            good = overlap >= 0.5
            flags.append(good)
            if good:
                matched[frame].add(index)
        tp = np.cumsum(flags)
        fp = np.cumsum(np.logical_not(flags))
        recall = tp / max(count, 1)
        precision = tp / np.maximum(tp + fp, 1)
        ap = (
            float(np.mean([precision[recall >= r].max(initial=0) for r in np.arange(1, 41) / 40]))
            if count
            else None
        )
        per_class[name] = {
            "GT": count,
            "predictions": len(detections),
            "TP": int(tp[-1]) if len(tp) else 0,
            "FP": int(fp[-1]) if len(fp) else 0,
            "AP40_proxy": ap,
            "maximum_recall": float(recall[-1]) if len(recall) else 0,
        }
    values = [p["AP40_proxy"] for p in per_class.values() if p["AP40_proxy"] is not None]
    return {
        "domain": "synthetic_validation",
        "evaluation_split": evaluation_split,
        "official": False,
        "per_class": per_class,
        "mean_AP40_proxy": float(np.mean(values)) if values else None,
        "warning": "Fixed recall-grid interpolation with greedy matching; not the official local_eval.py or a real-target score.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--track", choices=TRACKS, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--evaluation-split",
        default="development",
        choices=["development", "dev", "val", "validation", "locked_validation", "locked"],
        help="Scoring split (default development; locked only for explicit final eval)",
    )
    a = p.parse_args()
    result = evaluate(a.manifest, load_json(a.predictions), a.track, evaluation_split=a.evaluation_split)
    save_json(a.output, result)
    print(result)


if __name__ == "__main__":
    main()
