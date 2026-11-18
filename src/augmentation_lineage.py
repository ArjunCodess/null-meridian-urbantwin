"""Synthetic augmentation parent/variant provenance for experiment-runner validation.

Augmented frame IDs use ``<parent>__aug_<tag>_<nn>`` suffixes. Strict manifest
geometry audit binds canonical base identities only; variants are validated here
with explicit parent, donor, and content-hash lineage before geometry and class
coverage are merged into the runner audit.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit_dataset import (
    MINIMUM_INSTANCES,
    SCORED_CLASSES,
    SUPPORT_MIN_POINTS,
    _coverage,
    _dimension_medians,
    _raw_label_rows,
    bind_synthetic_row,
    diagnose_frame,
)
from .common import TRACKS, assert_not_held_out, digest, label_class, load_labels, load_points
from .training_splits import SplitPlan

KNOWN_AUGMENTATION_PROVENANCE = frozenset(
    {
        "synthetic_gt_copypaste_v1",
        "procedural_articulated_human_v1",
        "synthetic_gt_copypaste_plus_procedural_person_v1",
    }
)

_VARIANT_SUFFIX_RE = re.compile(r"^(.+)__aug_([a-zA-Z][a-zA-Z0-9_]*)_(\d{2})$")
_RAW_WALKER_DONOR_PREFIX = "raw_walker:"


@dataclass(frozen=True)
class ParsedVariantId:
    parent_token: str
    tag: str
    variant_index: int


def parse_augmented_variant_id(frame_id: str) -> ParsedVariantId | None:
    """Return parsed variant suffix fields, or None if the id is a base frame id."""
    text = str(frame_id).strip()
    if "__aug_" not in text:
        return None
    match = _VARIANT_SUFFIX_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"Ambiguous augmented frame id suffix: {text!r}")
    parent_token, tag, idx_text = match.groups()
    return ParsedVariantId(parent_token=parent_token, tag=tag, variant_index=int(idx_text))


def is_augmented_variant_row(row: dict[str, Any]) -> bool:
    if row.get("is_augmented_variant") is True:
        return True
    if row.get("is_augmented_variant") is False:
        return False
    return "__aug_" in str(row.get("frame_id", ""))


def split_base_and_variant_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bases: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    for row in rows:
        flagged = row.get("is_augmented_variant")
        parsed = parse_augmented_variant_id(str(row.get("frame_id", "")))
        if flagged is True or (flagged is None and parsed is not None):
            if flagged is False:
                raise ValueError(
                    f"Frame {row.get('frame_id')!r} has an augmentation suffix but is_augmented_variant=False"
                )
            if parsed is None and "__aug_" in str(row.get("frame_id", "")):
                raise ValueError(f"Ambiguous augmented frame id suffix: {row.get('frame_id')!r}")
            variants.append(row)
        else:
            if flagged is True:
                raise ValueError(f"Variant flag set without a valid augmented frame id: {row.get('frame_id')!r}")
            if parsed is not None:
                raise ValueError(
                    f"Augmented suffix on frame {row.get('frame_id')!r} requires is_augmented_variant=True"
                )
            bases.append(row)
    return bases, variants


def _parent_token(parent_frame_id: str) -> str:
    return str(parent_frame_id).replace(":", "_")


def _assert_variant_paths(row: dict[str, Any], track: str) -> dict[str, Any]:
    for key in ("frame_id", "path", "labels"):
        if key not in row:
            raise ValueError(f"Variant row missing required field {key!r}")
    assert_not_held_out(row["path"])
    assert_not_held_out(row["labels"])
    if row.get("domain") != "synthetic":
        raise ValueError("Variant rows must use domain='synthetic'")
    frame_id = str(row["frame_id"])
    parsed = parse_augmented_variant_id(frame_id)
    if parsed is None:
        raise ValueError(f"Invalid augmented variant frame id: {frame_id!r}")
    stem = Path(row["path"]).stem
    if stem != frame_id:
        raise ValueError(f"Variant cloud stem mismatch: {frame_id!r} vs {stem!r}")
    label_stem = Path(row["labels"]).stem
    if label_stem != frame_id:
        raise ValueError(f"Variant label stem mismatch: {frame_id!r} vs {label_stem!r}")
    if Path(row["path"]).suffix.lower() not in {".npy", ".bin", ".pcd"}:
        raise ValueError(f"Unrecognized synthetic cloud: {row['path']}")
    if Path(row["labels"]).suffix.lower() not in {".txt", ".json"}:
        raise ValueError(f"Unrecognized synthetic labels: {row['labels']}")
    parent_id = str(row.get("parent_frame_id", ""))
    if not parent_id:
        raise ValueError(f"Orphan augmented variant missing parent_frame_id: {frame_id!r}")
    if parsed.parent_token != _parent_token(parent_id):
        raise ValueError(
            f"Variant parent token mismatch for {frame_id!r}: "
            f"{parsed.parent_token!r} vs parent_frame_id {parent_id!r}"
        )
    expected_id = f"{parsed.parent_token}__aug_{parsed.tag}_{parsed.variant_index:02d}"
    if frame_id != expected_id:
        raise ValueError(f"Variant frame_id does not match parsed suffix fields: {frame_id!r}")
    return {**row, "track": track, "parsed_variant": parsed}


def validate_augmentation_lineage(
    rows: list[dict[str, Any]],
    track: str,
    plan: SplitPlan,
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    """Reject orphan/cross-split parents, invalid donors, and hash/provenance drift."""
    if track not in TRACKS:
        raise ValueError(f"Unrecognized track: {track!r}")
    bases, variants = split_base_and_variant_rows(rows)
    if not variants:
        return {"variant_count": 0, "base_count": len(bases)}

    base_by_id = {str(r["frame_id"]): r for r in bases}
    fit_ids = {str(r["frame_id"]) for r in plan.fit_rows}
    selection_ids = {str(r["frame_id"]) for r in plan.selection_rows}
    excluded_ids = {str(r["frame_id"]) for r in plan.excluded_rows}
    non_train_ids = selection_ids | excluded_ids
    variant_ids = {str(v["frame_id"]) for v in variants}

    seen_cloud: set[str] = set()
    seen_label: set[str] = set()
    seen_variant_ids: set[str] = set()
    lineage_rows: list[dict[str, Any]] = []

    for row in variants:
        bound = _assert_variant_paths(row, track)
        vfid = str(bound["frame_id"])
        if vfid in seen_variant_ids:
            raise ValueError(f"Duplicate augmented variant id: {vfid!r}")
        seen_variant_ids.add(vfid)

        parent_id = str(bound["parent_frame_id"])
        parent = base_by_id.get(parent_id)
        if parent is None:
            raise ValueError(f"Orphan augmented variant parent not in manifest: {parent_id!r}")
        if parent_id not in fit_ids:
            raise ValueError(
                f"Augmented variant {vfid!r} parent {parent_id!r} is outside the training fit split"
            )
        if parent_id in non_train_ids:
            raise ValueError(f"Augmented variant {vfid!r} parent {parent_id!r} is from a non-train split")

        if bound.get("split") != "train":
            raise ValueError(f"Augmented variant {vfid!r} must remain on split='train'")
        if vfid not in fit_ids:
            raise ValueError(f"Augmented variant {vfid!r} is outside the resolved training fit split")
        if bound.get("independent_scene") is not False:
            raise ValueError(f"Augmented variant {vfid!r} must have independent_scene=False")

        provenance = bound.get("provenance")
        if provenance not in KNOWN_AUGMENTATION_PROVENANCE:
            raise ValueError(f"Unknown augmentation provenance for {vfid!r}: {provenance!r}")

        for field_name, parent_field in (
            ("parent_sha256", "sha256"),
            ("parent_labels_sha256", "labels_sha256"),
        ):
            claimed = bound.get(field_name)
            actual = parent.get(parent_field)
            if claimed is None:
                raise ValueError(f"Variant {vfid!r} missing lineage field {field_name!r}")
            if actual is None:
                raise ValueError(f"Parent {parent_id!r} missing {parent_field!r} for lineage audit")
            if str(claimed) != str(actual):
                raise ValueError(f"Mutated parent hash on variant {vfid!r}: field {field_name!r}")

        for context_field in ("sequence", "sensor"):
            if bound.get(context_field) != parent.get(context_field):
                raise ValueError(
                    f"Variant {vfid!r} {context_field} does not match parent {parent_id!r}"
                )

        if verify_hash:
            cloud_sha = digest(bound["path"])
            label_sha = digest(bound["labels"])
            if bound.get("sha256") and bound["sha256"] != cloud_sha:
                raise ValueError(f"Variant cloud hash mismatch: {vfid!r}")
            if bound.get("labels_sha256") and bound["labels_sha256"] != label_sha:
                raise ValueError(f"Variant label hash mismatch: {vfid!r}")
        else:
            cloud_sha = str(bound.get("sha256") or digest(bound["path"]))
            label_sha = str(bound.get("labels_sha256") or digest(bound["labels"]))

        if cloud_sha in seen_cloud:
            raise ValueError(f"Duplicate variant cloud content hash: {vfid!r}")
        if label_sha in seen_label:
            raise ValueError(f"Duplicate variant label content hash: {vfid!r}")
        seen_cloud.add(cloud_sha)
        seen_label.add(label_sha)

        parent_cloud_sha = str(parent.get("sha256", ""))
        parent_label_sha = str(parent.get("labels_sha256", ""))
        if cloud_sha == parent_cloud_sha and label_sha == parent_label_sha:
            raise ValueError(f"Variant {vfid!r} must not duplicate parent content hashes")

        donors = [str(x) for x in bound.get("donor_frame_ids", [])]
        for donor_id in donors:
            if donor_id.startswith(_RAW_WALKER_DONOR_PREFIX):
                raise ValueError('External raw walker donor lacks original training-row lineage')
            if donor_id in variant_ids:
                raise ValueError(f"Augmented variant {vfid!r} used another variant as donor: {donor_id!r}")
            if donor_id not in fit_ids or donor_id in non_train_ids:
                raise ValueError(
                    f"Augmented variant {vfid!r} donor {donor_id!r} is outside original training scenes"
                )
            if donor_id not in base_by_id:
                raise ValueError(f"Augmented variant {vfid!r} donor {donor_id!r} is not an original base row")

        lineage_rows.append(
            {
                "variant_frame_id": vfid,
                "parent_frame_id": parent_id,
                "provenance": provenance,
                "donor_frame_ids": donors,
            }
        )

    return {
        "variant_count": len(variants),
        "base_count": len(bases),
        "variants": lineage_rows,
        "cross_split_leakage": False,
        "variants_claim_independent_scenes": False,
    }


def audit_synthetic_manifest(
    rows: list[dict[str, Any]],
    track: str,
    plan: SplitPlan,
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    """Audit base frames with canonical identity rules and variants via lineage context."""
    bases, variants = split_base_and_variant_rows(rows)
    lineage = validate_augmentation_lineage(rows, track, plan, verify_hash=verify_hash)

    base_audit = _audit_base_rows(bases, track, verify_hash=verify_hash)
    if not variants:
        base_audit["augmentation_lineage"] = lineage
        return base_audit

    variant_audit = _audit_variant_geometry(variants, track, verify_hash=verify_hash)
    merged = _merge_audits(base_audit, variant_audit, track)
    merged["augmentation_lineage"] = lineage
    return merged


def _audit_base_rows(
    rows: list[dict[str, Any]],
    track: str,
    *,
    verify_hash: bool,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Manifest has augmented variants but no base rows")
    bound_rows = []
    seen: dict[str, str] = {}
    for row in rows:
        bound = bind_synthetic_row(row, track)
        if bound["canonical_id"] in seen:
            raise ValueError(f'Duplicate frame identity: {bound["canonical_id"]}')
        seen[bound["canonical_id"]] = bound["path"]
        bound_rows.append(bound)
    return _scan_bound_rows(bound_rows, track, verify_hash=verify_hash)


def _audit_variant_geometry(
    variants: list[dict[str, Any]],
    track: str,
    *,
    verify_hash: bool,
) -> dict[str, Any]:
    class_counts: dict[str, Counter] = {split: Counter() for split in ("train", "locked_validation", "purged_gap", "unspecified")}
    raw_counts: Counter = Counter()
    unmapped: Counter = Counter()
    mapped: Counter = Counter()
    subtypes: Counter = Counter()
    dim_samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"dx": [], "dy": [], "dz": []})
    frames: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    totals: Counter = Counter()

    for row in variants:
        bound = _assert_variant_paths(row, track)
        split = bound.get("split") or "unspecified"
        if split not in class_counts:
            split = "unspecified"
        try:
            if verify_hash and bound.get("sha256") and digest(bound["path"]) != bound["sha256"]:
                raise ValueError(f'Content hash mismatch: {bound["path"]}')
            if verify_hash and bound.get("labels_sha256") and digest(bound["labels"]) != bound["labels_sha256"]:
                raise ValueError(f'Label hash mismatch: {bound["labels"]}')
            for item in _raw_label_rows(bound["labels"]):
                raw_counts[item["class"]] += 1
                mapped_name = label_class(item["class"], track)
                if mapped_name is None:
                    unmapped[item["class"]] += 1
                else:
                    mapped[mapped_name] += 1
                    if mapped_name == "pedestrian":
                        from .audit_dataset import _subtype

                        subtypes[_subtype(item["class"])] += 1
            points = load_points(bound["path"])
            objects = load_labels(bound["labels"], track)
            stats = diagnose_frame(points, objects, track)
        except ValueError as exc:
            from .audit_dataset import _FRAME_ERROR_PREFIXES

            if not str(exc).startswith(_FRAME_ERROR_PREFIXES):
                raise
            errors.append(
                {
                    "frame_id": str(bound["frame_id"]),
                    "parent_frame_id": str(bound.get("parent_frame_id", "")),
                    "error": str(exc),
                }
            )
            continue

        for obj in objects:
            class_counts[split][obj["class"]] += 1
        for name, samples in stats.pop("class_dims").items():
            for dx, dy, dz in samples:
                dim_samples[name]["dx"].append(dx)
                dim_samples[name]["dy"].append(dy)
                dim_samples[name]["dz"].append(dz)
        for key in (
            "points",
            "points_cropped",
            "boxes",
            "boxes_in_range",
            "boxes_supported",
            "z_closer_to_center",
            "z_closer_to_bottom",
            "z_ambiguous",
            "yaw_outside_pi",
            "overlapping_pairs",
        ):
            totals[key] += stats[key]
        frames.append(
            {
                "frame_id": str(bound["frame_id"]),
                "parent_frame_id": str(bound.get("parent_frame_id", "")),
                "is_augmented_variant": True,
                "sequence": bound.get("sequence"),
                "frame_index": None,
                "sensor": bound.get("sensor"),
                "split": split,
                "points": stats["points"],
                "points_cropped": stats["points_cropped"],
                "boxes": stats["boxes"],
                "boxes_in_range": stats["boxes_in_range"],
                "boxes_supported": stats["boxes_supported"],
                "z_closer_to_center": stats["z_closer_to_center"],
                "z_closer_to_bottom": stats["z_closer_to_bottom"],
                "overlapping_pairs": stats["overlapping_pairs"],
            }
        )

    coverage, missing, below = _coverage(class_counts, track)
    points = int(totals["points"])
    if totals["z_closer_to_center"] > totals["z_closer_to_bottom"]:
        interpretation = "Interior point z is closer to the cuboid center than to the bottom face."
    elif totals["z_closer_to_bottom"] > totals["z_closer_to_center"]:
        interpretation = "Interior point z is closer to the bottom face than to the cuboid center."
    else:
        interpretation = "Interior point z does not favor the center or the bottom face."

    return {
        "frames": frames,
        "frame_errors": errors,
        "class_coverage": coverage,
        "missing_classes": missing,
        "below_minimum": below,
        "raw_class_counts": dict(raw_counts),
        "mapped_class_counts": dict(mapped),
        "unmapped_class_counts": dict(unmapped),
        "pedestrian_subtypes": dict(subtypes),
        "geometry": {
            "boxes": int(totals["boxes"]),
            "boxes_in_range": int(totals["boxes_in_range"]),
            "boxes_supported": int(totals["boxes_supported"]),
            "z_closer_to_center": int(totals["z_closer_to_center"]),
            "z_closer_to_bottom": int(totals["z_closer_to_bottom"]),
            "z_ambiguous": int(totals["z_ambiguous"]),
            "z_interpretation": interpretation,
            "yaw_outside_pi": int(totals["yaw_outside_pi"]),
            "overlapping_pairs": int(totals["overlapping_pairs"]),
            "dimension_medians": _dimension_medians(dim_samples),
            "support_min_points": SUPPORT_MIN_POINTS,
            "box_origin": "openpcdet_center",
        },
        "crop": {
            "points": points,
            "points_cropped": int(totals["points_cropped"]),
            "fraction": None if not points else float(totals["points_cropped"] / points),
        },
    }


def _scan_bound_rows(bound_rows: list[dict[str, Any]], track: str, *, verify_hash: bool) -> dict[str, Any]:
    """Geometry/class scan shared with audit_dataset.audit_records."""
    class_counts: dict[str, Counter] = {
        split: Counter() for split in ("train", "locked_validation", "purged_gap", "unspecified")
    }
    raw_counts: Counter = Counter()
    unmapped: Counter = Counter()
    mapped: Counter = Counter()
    subtypes: Counter = Counter()
    dim_samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"dx": [], "dy": [], "dz": []})
    frames: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    totals: Counter = Counter()

    for bound in bound_rows:
        split = bound.get("split") or "unspecified"
        if split not in class_counts:
            split = "unspecified"
        try:
            if verify_hash and bound.get("sha256") and digest(bound["path"]) != bound["sha256"]:
                raise ValueError(f'Content hash mismatch: {bound["path"]}')
            if verify_hash and bound.get("labels_sha256") and digest(bound["labels"]) != bound["labels_sha256"]:
                raise ValueError(f'Label hash mismatch: {bound["labels"]}')
            for item in _raw_label_rows(bound["labels"]):
                raw_counts[item["class"]] += 1
                mapped_name = label_class(item["class"], track)
                if mapped_name is None:
                    unmapped[item["class"]] += 1
                else:
                    mapped[mapped_name] += 1
                    if mapped_name == "pedestrian":
                        from .audit_dataset import _subtype

                        subtypes[_subtype(item["class"])] += 1
            points = load_points(bound["path"])
            objects = load_labels(bound["labels"], track)
            stats = diagnose_frame(points, objects, track)
        except ValueError as exc:
            from .audit_dataset import _FRAME_ERROR_PREFIXES

            if not str(exc).startswith(_FRAME_ERROR_PREFIXES):
                raise
            errors.append({"frame_id": bound["canonical_id"], "error": str(exc)})
            continue

        for obj in objects:
            class_counts[split][obj["class"]] += 1
        for name, samples in stats.pop("class_dims").items():
            for dx, dy, dz in samples:
                dim_samples[name]["dx"].append(dx)
                dim_samples[name]["dy"].append(dy)
                dim_samples[name]["dz"].append(dz)
        for key in (
            "points",
            "points_cropped",
            "boxes",
            "boxes_in_range",
            "boxes_supported",
            "z_closer_to_center",
            "z_closer_to_bottom",
            "z_ambiguous",
            "yaw_outside_pi",
            "overlapping_pairs",
        ):
            totals[key] += stats[key]
        frames.append(
            {
                "frame_id": bound["canonical_id"],
                "sequence": bound["sequence"],
                "frame_index": bound["frame_index"],
                "sensor": bound["sensor"],
                "split": split,
                "points": stats["points"],
                "points_cropped": stats["points_cropped"],
                "boxes": stats["boxes"],
                "boxes_in_range": stats["boxes_in_range"],
                "boxes_supported": stats["boxes_supported"],
                "z_closer_to_center": stats["z_closer_to_center"],
                "z_closer_to_bottom": stats["z_closer_to_bottom"],
                "overlapping_pairs": stats["overlapping_pairs"],
            }
        )

    coverage, missing, below = _coverage(class_counts, track)
    points = int(totals["points"])
    if totals["z_closer_to_center"] > totals["z_closer_to_bottom"]:
        interpretation = "Interior point z is closer to the cuboid center than to the bottom face."
    elif totals["z_closer_to_bottom"] > totals["z_closer_to_center"]:
        interpretation = "Interior point z is closer to the bottom face than to the cuboid center."
    else:
        interpretation = "Interior point z does not favor the center or the bottom face."

    return {
        "track": track,
        "official_inputs_read": False,
        "frames_audited": len(frames),
        "frame_errors": errors,
        "frames": frames,
        "class_coverage": coverage,
        "missing_classes": missing,
        "below_minimum": below,
        "minimum_instances": MINIMUM_INSTANCES,
        "raw_class_counts": dict(raw_counts),
        "mapped_class_counts": dict(mapped),
        "unmapped_class_counts": dict(unmapped),
        "pedestrian_subtypes": dict(subtypes),
        "crop": {
            "points": points,
            "points_cropped": int(totals["points_cropped"]),
            "fraction": None if not points else float(totals["points_cropped"] / points),
        },
        "geometry": {
            "box_origin": "openpcdet_center",
            "support_min_points": SUPPORT_MIN_POINTS,
            "boxes": int(totals["boxes"]),
            "boxes_in_range": int(totals["boxes_in_range"]),
            "boxes_supported": int(totals["boxes_supported"]),
            "z_closer_to_center": int(totals["z_closer_to_center"]),
            "z_closer_to_bottom": int(totals["z_closer_to_bottom"]),
            "z_ambiguous": int(totals["z_ambiguous"]),
            "z_interpretation": interpretation,
            "yaw_outside_pi": int(totals["yaw_outside_pi"]),
            "overlapping_pairs": int(totals["overlapping_pairs"]),
            "dimension_medians": _dimension_medians(dim_samples),
        },
    }


def _merge_audits(base: dict[str, Any], variant: dict[str, Any], track: str) -> dict[str, Any]:
    merged = dict(base)
    merged["frames"] = list(base.get("frames", [])) + list(variant.get("frames", []))
    merged["frame_errors"] = list(base.get("frame_errors", [])) + list(variant.get("frame_errors", []))
    merged["frames_audited"] = len(merged["frames"])

    for counter_name in ("raw_class_counts", "mapped_class_counts", "unmapped_class_counts", "pedestrian_subtypes"):
        left = Counter(base.get(counter_name, {}))
        left.update(Counter(variant.get(counter_name, {})))
        merged[counter_name] = dict(left)

    class_counts: dict[str, Counter] = {
        split: Counter() for split in ("train", "locked_validation", "purged_gap", "unspecified")
    }
    for split in class_counts:
        class_counts[split].update(Counter(base.get("class_coverage", {}).get(split, {})))
    for split in class_counts:
        # variant coverage only contributes train counts (variants are train-only).
        if split == "train":
            class_counts[split].update(Counter(variant.get("class_coverage", {}).get(split, {})))

    coverage, missing, below = _coverage(class_counts, track)
    merged["class_coverage"] = coverage
    merged["missing_classes"] = missing
    merged["below_minimum"] = below

    for key in ("points", "points_cropped"):
        merged["crop"][key] = int(base["crop"].get(key, 0)) + int(variant["crop"].get(key, 0))
    total_points = merged["crop"]["points"]
    merged["crop"]["fraction"] = None if not total_points else float(merged["crop"]["points_cropped"] / total_points)

    geom = dict(base["geometry"])
    vgeom = variant["geometry"]
    for key in (
        "boxes",
        "boxes_in_range",
        "boxes_supported",
        "z_closer_to_center",
        "z_closer_to_bottom",
        "z_ambiguous",
        "yaw_outside_pi",
        "overlapping_pairs",
    ):
        geom[key] = int(geom.get(key, 0)) + int(vgeom.get(key, 0))
    if geom["z_closer_to_center"] > geom["z_closer_to_bottom"]:
        geom["z_interpretation"] = (
            "Interior point z is closer to the cuboid center than to the bottom face."
        )
    elif geom["z_closer_to_bottom"] > geom["z_closer_to_center"]:
        geom["z_interpretation"] = (
            "Interior point z is closer to the bottom face than to the cuboid center."
        )
    else:
        geom["z_interpretation"] = "Interior point z does not favor the center or the bottom face."

    geom["dimension_medians"] = dict(base["geometry"].get("dimension_medians", {}))
    merged["geometry"] = geom
    merged["track"] = track
    merged["official_inputs_read"] = False
    merged["minimum_instances"] = MINIMUM_INSTANCES
    return merged
