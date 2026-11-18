"""Shared manifest split selection for detector training, fallback training, and scoring.

Supports explicit corpus labels (train, development, val, locked_validation,
purged_gap, test) with rejection of unknown or mixed metadata. Legacy manifests
without split fields keep the deterministic sorted 80/20 holdout. When only train
(and optional locked) splits are present, a deterministic development holdout is
carved from train for checkpoint selection; locked frames are never used for selection.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

SPLIT_TRAIN = frozenset({"train"})
SPLIT_DEVELOPMENT = frozenset({"development", "val"})
SPLIT_EXCLUDED = frozenset({"locked_validation", "purged_gap", "test"})

EVALUATION_SPLIT_ALIASES = {
    "development": SPLIT_DEVELOPMENT,
    "dev": SPLIT_DEVELOPMENT,
    "val": SPLIT_DEVELOPMENT,
    "validation": SPLIT_DEVELOPMENT,
    "locked_validation": frozenset({"locked_validation"}),
    "locked": frozenset({"locked_validation"}),
}


@dataclass
class SplitPlan:
    fit_rows: list[dict[str, Any]]
    selection_rows: list[dict[str, Any]]
    excluded_rows: list[dict[str, Any]]
    mode: str
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def fit_frame_ids(self) -> list[str]:
        return [str(r["frame_id"]) for r in self.fit_rows]

    @property
    def selection_frame_ids(self) -> list[str]:
        return [str(r["frame_id"]) for r in self.selection_rows]


def _ordered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: str(r["frame_id"]))


def derive_internal_development(
    train_rows: list[dict[str, Any]],
    *,
    fraction: float = 0.2,
    seed: int = 1234,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically hold out a development subset from explicit train rows."""
    ordered = _ordered(train_rows)
    if len(ordered) < 2:
        raise ValueError("Need at least two train-split frames to derive internal development")
    n_dev = max(1, int(round(len(ordered) * fraction)))
    n_dev = min(n_dev, len(ordered) - 1)

    def rank_key(row: dict[str, Any]) -> str:
        token = f"{seed}:{row['frame_id']}".encode("utf-8")
        return hashlib.sha256(token).hexdigest()

    ranked = sorted(ordered, key=rank_key)
    dev_rows = ranked[:n_dev]
    dev_ids = {str(r["frame_id"]) for r in dev_rows}
    fit_rows = [r for r in ordered if str(r["frame_id"]) not in dev_ids]
    return dev_rows, fit_rows


def resolve_training_splits(
    rows: list[dict[str, Any]],
    *,
    min_fit: int = 4,
    dev_fraction: float = 0.2,
    seed: int = 1234,
) -> SplitPlan:
    """Partition manifest rows for fitting, checkpoint selection, and excluded pools."""
    if len(rows) < min_fit:
        raise ValueError(f"At least {min_fit} manifest rows required for split resolution")

    ordered = _ordered(rows)
    has_split = [("split" in r and r.get("split") is not None) for r in ordered]
    if any(has_split) and not all(has_split):
        raise ValueError("Mixed split metadata: every manifest row must include split or none")

    if not any(has_split):
        cut = max(1, int(0.8 * len(ordered)))
        return SplitPlan(
            fit_rows=ordered[:cut],
            selection_rows=ordered[cut:],
            excluded_rows=[],
            mode="legacy_80_20",
            report={
                "legacy_cut_fraction": 0.8,
                "fit_count": cut,
                "selection_count": len(ordered) - cut,
                "excluded_count": 0,
            },
        )

    fit = [r for r in ordered if r.get("split") in SPLIT_TRAIN]
    selection = [r for r in ordered if r.get("split") in SPLIT_DEVELOPMENT]
    excluded = [r for r in ordered if r.get("split") in SPLIT_EXCLUDED]
    unknown = [r for r in ordered if r not in fit and r not in selection and r not in excluded]
    if unknown:
        labels = sorted({r.get("split") for r in unknown})
        raise ValueError(f"Unrecognized split labels: {labels}")

    mode = "explicit"
    report: dict[str, Any] = {
        "fit_count": len(fit),
        "selection_count": len(selection),
        "excluded_count": len(excluded),
        "excluded_splits": sorted({r.get("split") for r in excluded}),
    }

    if not selection and fit:
        dev_rows, fit_rows = derive_internal_development(fit, fraction=dev_fraction, seed=seed)
        selection = dev_rows
        fit = fit_rows
        mode = "internal_dev_from_train"
        report.update(
            {
                "internal_development_fraction": dev_fraction,
                "internal_development_seed": seed,
                "internal_development_frame_ids": [str(r["frame_id"]) for r in dev_rows],
                "note": (
                    "No development/val split in manifest; deterministic holdout carved from train. "
                    "locked_validation and purged_gap remain excluded from fitting and selection."
                ),
            }
        )

    if len(fit) < 1:
        raise ValueError("No train-split frames available after split resolution")
    if not selection:
        raise ValueError("No development or validation frames available for checkpoint selection")

    report["fit_count"] = len(fit)
    report["selection_count"] = len(selection)
    return SplitPlan(
        fit_rows=fit,
        selection_rows=selection,
        excluded_rows=excluded,
        mode=mode,
        report=report,
    )


def rows_for_evaluation(plan: SplitPlan, evaluation_split: str = "development") -> list[dict[str, Any]]:
    """Return manifest rows for scoring; default development never includes locked_validation."""
    key = evaluation_split.strip().lower()
    if key not in EVALUATION_SPLIT_ALIASES:
        raise ValueError(
            f"Unsupported evaluation_split {evaluation_split!r}; "
            f"expected one of {sorted(EVALUATION_SPLIT_ALIASES)}"
        )
    allowed = EVALUATION_SPLIT_ALIASES[key]
    if allowed <= SPLIT_DEVELOPMENT:
        if not plan.selection_rows:
            raise ValueError("No development/validation rows available for evaluation")
        return list(plan.selection_rows)

    locked = [r for r in plan.excluded_rows if r.get("split") in allowed]
    if not locked:
        raise ValueError("No locked_validation rows available for evaluation")
    return locked


def legacy_selection_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Legacy 80/20 validation slice when callers only need evaluation rows."""
    ordered = _ordered(rows)
    return ordered[int(0.8 * len(ordered)) :]
