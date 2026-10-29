"""Freeze hand-labelled tracklets into box-level ground truth.

Tracklet ids are only valid for one cached detection run; the boxes are not tied
to ids, so the frozen CSV (frame, player, x1, y1, x2, y2) stays a valid ground
truth after detection is re-run. Same format as evaluation_data/heldout_ground_truth.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from padel_reid.evaluation.tracklet_annotation import load_labels


def freeze_tracklet_labels(labels_file: str | Path, tracklets_parquet: str | Path, out_csv: str | Path) -> Path:
    labels, _ = load_labels(labels_file)
    table = pd.read_parquet(tracklets_parquet)
    boxes = table[table["tracklet_id"].isin(labels)].copy()
    missing = set(labels) - set(boxes["tracklet_id"].unique())
    if missing:
        raise ValueError(f"tracklets {sorted(missing)} are not in {tracklets_parquet}: "
                         "the labels belong to a different detection run")
    boxes["player"] = boxes["tracklet_id"].map(labels) + 1
    out = boxes[["frame", "player", "x1", "y1", "x2", "y2"]].round(1).sort_values(["player", "frame"])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv
