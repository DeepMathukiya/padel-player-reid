"""Identity evaluation on held-out, hand-labelled tracklets.

Cheaper than full MOT ground truth: tracklets were labelled by eye from the contact
sheets (evaluation_data/heldout_tracklet_labels.yaml) and none of them seed the
gallery. Their detections were frozen into box-level ground truth
(evaluation_data/heldout_ground_truth.csv), so the evaluation does not depend on
tracklet ids and stays valid when detection is re-run. Each method's predicted box in the same frame (IoU >= 0.5) is matched,
and we report:
  * identity accuracy - share of GT boxes whose predicted id maps to the right
                        player under the best one-to-one id->player mapping
                        (fair to methods whose ids are arbitrary)
  * ID switches       - changes of the matched predicted id along each player's
                        GT sequence (ordered by frame)
  * coverage          - share of GT boxes that have any matched prediction
  * IDSW / IDF1 / MOTA / HOTA - standard MOT metrics on the same GT, with predictions
                        on unannotated players ignored. Identity terms (IDSW, IDF1,
                        AssA) are valid; detection terms (MOTA, DetA) are optimistic
                        because the GT boxes come from our own detector, so missed
                        people cannot show up as false negatives.
Long clean tracklets are the easier cases; crossings and occlusions are covered
by the MOT ground-truth clips (evaluation/ground_truth_clips.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from padel_reid.common.cache_paths import CachePaths
from padel_reid.evaluation.metrics_report import KNOWN_PREDICTIONS, _load_prediction
from padel_reid.evaluation.mot_metrics import evaluate_tracking, iou_matrix


def heldout_ground_truth(ground_truth_csv: str | Path) -> pd.DataFrame:
    """Hand-labelled boxes (frame, player, x1, y1, x2, y2), independent of tracklet ids."""
    return pd.read_csv(ground_truth_csv).rename(columns={"player": "id"})[["frame", "id", "x1", "y1", "x2", "y2"]]


def evaluate_identity(gt: pd.DataFrame, pred: pd.DataFrame, threshold: float = 0.5) -> dict:
    pred_by_frame = dict(tuple(pred[pred["frame"].isin(gt["frame"].unique())].groupby("frame")))
    matched = np.full(len(gt), -1, dtype=np.int64)
    gt_rows = gt.reset_index(drop=True)
    for frame, index in gt_rows.groupby("frame").indices.items():
        p = pred_by_frame.get(frame)
        if p is None:
            continue
        iou = iou_matrix(gt_rows.loc[index, ["x1", "y1", "x2", "y2"]].to_numpy(float),
                         p[["x1", "y1", "x2", "y2"]].to_numpy(float))
        rows, cols = linear_sum_assignment(-iou)
        for r, c in zip(rows, cols):
            if iou[r, c] >= threshold:
                matched[index[r]] = int(p["id"].iloc[c])
    gt_rows["pred"] = matched
    hit = gt_rows[gt_rows["pred"] >= 0]

    pred_ids = np.unique(hit["pred"])
    players = np.unique(gt_rows["id"])
    counts = np.array([[((hit["id"] == g) & (hit["pred"] == p)).sum() for p in pred_ids] for g in players])
    rows, cols = linear_sum_assignment(-counts) if counts.size else ([], [])
    correct = int(counts[rows, cols].sum()) if counts.size else 0

    switches = 0
    for _, seq in gt_rows.sort_values("frame").groupby("id"):
        ids = seq["pred"].to_numpy()
        ids = ids[ids >= 0]
        switches += int((ids[1:] != ids[:-1]).sum())
    return {"identity_accuracy": correct / max(len(gt_rows), 1), "id_switches": switches,
            "coverage": len(hit) / max(len(gt_rows), 1), "gt_boxes": len(gt_rows),
            "distinct_pred_ids": int(len(pred_ids))}


def drop_unannotated_predictions(gt: pd.DataFrame, pred: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Keep only predictions in GT frames that overlap a GT box ("ignore region" for unlabelled players).

    The held-out GT annotates only some players in each frame; predictions on the
    others are not false positives, so they are removed before MOT scoring.
    """
    gt_by_frame = dict(tuple(gt.groupby("frame")))
    keep = []
    for frame, group in pred[pred["frame"].isin(gt_by_frame)].groupby("frame"):
        iou = iou_matrix(group[["x1", "y1", "x2", "y2"]].to_numpy(float),
                         gt_by_frame[frame][["x1", "y1", "x2", "y2"]].to_numpy(float))
        keep.extend(group.index[iou.max(axis=1) >= threshold])
    return pred.loc[keep]


def mot_metrics_heldout(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """IDSW / IDF1 / MOTA / HOTA on the held-out GT (detection terms are optimistic: see module doc)."""
    result = evaluate_tracking(gt, drop_unannotated_predictions(gt, pred))
    return {k: result[k] for k in ("IDSW", "IDF1", "MOTA", "HOTA", "AssA", "DetA",
                                   "identities_maintained_95pct", "identities_total")}


def evaluate_heldout(cfg, paths: CachePaths, ground_truth_csv: str = "evaluation_data/heldout_ground_truth.csv") -> pd.DataFrame:
    gt = heldout_ground_truth(ground_truth_csv)
    rows = []
    for name, (file, column) in KNOWN_PREDICTIONS.items():
        path = paths.root / file
        if path.exists():
            pred = _load_prediction(path, column)
            rows.append({"method": name, **evaluate_identity(gt, pred), **mot_metrics_heldout(gt, pred)})
    report = pd.DataFrame(rows).set_index("method")
    out = Path(cfg.output_dir) / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    report.to_csv(out / "heldout_identity.csv")
    return report
