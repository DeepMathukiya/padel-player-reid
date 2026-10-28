"""MOT metrics: CLEAR-MOT (MOTA, IDSW), identity metrics (IDF1) and HOTA.

Implemented here (numpy-2 compatible) following the definitions used by
py-motmetrics / TrackEval:
  * CLEAR-MOT: per-frame matching at IoU >= 0.5 that keeps previous matches when
    still valid; IDSW = a GT object matched to a different predicted id than last time.
  * IDF1: one global GT-id <-> predicted-id bijection maximising identity true positives.
  * HOTA: TrackEval's algorithm, averaged over IoU thresholds 0.05 ... 0.95.
Inputs are DataFrames with columns frame, id, x1, y1, x2, y2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ALPHAS = np.arange(0.05, 0.99, 0.05)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _frames(gt: pd.DataFrame, pred: pd.DataFrame):
    gt_by = dict(tuple(gt.groupby("frame")))
    pred_by = dict(tuple(pred.groupby("frame")))
    empty = pd.DataFrame(columns=gt.columns)
    for frame in sorted(gt_by):            # evaluation is restricted to annotated frames
        g = gt_by[frame]
        p = pred_by.get(frame, empty)
        yield (frame, g["id"].to_numpy(), g[["x1", "y1", "x2", "y2"]].to_numpy(float),
               p["id"].to_numpy(), p[["x1", "y1", "x2", "y2"]].to_numpy(float))


def clear_mot(gt: pd.DataFrame, pred: pd.DataFrame, threshold: float = 0.5) -> dict:
    last_match: dict = {}           # gt id -> pred id it was last matched to
    current: dict = {}              # gt id -> pred id matched in the previous frame
    tp = fp = fn = idsw = 0
    num_gt = 0
    for _, gt_ids, gt_boxes, pred_ids, pred_boxes in _frames(gt, pred):
        num_gt += len(gt_ids)
        iou = iou_matrix(gt_boxes, pred_boxes)
        matched_gt, matched_pred, matches = set(), set(), {}
        # 1) keep last frame's matches that are still valid
        for i, g in enumerate(gt_ids):
            if g in current:
                js = np.where(pred_ids == current[g])[0]
                if len(js) and iou[i, js[0]] >= threshold:
                    matches[i] = js[0]
                    matched_gt.add(i)
                    matched_pred.add(js[0])
        # 2) Hungarian on the rest
        free_g = [i for i in range(len(gt_ids)) if i not in matched_gt]
        free_p = [j for j in range(len(pred_ids)) if j not in matched_pred]
        if free_g and free_p:
            sub = iou[np.ix_(free_g, free_p)]
            rows, cols = linear_sum_assignment(-sub)
            for r, c in zip(rows, cols):
                if sub[r, c] >= threshold:
                    matches[free_g[r]] = free_p[c]
        current = {}
        for i, j in matches.items():
            g, p = gt_ids[i], pred_ids[j]
            if g in last_match and last_match[g] != p:
                idsw += 1
            last_match[g] = p
            current[g] = p
        tp += len(matches)
        fn += len(gt_ids) - len(matches)
        fp += len(pred_ids) - len(matches)
    mota = 1 - (fn + fp + idsw) / max(num_gt, 1)
    return {"MOTA": mota, "IDSW": idsw, "FP": fp, "FN": fn, "TP": tp, "GT_dets": num_gt}


def identity_metrics(gt: pd.DataFrame, pred: pd.DataFrame, threshold: float = 0.5) -> dict:
    gt_ids = np.unique(gt["id"])
    pred_ids = np.unique(pred["id"]) if len(pred) else np.array([])
    gi = {g: i for i, g in enumerate(gt_ids)}
    pi = {p: i for i, p in enumerate(pred_ids)}
    pair_counts = np.zeros((len(gt_ids), len(pred_ids)))
    num_pred = 0
    for _, g_ids, g_boxes, p_ids, p_boxes in _frames(gt, pred):
        num_pred += len(p_ids)
        iou = iou_matrix(g_boxes, p_boxes)
        for a, b in zip(*np.where(iou >= threshold)):
            pair_counts[gi[g_ids[a]], pi[p_ids[b]]] += 1
    num_gt = len(gt)
    if pair_counts.size:
        rows, cols = linear_sum_assignment(-pair_counts)
        idtp = pair_counts[rows, cols].sum()
        mapping = {gt_ids[r]: pred_ids[c] for r, c in zip(rows, cols) if pair_counts[r, c] > 0}
    else:
        idtp, mapping = 0.0, {}
    # Correctly maintained identity: share of a GT player's frames carried by its mapped id.
    gt_frames = gt.groupby("id").size()
    maintained = {int(g): (pair_counts[gi[g], pi[mapping[g]]] / gt_frames[g] if g in mapping else 0.0)
                  for g in gt_ids}
    return {"IDF1": 2 * idtp / max(num_gt + num_pred, 1), "IDP": idtp / max(num_pred, 1),
            "IDR": idtp / max(num_gt, 1), "id_mapping": {int(k): int(v) for k, v in mapping.items()},
            "maintained_fraction": maintained}


def hota(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    gt_ids = np.unique(gt["id"])
    pred_ids = np.unique(pred["id"]) if len(pred) else np.array([0])
    gi = {g: i for i, g in enumerate(gt_ids)}
    pi = {p: i for i, p in enumerate(pred_ids)}
    potential = np.zeros((len(gt_ids), len(pred_ids)))
    gt_count = np.zeros((len(gt_ids), 1))
    pred_count = np.zeros((1, len(pred_ids)))
    frames = list(_frames(gt, pred))
    for _, g_ids, g_boxes, p_ids, p_boxes in frames:
        sim = iou_matrix(g_boxes, p_boxes)
        g_idx = np.array([gi[g] for g in g_ids], int)
        p_idx = np.array([pi[p] for p in p_ids], int)
        if len(p_idx):
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            norm = np.divide(sim, denom, out=np.zeros_like(sim), where=denom > 1e-9)
            potential[np.ix_(g_idx, p_idx)] += norm
        gt_count[g_idx] += 1
        pred_count[0, p_idx] += 1
    global_alignment = potential / np.maximum(gt_count + pred_count - potential, 1e-9)

    tp = np.zeros(len(ALPHAS))
    loc = np.zeros(len(ALPHAS))
    matches = np.zeros((len(ALPHAS), len(gt_ids), len(pred_ids)))
    num_gt, num_pred = 0, 0
    for _, g_ids, g_boxes, p_ids, p_boxes in frames:
        num_gt += len(g_ids)
        num_pred += len(p_ids)
        if len(p_ids) == 0:
            continue
        sim = iou_matrix(g_boxes, p_boxes)
        g_idx = np.array([gi[g] for g in g_ids], int)
        p_idx = np.array([pi[p] for p in p_ids], int)
        score = global_alignment[np.ix_(g_idx, p_idx)] * sim
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(ALPHAS):
            ok = sim[rows, cols] >= alpha - np.finfo(float).eps
            r, c = rows[ok], cols[ok]
            tp[a] += len(r)
            loc[a] += sim[r, c].sum()
            matches[a, g_idx[r], p_idx[c]] += 1
    fn, fp = num_gt - tp, num_pred - tp
    det_a = tp / np.maximum(tp + fn + fp, 1)
    ass_a = np.zeros(len(ALPHAS))
    for a in range(len(ALPHAS)):
        m = matches[a]
        ass = m / np.maximum(gt_count + pred_count - m, 1)
        ass_a[a] = (m * ass).sum() / max(tp[a], 1)
    hota_alpha = np.sqrt(det_a * ass_a)
    return {"HOTA": float(hota_alpha.mean()), "DetA": float(det_a.mean()), "AssA": float(ass_a.mean()),
            "LocA": float((loc / np.maximum(tp, 1)).mean())}


def evaluate_tracking(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    clear = clear_mot(gt, pred)
    ident = identity_metrics(gt, pred)
    h = hota(gt, pred)
    maintained = ident["maintained_fraction"]
    return {
        "HOTA": h["HOTA"], "DetA": h["DetA"], "AssA": h["AssA"],
        "MOTA": clear["MOTA"], "IDF1": ident["IDF1"], "IDSW": clear["IDSW"],
        "FP": clear["FP"], "FN": clear["FN"], "GT_dets": clear["GT_dets"],
        "identities_maintained_95pct": sum(v >= 0.95 for v in maintained.values()),
        "identities_total": len(maintained),
        "maintained_fraction": maintained, "id_mapping": ident["id_mapping"],
    }
