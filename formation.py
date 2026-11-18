"""Coordinated-formation analysis: which drones fly the same route shape?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def _grid(df, ids):
    p = df.assign(t=df["timestamp_s"].round().astype(int))
    E = p.pivot_table(index="t", columns="drone_id", values="east_m", aggfunc="mean")[ids]
    N = p.pivot_table(index="t", columns="drone_id", values="north_m", aggfunc="mean")[ids]
    return E, N


def analyse_formation(df: pd.DataFrame, tolerance_m: float | None = None, min_overlap: int = 10) -> dict:
    """Robust, offset-invariant route-shape comparison between all drone pairs.

    distance(i, j) = median over time of |(p_i - median(p_i)) - (p_j - median(p_j))|, which ignores
    constant lane offsets and is robust to short deviations. Drones are grouped by DBSCAN on this
    precomputed matrix; tolerance is auto-derived from the data unless supplied.
    """
    ids = sorted(df["drone_id"].unique())
    out = {"available": False, "ids": ids, "note": ""}
    if len(ids) < 2:
        out["note"] = "Formation analysis needs at least two drones."
        return out
    E, N = _grid(df, ids)
    common = E.dropna(thresh=2).index
    if len(common) < min_overlap:
        out["note"] = "Drones share too little of the same timeline for formation analysis."
        return out
    D = pd.DataFrame(0.0, index=ids, columns=ids)
    cE, cN = E.median(), N.median()
    for a in ids:
        for b in ids:
            if a < b:
                m = E[a].notna() & E[b].notna()
                if m.sum() < min_overlap:
                    d = np.nan
                else:
                    d = float(np.median(np.hypot((E[a][m] - cE[a]) - (E[b][m] - cE[b]),
                                                 (N[a][m] - cN[a]) - (N[b][m] - cN[b]))))
                D.loc[a, b] = D.loc[b, a] = d
    D = D.fillna(D.max().max() if D.notna().any().any() else 1.0)
    off = D.to_numpy()[~np.eye(len(ids), dtype=bool)]
    nn = np.sort(D.to_numpy() + np.eye(len(ids)) * 1e9, axis=1)[:, 0]
    tol = float(tolerance_m) if tolerance_m else max(3.0 * float(np.median(nn)), 1.0)
    lab = DBSCAN(eps=tol, min_samples=2, metric="precomputed").fit_predict(D.to_numpy())
    groups = {int(g): [ids[i] for i in np.where(lab == g)[0]] for g in sorted(set(lab)) if g >= 0}
    out.update(available=True, distance=D, tolerance_m=tol, labels=dict(zip(ids, lab.tolist())), groups=groups)
    if not groups:
        out["note"] = "No drones flew a matching route shape within the tolerance - all routes look independent."
        out["members"] = []
        return out

    main = max(groups, key=lambda g: len(groups[g]))
    members = groups[main]
    out["main_group"], out["members"] = main, members
    refE = (E[members] - cE[members]).median(axis=1)
    refN = (N[members] - cN[members]).median(axis=1)
    dev = pd.DataFrame(index=E.index, columns=ids, dtype=float)
    for d in ids:
        dev[d] = np.hypot((E[d] - cE[d]) - refE, (N[d] - cN[d]) - refN)
    out["deviation_by_time"] = dev
    spread = np.hypot((E[members] - cE[members]).sub(refE, axis=0), (N[members] - cN[members]).sub(refN, axis=0))
    out["cohesion_by_time"] = spread.median(axis=1)
    out["cohesion_mean_m"] = float(out["cohesion_by_time"].mean())
    out["reference_track"] = pd.DataFrame({"east_m": refE, "north_m": refN})
    # centroid of formation members (absolute metres) for display
    out["centroid_by_time"] = pd.DataFrame({"east_m": E[members].mean(axis=1), "north_m": N[members].mean(axis=1)})
    out["mean_spacing_by_time"] = _mean_spacing(E[members], N[members])
    return out


def _mean_spacing(E, N):
    vals = []
    for t in E.index:
        e, n = E.loc[t].dropna().to_numpy(), N.loc[t].dropna().to_numpy()
        if len(e) < 2:
            vals.append(np.nan)
            continue
        d = np.hypot(e[:, None] - e[None, :], n[:, None] - n[None, :])
        vals.append(d[np.triu_indices(len(e), 1)].mean())
    return pd.Series(vals, index=E.index)


def row_deviation(df: pd.DataFrame, formation: dict) -> np.ndarray:
    """Per-row distance from the formation reference path (0 for non-members / unavailable)."""
    out = np.zeros(len(df))
    if not formation.get("available") or "deviation_by_time" not in formation:
        return out
    dev = formation["deviation_by_time"]
    members = set(formation["members"])
    t = df["timestamp_s"].round().astype(int).to_numpy()
    for i, (d, tt) in enumerate(zip(df["drone_id"].to_numpy(), t)):
        if d in members and tt in dev.index:
            v = dev.at[tt, d]
            out[i] = 0.0 if np.isnan(v) else v
    return out
