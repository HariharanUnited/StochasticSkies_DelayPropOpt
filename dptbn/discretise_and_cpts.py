"""
Discretization and CPT construction helpers.
"""
from typing import Dict, List, Sequence, Optional, Tuple, Any
import numpy as np
import pandas as pd


def discretize_value(value: float, bins: Sequence[int]) -> int:
    return int(np.digitize([value], bins, right=False)[0])


def discretize_day_results(
    day_results: Dict[str, Dict[str, float]], bins: Sequence[int], numeric_keys: Optional[Sequence[str]] = None
) -> Dict[str, Dict[str, int]]:
    """
    Convert continuous delays for a day into bin indices per variable.
    """
    discrete: Dict[str, Dict[str, int]] = {}
    for fid, vals in day_results.items():
        out: Dict[str, int] = {}
        for var, val in vals.items():
            if numeric_keys is not None and var not in numeric_keys:
                continue
            if isinstance(val, (int, float)):
                out[var] = discretize_value(val, bins)
        discrete[fid] = out
    return discrete


def discretize_many(
    days_results: List[Dict[str, Dict[str, float]]], bins: Sequence[int], numeric_keys: Optional[Sequence[str]] = None
) -> List[Dict[str, Dict[str, int]]]:
    return [discretize_day_results(day, bins, numeric_keys=numeric_keys) for day in days_results]


def build_dataset(
    days_continuous: List[Dict[str, Dict[str, Any]]],
    bins: Sequence[int],
    numeric_keys: Optional[Sequence[str]] = None,
    flight_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a pandas DataFrame with continuous and binned variables per day/flight.
    """
    numeric_keys = numeric_keys or ["t", "k", "k_eff", "q", "c", "p", "r", "b"]
    records = []
    for day_idx, day in enumerate(days_continuous):
        for fid, vals in day.items():
            if flight_ids and fid not in flight_ids:
                continue
            rec = {"day": day_idx, "flight_id": fid}
            for k in numeric_keys:
                rec[k] = vals.get(k, 0.0)
                rec[f"{k}_bin"] = discretize_value(rec[k], bins)
            rec["g"] = vals.get("g", "None")
            rec["base_cat"] = vals.get("base_cat", "None")
            records.append(rec)
    return pd.DataFrame.from_records(records)


def _all_parent_states(df_f: pd.DataFrame, parents: List[str], g_states: Optional[List[str]] = None) -> List[Tuple]:
    domains = []
    for p in parents:
        if p == "g" and g_states is not None:
            domains.append(sorted(g_states))
        else:
            domains.append(sorted(df_f[p].unique()))
    grids = np.array(np.meshgrid(*domains, indexing="ij"), dtype=object)
    combos = grids.reshape(len(parents), -1).T
    return [tuple(c) for c in combos]


def _full_parent_states(num_bins: int, g_states: List[str], parents: List[str]) -> List[Tuple]:
    domains = []
    for p in parents:
        if p == "g":
            domains.append(g_states)
        else:
            domains.append(list(range(num_bins)))
    grids = np.array(np.meshgrid(*domains, indexing="ij"), dtype=object)
    combos = grids.reshape(len(parents), -1).T
    return [tuple(c) for c in combos]


def compute_cpt_for_flight(
    df: pd.DataFrame,
    flight_id: str,
    target: str = "t_bin",
    parents: Optional[List[str]] = None,
    smoothing: float = 1.0,
    complete: bool = True,
    complete_mode: str = "observed",
    num_bin_states: Optional[int] = None,
    g_states: Optional[List[str]] = None,
) -> Dict[Tuple, Dict[int, float]]:
    """
    Compute CPT P(target | parents) for a single flight using Laplace smoothing.
    Returns {parent_tuple: {target_value: prob}}
    If complete=True, include all parent combos seen in the flight data (and all g_states if provided),
    filling missing combos with smoothed priors.
    complete_mode: "observed" uses domains observed for this flight; "full" uses full bin range (0..num_bin_states-1) and all g_states.
    """
    parents = parents or ["k_bin", "q_bin", "c_bin", "p_bin", "g"]
    df_f = df[df["flight_id"] == flight_id]
    if df_f.empty:
        return {}
    parent_cols = parents
    target_values = sorted(df_f[target].unique())
    cpt: Dict[Tuple, Dict[int, float]] = {}
    grouped = df_f.groupby(parent_cols)
    # Precompute all parent states if completing
    all_parents: List[Tuple] = []
    if complete:
        if complete_mode == "full":
            if num_bin_states is None or g_states is None:
                raise ValueError("num_bin_states and g_states required for complete_mode='full'")
            all_parents = _full_parent_states(num_bin_states, g_states, parents)
        else:
            all_parents = _all_parent_states(df_f, parents, g_states=g_states)
    seen = set()
    for parent_vals, group in grouped:
        counts = group[target].value_counts().to_dict()
        total = sum(counts.values()) + smoothing * len(target_values)
        probs = {}
        if total == 0:
            probs = {tval: 1.0 / len(target_values) for tval in target_values}
        else:
            for tval in target_values:
                probs[tval] = (counts.get(tval, 0) + smoothing) / total
        key = parent_vals if isinstance(parent_vals, tuple) else (parent_vals,)
        cpt[key] = probs
        seen.add(key)
    if complete:
        for parent_vals in all_parents:
            key = parent_vals if isinstance(parent_vals, tuple) else (parent_vals,)
            if key in seen:
                continue
            total = smoothing * len(target_values)
            if total == 0:
                probs = {tval: 1.0 / len(target_values) for tval in target_values}
            else:
                probs = {tval: smoothing / total for tval in target_values}
            cpt[key] = probs
    return cpt


def compute_cpts(
    df: pd.DataFrame,
    target: str = "t_bin",
    parents: Optional[List[str]] = None,
    smoothing: float = 1.0,
    complete: bool = True,
    complete_mode: str = "observed",
    num_bin_states: Optional[int] = None,
) -> Dict[str, Dict[Tuple, Dict[int, float]]]:
    """
    Compute CPTs for all flights in the dataset.
    Returns {flight_id: {parent_tuple: {target: prob}}}
    If complete=True, all parent combos are included with smoothed priors if unseen.
    """
    parents = parents or ["k_bin", "q_bin", "c_bin", "p_bin", "g"]
    g_states = sorted(df["g"].unique())
    if complete_mode == "full" and num_bin_states is None:
        raise ValueError("num_bin_states required for complete_mode='full'")
    cpts = {}
    for fid in df["flight_id"].unique():
        cpts[fid] = compute_cpt_for_flight(
            df,
            fid,
            target=target,
            parents=parents,
            smoothing=smoothing,
            complete=complete,
            complete_mode=complete_mode,
            num_bin_states=num_bin_states,
            g_states=g_states,
        )
    return cpts


def discretize_value_edges(bin_idx: int, bins: Sequence[int]) -> float:
    if bin_idx <= 0:
        return 0.0
    if bin_idx >= len(bins):
        return float(bins[-1])
    return float(bins[bin_idx - 1])
