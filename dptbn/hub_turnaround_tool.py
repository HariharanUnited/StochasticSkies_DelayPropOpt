"""
Hub turnaround profile modifier (Scenario A).

Given a model name, hub code, and improvement percent, this tool:
  - Loads baseline network and CPTs from bn-viz/public/data
  - Adjusts g_priors for flights departing the hub (controllable categories reduced)
  - Writes new CPTs/network files with suffix _hub_{HUB}_{pct}
  - Builds and writes a pickle for the new CPTs
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json
import pickle

from dptbn.bn_pgmpy import build_model

DATA_DIR = Path(__file__).parent.parent / "bn-viz" / "public" / "data"


def _ensure_data_dir() -> Path:
    if not DATA_DIR.exists():
        raise FileNotFoundError("bn-viz/public/data not found")
    return DATA_DIR


def _adjust_prior(prior: Dict[str, float], factor: float) -> Dict[str, float]:
    controllable = {"Pax", "Ops", "Reactionary", "Other"}
    keep = {"ATC", "Weather"}

    newp = dict(prior)
    for k, v in prior.items():
        if k in controllable:
            newp[k] = float(v) * factor
        elif k in keep:
            newp[k] = float(v)
        else:
            newp[k] = float(v)

    total_except_none = sum(v for k, v in newp.items() if k != "None")
    none_val = 1.0 - total_except_none
    if none_val < 0:
        none_val = 0.0
    newp["None"] = none_val

    s = sum(newp.values())
    if s > 0:
        for k in list(newp.keys()):
            newp[k] = newp[k] / s
    return newp


def run_hub_turnaround(model_name: str, hub: str, pct: float) -> Dict[str, object]:
    data_dir = _ensure_data_dir()
    hub_code = hub.upper()
    factor = max(0.0, 1.0 - pct / 100.0)
    pct_tag = str(int(pct)).replace("-", "m")

    net_path = data_dir / f"{model_name}_network_opt.json"
    if not net_path.exists():
        net_path = data_dir / f"{model_name}_network.json"
    cpt_path = data_dir / f"{model_name}_cpts.json"
    if not net_path.exists() or not cpt_path.exists():
        raise FileNotFoundError("Baseline network/CPTs not found in bn-viz/public/data")

    network = json.loads(net_path.read_text())
    cpts = json.loads(cpt_path.read_text())
    g_priors = cpts.get("g_priors", {})

    affected: List[str] = [f["flight_id"] for f in network.get("flights", []) if f.get("departure_airport") == hub_code]

    for fid in affected:
        if fid in g_priors:
            g_priors[fid] = _adjust_prior(g_priors[fid], factor)

    new_suffix = f"_hub_{hub_code}_{pct_tag}"
    new_cpt_path = data_dir / f"{model_name}{new_suffix}_cpts.json"
    new_net_path = data_dir / f"{model_name}{new_suffix}_network_opt.json"
    new_pkl_path = data_dir / f"{model_name}{new_suffix}_model.pkl"

    new_cpts = dict(cpts)
    new_cpts["g_priors"] = g_priors
    new_cpt_path.write_text(json.dumps(new_cpts, indent=2))
    new_net_path.write_text(json.dumps(network, indent=2))

    model = build_model(str(new_cpt_path))
    with open(new_pkl_path, "wb") as f:
        pickle.dump(model, f)

    return {
        "new_model": f"{model_name}{new_suffix}",
        "paths": {"cpts": str(new_cpt_path), "network": str(new_net_path), "model": str(new_pkl_path)},
        "affected": affected,
    }
