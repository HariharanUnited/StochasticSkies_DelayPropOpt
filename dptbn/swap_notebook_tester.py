"""
Lightweight swap tester for notebooks.

Paste/import this into a Jupyter cell to:
  - Load baseline model/network
  - Compute baseline expected delays on a scoped set of flights
  - Apply a swap (rewire inbound parent), reuse CPDs, rebuild inference
  - Compute benefit = sum(max(0, E0 - E_new)) over the scope

Dependencies: pgmpy, networkx (already used in the repo).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple
from copy import deepcopy
import json
import pickle
import networkx as nx

from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

from dptbn.math_engine import calculate_expected_delay
from dptbn.config import default_config

ResourceKey = Literal["k", "q", "c"]


@dataclass
class SwapSpec:
    pred1: str
    succ1: str
    pred2: str
    succ2: str
    role: Literal["aircraft", "pilot", "crew"] = "aircraft"


# ------------ Helpers ------------

def _locate(model_name: str, suffix: str) -> Path:
    try:
        base_dir = Path(__file__).parent.parent
    except NameError:
        base_dir = Path.cwd()
    roots = [
        Path("data"),
        base_dir / "data",
        base_dir / "bn-viz" / "public" / "data",
        Path("../bn-viz/public/data"),
    ]
    for r in roots:
        cand = r / f"{model_name}{suffix}"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{model_name}{suffix} not found")


def load_artifacts(model_name: str):
    net_path = _locate(model_name, "_network_opt.json")
    cpt_path = _locate(model_name, "_cpts.json")
    mdl_path = _locate(model_name, "_model.pkl")
    network = json.loads(net_path.read_text())
    with open(mdl_path, "rb") as f:
        model = pickle.load(f)
    return network, model, net_path, cpt_path, mdl_path


def _build_adj(network: Dict) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {}
    for f in network.get("flights", []):
        tgt = f.get("flight_id")
        if not tgt:
            continue
        for src in [
            f.get("inbound_aircraft_flight"),
            f.get("inbound_pilot_flight"),
            f.get("inbound_cabin_flight"),
            *(f.get("inbound_passenger_flights") or []),
        ]:
            if not src:
                continue
            adj.setdefault(src, []).append(tgt)
    return adj


def _descendants(start_nodes: List[str], adj: Dict[str, List[str]], hop_limit: int = 5) -> Set[str]:
    scope: Set[str] = set()
    queue: List[Tuple[str, int]] = [(n, 0) for n in start_nodes]
    while queue:
        node, depth = queue.pop(0)
        if depth > hop_limit:
            continue
        scope.add(node)
        for nxt in adj.get(node, []):
            if nxt not in scope:
                queue.append((nxt, depth + 1))
    return scope


def _to_dist_map(dist_obj) -> Dict[str, float]:
    vals = getattr(dist_obj, "values", None)
    if vals is None:
        return {}
    return {str(i): float(v) for i, v in enumerate(vals)}


def _expected_delay(infer: VariableElimination, node: str, bins: List[int]) -> float:
    res = infer.query([node], evidence={}, show_progress=False)
    dist = res[node] if hasattr(res, "__getitem__") else res
    return float(calculate_expected_delay(_to_dist_map(dist), bins))


def _swap_model_edges_and_cpds(model, swap: SwapSpec, res_key: ResourceKey):
    res1 = f"{swap.succ1}_{res_key}"
    res2 = f"{swap.succ2}_{res_key}"
    old_p1 = f"{swap.pred1}_t"
    old_p2 = f"{swap.pred2}_t"
    new_p1 = old_p2
    new_p2 = old_p1

    def retarget(node: str, old_parent: str, new_parent: str):
        if model.has_edge(old_parent, node):
            try:
                model.remove_edge(old_parent, node)
            except Exception:
                pass
        if not model.has_edge(new_parent, node):
            model.add_edge(new_parent, node)

        old_cpd = model.get_cpds(node)
        if not old_cpd:
            return
        values = old_cpd.get_values()
        evidence_nodes = list(getattr(old_cpd, "evidence", []) or getattr(old_cpd, "get_evidence", lambda: [])())
        if hasattr(old_cpd, "evidence_card"):
            evidence_card = list(old_cpd.evidence_card or [])
        else:
            # Fallback: infer from table width if single parent, else default to 1s
            if evidence_nodes and values.shape[1] >= 1 and len(evidence_nodes) == 1:
                evidence_card = [values.shape[1]]
            else:
                evidence_card = [1 for _ in evidence_nodes]
        state_names = getattr(old_cpd, "state_names", None)
        if state_names and old_parent in state_names:
            state_names = dict(state_names)
            state_names[new_parent] = state_names.pop(old_parent)
        new_cpd = TabularCPD(
            variable=old_cpd.variable,
            variable_card=old_cpd.variable_card,
            values=values,
            evidence=[new_parent],
            evidence_card=evidence_card,
            state_names=state_names,
        )
        try:
            model.remove_cpds(old_cpd)
        except Exception:
            pass
        model.add_cpds(new_cpd)

    retarget(res1, old_p1, new_p1)
    retarget(res2, old_p2, new_p2)


def _feasible(model, swap: SwapSpec, res_key: ResourceKey) -> bool:
    req = {
        f"{swap.succ1}_t",
        f"{swap.succ2}_t",
        f"{swap.pred1}_t",
        f"{swap.pred2}_t",
        f"{swap.succ1}_{res_key}",
        f"{swap.succ2}_{res_key}",
    }
    if any(n not in model.nodes() for n in req):
        return False
    old1 = (f"{swap.pred1}_t", f"{swap.succ1}_{res_key}")
    old2 = (f"{swap.pred2}_t", f"{swap.succ2}_{res_key}")
    new1 = (f"{swap.pred2}_t", f"{swap.succ1}_{res_key}")
    new2 = (f"{swap.pred1}_t", f"{swap.succ2}_{res_key}")
    if not model.has_edge(*old1) or not model.has_edge(*old2):
        return False
    if model.has_edge(*new1) or model.has_edge(*new2):
        return False
    return True


# ------------ Main scoring routine ------------

def score_swap(model_name: str, swap: SwapSpec, hop_limit: int = 5):
    """
    Returns benefit, deltas per node, and the mutated model.
    Benefit = sum_{i in scope} max(0, E0[i] - E1[i]).
    """
    network, base_model, _, _, _ = load_artifacts(model_name)
    res_key = {"aircraft": "k", "pilot": "q", "crew": "c"}[swap.role]

    if not _feasible(base_model, swap, res_key):
        return {"benefit": float("-inf"), "reason": "feasibility failed"}

    adj = _build_adj(network)
    scope = _descendants([swap.succ1, swap.succ2], adj, hop_limit)
    bins = list(default_config.delay_bins)

    base_infer = VariableElimination(base_model)
    base_exp = {f"{n}_t": _expected_delay(base_infer, f"{n}_t", bins) for n in scope}

    test_model = deepcopy(base_model)
    _swap_model_edges_and_cpds(test_model, swap, res_key)

    # Abort if cycle introduced
    try:
        g = test_model.to_networkx()
    except Exception:
        g = nx.DiGraph()
        g.add_edges_from(getattr(test_model, "edges", lambda: [])())
    if not nx.is_directed_acyclic_graph(g):
        return {"benefit": float("-inf"), "reason": "cycle detected"}

    test_infer = VariableElimination(test_model)
    deltas = {}
    benefit = 0.0
    for n in scope:
        node = f"{n}_t"
        new_exp = _expected_delay(test_infer, node, bins)
        d = base_exp[node] - new_exp
        deltas[node] = d
        if d > 0:
            benefit += d

    return {"benefit": benefit, "deltas": deltas, "scope": scope, "model": test_model}


# ------------ Example usage ------------
if __name__ == "__main__":
    # Example swap (adjust to your network ids)
    swap = SwapSpec(pred1="F051", succ1="F052", pred2="F091", succ2="F092", role="aircraft")
    result = score_swap("v3", swap, hop_limit=5)
    print("Benefit:", result.get("benefit"))
    if "deltas" in result:
        top = sorted(result["deltas"].items(), key=lambda x: -x[1])[:5]
        print("Top improvements:")
        for k, v in top:
            print(f"  {k}: {v:.2f}")
