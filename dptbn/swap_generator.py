"""
Swap candidate generator for rerouting/rewiring (aircraft/pilot/crew).

Given a model name (e.g., "v3" or "v3_act1"), this module loads the corresponding
network_opt JSON and enumerates feasible swap options between two chains.

Notation (aircraft swaps):
    Tail A: fi -> fj   (fj is the immediate successor of fi on tail A)
    Tail B: fk -> fl   (fl is the immediate successor of fk on tail B)
Proposed swap rewires edges to:
    fi -> fl   and   fk -> fj

Constraints:
  - tail(fi) == tail(fj) == A, tail(fk) == tail(fl) == B, A != B
  - Stations: arr(fi) == dep(fl), arr(fk) == dep(fj)
  - Times:  min_ct <= dep(fl) - arr(fi) <= horizon
            min_ct <= dep(fj) - arr(fk) <= horizon
  - fi,fj are consecutive on A; fk,fl are consecutive on B
  - Exclude degenerate swaps (repeated flights or same tails)

The same structure can be reused for pilot/crew swaps by grouping chains on
pilot_id / crew_id instead of tail_id.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Set
from copy import deepcopy
import json
import pickle
import copy
import networkx as nx

from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

from dptbn.bn_pgmpy import build_model
from dptbn.math_engine import calculate_expected_delay
from dptbn.config import default_config

# Global parameters (can be tuned)
MIN_CT_DEFAULT = 25      # minutes
HORIZON_DEFAULT = 180    # minutes (3 hours)
HOP_LIMIT_DEFAULT = 5    # reachability depth for scoring

ResourceKey = Literal["k", "q", "c"]
SwapRole = Literal["aircraft", "pilot", "crew"]


@dataclass
class FlightRec:
    flight_id: str
    dep: str
    arr: str
    dep_time: int
    arr_time: int
    tail_id: Optional[str] = None
    pilot_id: Optional[str] = None
    crew_id: Optional[str] = None


@dataclass
class SwapOption:
    pred1: str
    succ1: str
    pred2: str
    succ2: str
    chain1: str
    chain2: str
    role: Literal["aircraft", "pilot", "crew"]
    benefit: float = 0.0  # placeholder until scoring


def swap_key(s: SwapOption) -> str:
    """Stable identifier used by the UI."""
    return f"{s.role}|{s.chain1}|{s.pred1}->{s.succ1}||{s.chain2}|{s.pred2}->{s.succ2}"


def _locate_network_opt(model_name: str) -> Path:
    base = Path("bn-viz") / "public" / "data"
    opt = base / f"{model_name}_network_opt.json"
    raw = base / f"{model_name}_network.json"
    if opt.exists():
        return opt
    if raw.exists():
        return raw
    raise FileNotFoundError(f"network_opt file not found for model '{model_name}'")


import numpy as np
import networkx as nx
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def _as_nx_digraph(model) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(list(model.nodes()))
    g.add_edges_from(list(model.edges()))
    return g

def _would_create_cycle(g: nx.DiGraph, u: str, v: str) -> bool:
    # adding u->v creates a cycle iff there's already a path v -> u
    if u == v:
        return True
    if v not in g or u not in g:
        return False
    return nx.has_path(g, v, u)

def _expected_delay(infer: VariableElimination, node: str, bins: List[int]) -> float:
    res = infer.query([node], evidence={}, show_progress=False)
    dist = res[node] if hasattr(res, "__getitem__") else res
    values = getattr(dist, "values", None)
    if values is None:
        return 0.0
    return float(calculate_expected_delay({str(i): float(v) for i, v in enumerate(values)}, bins))

def _apply_swap_in_place_single_parent(model, swap: SwapOption, role: str):
    """
    In-place: rewire old_parent->res_node to new_parent->res_node AND
    retarget the CPD evidence to the new parent, keeping values identical.
    Returns an undo() closure.
    Assumes resource node has exactly one parent.
    """
    res_key = {"aircraft": "k", "pilot": "q", "crew": "c"}[role]

    res1 = f"{swap.succ1}_{res_key}"
    res2 = f"{swap.succ2}_{res_key}"
    old_p1 = f"{swap.pred1}_t"
    old_p2 = f"{swap.pred2}_t"
    new_p1 = old_p2
    new_p2 = old_p1

    # capture state needed to undo
    undo_ops = []

    def retarget(child: str, old_parent: str, new_parent: str):
        # --- edge swap ---
        if model.has_edge(old_parent, child):
            model.remove_edge(old_parent, child)
            undo_ops.append(lambda: model.add_edge(old_parent, child))
        else:
            undo_ops.append(lambda: None)

        model.add_edge(new_parent, child)
        undo_ops.append(lambda: model.remove_edge(new_parent, child))

        # --- CPD swap ---
        old_cpd = model.get_cpds(child)
        if old_cpd is None:
            undo_ops.append(lambda: None)
            return

        old_evs = list(old_cpd.get_evidence())
        if len(old_evs) != 1 or old_evs[0] != old_parent:
            raise ValueError(f"{child} expected single evidence [{old_parent}], got {old_evs}")

        values = old_cpd.get_values()
        new_parent_card = model.get_cardinality(new_parent)

        if values.shape[1] != new_parent_card:
            raise ValueError(f"{child} CPT width {values.shape[1]} != card({new_parent}) {new_parent_card}")

        state_names = getattr(old_cpd, "state_names", None)
        if state_names and old_parent in state_names:
            state_names = dict(state_names)
            state_names[new_parent] = state_names.pop(old_parent)

        new_cpd = TabularCPD(
            variable=old_cpd.variable,
            variable_card=old_cpd.variable_card,
            values=values,
            evidence=[new_parent],
            evidence_card=[new_parent_card],
            state_names=state_names,
        )

        model.remove_cpds(old_cpd)
        model.add_cpds(new_cpd)

        def _undo_cpd():
            model.remove_cpds(new_cpd)
            model.add_cpds(old_cpd)

        undo_ops.append(_undo_cpd)

    retarget(res1, old_p1, new_p1)
    retarget(res2, old_p2, new_p2)

    def undo():
        # reverse order
        for op in reversed(undo_ops):
            op()

    return undo


def _locate_cpts(model_name: str) -> Path:
    base = Path("bn-viz") / "public" / "data"
    cand = base / f"{model_name}_cpts.json"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"CPT file not found for model '{model_name}'")


def _locate_model_pkl(model_name: str) -> Optional[Path]:
    cand = Path("bn-viz") / "public" / "data" / f"{model_name}_model.pkl"
    if cand.exists():
        return cand
    return None

from copy import deepcopy
from typing import Dict, Tuple, Literal, Optional

SwapRole = Literal["aircraft", "pilot", "crew"]

def _ensure_resource_entry(resource_cpds: Dict, fid: str):
    """
    Make sure resource_cpds has the standard skeleton for this flight.
    We do NOT touch parents_order/pax_parents unless missing.
    """
    if fid not in resource_cpds:
        resource_cpds[fid] = {
            "k": {}, "q": {}, "c": {}, "pax": {},
            "pax_parents": [],
            "parents_order": ["k_bin", "q_bin", "c_bin", "g"],
        }
    else:
        resource_cpds[fid].setdefault("k", {})
        resource_cpds[fid].setdefault("q", {})
        resource_cpds[fid].setdefault("c", {})
        resource_cpds[fid].setdefault("pax", {})
        resource_cpds[fid].setdefault("pax_parents", [])
        resource_cpds[fid].setdefault("parents_order", ["k_bin", "q_bin", "c_bin", "g"])


def _retarget_one_parent_resource_cpd(
    cpts_data: Dict,
    fid: str,
    res_key: str,
    old_parent: Optional[str],
    new_parent: Optional[str],
    strict: bool = False,
) -> bool:
    """
    Move the CPD table for (fid,res_key) from old_parent -> new_parent.

    Assumption (your case): each resource has exactly ONE parent.
    So the dict should look like: resource_cpds[fid][res_key] == {old_parent: CPD_table}.
    """
    if not old_parent or not new_parent or old_parent == new_parent:
        return False

    resource_cpds = cpts_data.setdefault("resource_cpds", {})
    _ensure_resource_entry(resource_cpds, fid)
    m = resource_cpds[fid].setdefault(res_key, {})

    table = m.pop(old_parent, None)

    # fallback: if caller's old_parent doesn't match but there's exactly one entry, treat that as the old parent
    if table is None and len(m) == 1:
        only_parent, table = next(iter(m.items()))
        m.pop(only_parent, None)

    if table is None:
        if strict:
            raise KeyError(f"Missing CPD table for {fid}:{res_key} old_parent={old_parent}")
        return False

    m[new_parent] = table
    return True


def update_cpts_after_network_change(
    base_cpts: Dict,
    base_network: Dict,
    new_network: Dict,
    role: SwapRole,
    strict: bool = False,
) -> Tuple[Dict, int]:
    """
    Update CPTs by diffing inbound parent changes in network_opt.

    For every flight whose inbound_<role>_flight changed:
      resource_cpds[flight][k/q/c] remaps from old_parent -> new_parent
    """
    inbound_field = {
        "aircraft": "inbound_aircraft_flight",
        "pilot": "inbound_pilot_flight",
        "crew": "inbound_cabin_flight",
    }[role]
    res_key = {"aircraft": "k", "pilot": "q", "crew": "c"}[role]

    old_map = {f["flight_id"]: f.get(inbound_field) for f in base_network.get("flights", []) if f.get("flight_id")}
    new_map = {f["flight_id"]: f.get(inbound_field) for f in new_network.get("flights", []) if f.get("flight_id")}

    out = deepcopy(base_cpts)
    changed = 0

    # iterate flights present in new network
    for fid, new_parent in new_map.items():
        old_parent = old_map.get(fid)
        if old_parent != new_parent:
            did = _retarget_one_parent_resource_cpd(out, fid, res_key, old_parent, new_parent, strict=strict)
            if did:
                changed += 1

    return out, changed


def _load_flights(path: Path) -> List[FlightRec]:
    data = json.loads(path.read_text())
    flights = []
    for f in data.get("flights", []):
        flights.append(
            FlightRec(
                flight_id=f.get("flight_id"),
                dep=f.get("departure_airport"),
                arr=f.get("arrival_airport"),
                dep_time=f.get("scheduled_departure"),
                arr_time=f.get("scheduled_arrival"),
                tail_id=f.get("tail_id"),
                pilot_id=f.get("pilot_id"),
                crew_id=f.get("crew_id"),
            )
        )
    return flights


def _build_chains(flights: List[FlightRec], role: Literal["aircraft", "pilot", "crew"]) -> Dict[str, List[FlightRec]]:
    key = {"aircraft": "tail_id", "pilot": "pilot_id", "crew": "crew_id"}[role]
    chains: Dict[str, List[FlightRec]] = {}
    for f in flights:
        chain_id = getattr(f, key)
        if not chain_id:
            continue
        chains.setdefault(chain_id, []).append(f)
    # sort each chain by departure time
    for cid in chains:
        chains[cid].sort(key=lambda x: x.dep_time)
    return chains


def _feasible_time_gap(arr_time: int, dep_time: int, min_ct: int, horizon: int) -> bool:
    gap = dep_time - arr_time
    return min_ct <= gap <= horizon


def list_swaps(
    model_name: str,
    role: Literal["aircraft", "pilot", "crew"] = "aircraft",
    min_ct: int = MIN_CT_DEFAULT,
    horizon: int = HORIZON_DEFAULT,
    limit: Optional[int] = None,
) -> List[SwapOption]:
    """
    Enumerate feasible swap options for the given model/network.
    Returns SwapOption records with benefit=0.0.
    """
    path = _locate_network_opt(model_name)
    flights = _load_flights(path)
    chains = _build_chains(flights, role)

    # quick access map
    flight_by_id = {f.flight_id: f for f in flights}
    swaps: List[SwapOption] = []

    chain_ids = list(chains.keys())
    for i in range(len(chain_ids)):
        chain1_id = chain_ids[i]
        chain1 = chains[chain1_id]
        for j in range(i + 1, len(chain_ids)):
            chain2_id = chain_ids[j]
            chain2 = chains[chain2_id]

            # iterate consecutive edges on chain1
            for idx1 in range(len(chain1) - 1):
                fi = chain1[idx1]
                fj = chain1[idx1 + 1]

                # consecutive edges on chain2
                for idx2 in range(len(chain2) - 1):
                    fk = chain2[idx2]
                    fl = chain2[idx2 + 1]

                    # station continuity
                    if fi.arr != fl.dep:
                        continue
                    if fk.arr != fj.dep:
                        continue

                    # time feasibility
                    if not _feasible_time_gap(fi.arr_time, fl.dep_time, min_ct, horizon):
                        continue
                    if not _feasible_time_gap(fk.arr_time, fj.dep_time, min_ct, horizon):
                        continue

                    # avoid degenerate cases
                    if len({fi.flight_id, fj.flight_id, fk.flight_id, fl.flight_id}) < 4:
                        continue

                    swaps.append(
                        SwapOption(
                            pred1=fi.flight_id,
                            succ1=fj.flight_id,
                            pred2=fk.flight_id,
                            succ2=fl.flight_id,
                            chain1=chain1_id,
                            chain2=chain2_id,
                            role=role,
                            benefit=0.0,
                        )
                    )

                    if limit and len(swaps) >= limit:
                        return swaps
    return swaps


def _ensure_res_entry(res_cpds: Dict[str, Dict[str, Dict]], fid: str):
    res_cpds.setdefault(
        fid,
        {
            "k": {},
            "q": {},
            "c": {},
            "pax": {},
            "pax_parents": [],
            "parents_order": ["k", "q", "c", "g"],
        },
    )


def apply_swap_to_network_data(network_data: Dict, swap: SwapOption, role: Literal["aircraft", "pilot", "crew"]) -> Dict:
    net = deepcopy(network_data)
    flights = {f["flight_id"]: f for f in net.get("flights", [])}
    inbound_field = {"aircraft": "inbound_aircraft_flight", "pilot": "inbound_pilot_flight", "crew": "inbound_cabin_flight"}[role]

    if swap.succ1 in flights:
        flights[swap.succ1][inbound_field] = swap.pred2
    if swap.succ2 in flights:
        flights[swap.succ2][inbound_field] = swap.pred1
    return net


def apply_swap_to_cpts_data(cpts_data: Dict, swap: SwapOption, role: Literal["aircraft", "pilot", "crew"]) -> Dict:
    """Rewire resource CPDs to follow the new inbound parents while keeping table values."""
    data = deepcopy(cpts_data)
    res_key = {"aircraft": "k", "pilot": "q", "crew": "c"}[role]
    res_cpds = data.get("resource_cpds", {})

    _ensure_res_entry(res_cpds, swap.succ1)
    _ensure_res_entry(res_cpds, swap.succ2)

    m1 = res_cpds[swap.succ1].setdefault(res_key, {})
    m2 = res_cpds[swap.succ2].setdefault(res_key, {})

    # Move the existing mapping to the new inbound parent IDs
    v1 = m1.pop(swap.pred1, None)
    v2 = m2.pop(swap.pred2, None)

    if v1 is not None:
        m1[swap.pred2] = v1
    if v2 is not None:
        m2[swap.pred1] = v2
    return data


def compute_swap_score(network_data: Dict, swap: SwapOption) -> float:
    """Deprecated placeholder (kept for compatibility)."""
    return 0.0


def _retarget_resource(model, child_id: str, res_key: ResourceKey, new_parent: str):
    """Retarget a resource CPD/edge to a new inbound parent."""
    res_node = f"{child_id}_{res_key}"
    old_parent = None
    for u, v in list(model.edges()):
        if v == res_node and u.endswith("_t"):
            old_parent = u
            try:
                model.remove_edge(u, v)
            except Exception:
                pass
    new_parent_node = f"{new_parent}_t"
    if not model.has_edge(new_parent_node, res_node):
        model.add_edge(new_parent_node, res_node)

    cpd = model.get_cpds(res_node)
    if not cpd:
        return
    values = cpd.get_values()
    evidence_nodes = list(getattr(cpd, "evidence", []) or getattr(cpd, "get_evidence", lambda: [])())
    if hasattr(cpd, "evidence_card"):
        evidence_card = list(cpd.evidence_card or [])
    else:
        if evidence_nodes and values.shape[1] >= 1 and len(evidence_nodes) == 1:
            evidence_card = [values.shape[1]]
        else:
            evidence_card = [1 for _ in evidence_nodes]
    state_names = getattr(cpd, "state_names", None)
    if state_names and old_parent and old_parent in state_names:
        state_names = dict(state_names)
        state_names[new_parent_node] = state_names.pop(old_parent)
    new_cpd = TabularCPD(
        variable=cpd.variable,
        variable_card=cpd.variable_card,
        values=values,
        evidence=[new_parent_node],
        evidence_card=evidence_card,
        state_names=state_names,
    )
    try:
        model.remove_cpds(cpd)
    except Exception:
        pass
    model.add_cpds(new_cpd)


def _swap_model_edges_and_cpds(model, swap: SwapOption, role: Literal["aircraft", "pilot", "crew"]):
    """Adjust the in-memory pgmpy model: swap inbound parent edges and retarget the resource CPDs."""
    res_key = {"aircraft": "k", "pilot": "q", "crew": "c"}[role]

    def retarget(node: str, old_parent: str, new_parent: str):
        # Edges
        if model.has_edge(old_parent, node):
            try:
                model.remove_edge(old_parent, node)
            except Exception:
                pass
        if not model.has_edge(new_parent, node):
            model.add_edge(new_parent, node)

        # CPD
        old_cpd = model.get_cpds(node)
        if not old_cpd:
            return
        values = old_cpd.get_values()
        evidence_card = old_cpd.evidence_card or []
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

    res1 = f"{swap.succ1}_{res_key}"
    res2 = f"{swap.succ2}_{res_key}"
    old_p1 = f"{swap.pred1}_t"
    old_p2 = f"{swap.pred2}_t"
    new_p1 = old_p2
    new_p2 = old_p1

    retarget(res1, old_p1, new_p1)
    retarget(res2, old_p2, new_p2)


def _build_adj_from_network(network_data: Dict) -> Dict[str, List[str]]:
    """Flight-to-flight adjacency using all inbound arcs."""
    adj: Dict[str, List[str]] = {}
    for f in network_data.get("flights", []):
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


def _descendants(start_nodes: List[str], adj: Dict[str, List[str]], hop_limit: int) -> Set[str]:
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


def _to_marginal_map(dist_obj) -> Dict[str, float]:
    if dist_obj is None:
        return {}
    values = getattr(dist_obj, "values", None)
    if values is None:
        return {}
    return {str(i): float(v) for i, v in enumerate(values)}


def _expected_delay(infer: VariableElimination, node: str, bins: List[int]) -> float:
    try:
        res = infer.query([node], evidence={}, show_progress=False)
        dist = res[node] if hasattr(res, "__getitem__") else res
        return float(calculate_expected_delay(_to_marginal_map(dist), bins))
    except Exception:
        return 0.0


def _feasible_swap(model, swap: SwapOption, role: Literal["aircraft", "pilot", "crew"]) -> bool:
    res_key = {"aircraft": "k", "pilot": "q", "crew": "c"}[role]
    req_nodes = {
        f"{swap.succ1}_t",
        f"{swap.succ1}_{res_key}",
        f"{swap.succ2}_t",
        f"{swap.succ2}_{res_key}",
        f"{swap.pred1}_t",
        f"{swap.pred2}_t",
    }
    if any(n not in model.nodes() for n in req_nodes):
        return False

    res1 = f"{swap.succ1}_{res_key}"
    res2 = f"{swap.succ2}_{res_key}"
    old1 = (f"{swap.pred1}_t", res1)
    old2 = (f"{swap.pred2}_t", res2)
    new1 = (f"{swap.pred2}_t", res1)
    new2 = (f"{swap.pred1}_t", res2)

    if not model.has_edge(*old1) or not model.has_edge(*old2):
        return False
    if model.has_edge(*new1) or model.has_edge(*new2):
        return False
    return True


def _compute_scope(swap: SwapOption, adj: Dict[str, List[str]], hop_limit: int) -> Set[str]:
    touched = [swap.succ1, swap.succ2]
    return _descendants(touched, adj, hop_limit)


def _rebuild_chains_by_role(network_data: Dict, role: SwapRole) -> Dict[str, List[Dict]]:
    key = {"aircraft": "tail_id", "pilot": "pilot_id", "crew": "crew_id"}[role]
    chains: Dict[str, List[Dict]] = {}
    for f in network_data.get("flights", []):
        cid = f.get(key)
        if not cid:
            continue
        chains.setdefault(cid, []).append(f)
    for cid in chains:
        chains[cid].sort(key=lambda x: x["scheduled_departure"])
    return chains


def reroute_network_and_model(
    network_data: Dict,
    model,
    swap: SwapOption,
    role: SwapRole,
) -> Tuple[Dict, object]:
    """
    Apply a swap to network (inbound fields + role id propagation) and model (edges/CPDs) in-memory.
    Returns (updated_network, updated_model).
    """
    key = {"aircraft": "tail_id", "pilot": "pilot_id", "crew": "crew_id"}[role]
    inbound_field = {
        "aircraft": "inbound_aircraft_flight",
        "pilot": "inbound_pilot_flight",
        "crew": "inbound_cabin_flight",
    }[role]

    net = deepcopy(network_data)
    flights = {f["flight_id"]: f for f in net.get("flights", [])}
    chains = _rebuild_chains_by_role(net, role)

    if swap.pred1 not in flights or swap.pred2 not in flights or swap.succ1 not in flights or swap.succ2 not in flights:
        raise ValueError("Missing flights for swap")

    chain1_id = flights[swap.pred1].get(key)
    chain2_id = flights[swap.pred2].get(key)
    if chain1_id not in chains or chain2_id not in chains:
        raise ValueError("Chain ids not found")

    chain1 = chains[chain1_id]
    chain2 = chains[chain2_id]

    def idx(chain, fid):
        for i, f in enumerate(chain):
            if f["flight_id"] == fid:
                return i
        return -1

    i1 = idx(chain1, swap.pred1)
    j1 = idx(chain1, swap.succ1)
    i2 = idx(chain2, swap.pred2)
    j2 = idx(chain2, swap.succ2)
    if i1 == -1 or j1 == -1 or i2 == -1 or j2 == -1 or j1 != i1 + 1 or j2 != i2 + 1:
        raise ValueError("Swap flights not consecutive")

    chain1_pre = chain1[: i1 + 1]  # includes pred1
    chain1_post = chain1[j1 + 1 :]  # after succ1
    chain2_pre = chain2[: i2 + 1]  # includes pred2
    chain2_post = chain2[j2 + 1 :]  # after succ2 (succ2 itself is chain2[j2])

    new_chain1 = chain1_pre + [chain2[j2]] + chain2_post
    new_chain2 = chain2_pre + [chain1[j1]] + chain1_post

    # flights moving to chain1 (including succ2 and chain2_post)
    start_idx1 = len(chain1_pre)
    start_idx2 = len(chain2_pre)

    moved_to_chain1 = {f["flight_id"] for f in new_chain1[start_idx1:]}
    moved_to_chain2 = {f["flight_id"] for f in new_chain2[start_idx2:]}

    # Update ids and inbound for chain1 suffix
    for idx_n, f in enumerate(new_chain1):
        fid = f["flight_id"]
        if fid in flights:
            if idx_n >= start_idx1:
                flights[fid][key] = chain1_id
                flights[fid][inbound_field] = new_chain1[idx_n - 1]["flight_id"] if idx_n > 0 else flights[fid].get(inbound_field)
    # Update ids and inbound for chain2 suffix
    for idx_n, f in enumerate(new_chain2):
        fid = f["flight_id"]
        if fid in flights:
            if idx_n >= start_idx2:
                flights[fid][key] = chain2_id
                flights[fid][inbound_field] = new_chain2[idx_n - 1]["flight_id"] if idx_n > 0 else flights[fid].get(inbound_field)

    # Retarget model edges/CPDs for flights whose inbound changed
    updated_model = copy.deepcopy(model)
    res_key = {"aircraft": "k", "pilot": "q", "crew": "c"}[role]

    # Chain1 suffix (excluding first element if no predecessor change)
    for idx_n in range(start_idx1, len(new_chain1)):
        fid = new_chain1[idx_n]["flight_id"]
        if idx_n == 0:
            continue
        new_parent = new_chain1[idx_n - 1]["flight_id"]
        _retarget_resource(updated_model, fid, res_key, new_parent)

    for idx_n in range(start_idx2, len(new_chain2)):
        fid = new_chain2[idx_n]["flight_id"]
        if idx_n == 0:
            continue
        new_parent = new_chain2[idx_n - 1]["flight_id"]
        _retarget_resource(updated_model, fid, res_key, new_parent)

    return net, updated_model


def score_swaps_fast(
    model_name: str,
    swaps: List[Dict[str, object]],
    role: SwapRole = "aircraft",
    hop_limit: int = HOP_LIMIT_DEFAULT,
) -> List[Dict[str, object]]:
    """
    Fast, in-memory scoring:
      - load baseline network/model from data/
      - precompute baseline expected delays on union scope
      - for each swap: reroute network/model in-memory, run inference on scope, compute benefit
    No files are written.
    """
    net_path = _locate_network_opt(model_name)
    model_path = _locate_model_pkl(model_name)
    if not model_path or not model_path.exists():
        raise FileNotFoundError(f"Model pickle not found for {model_name}")

    base_network = json.loads(net_path.read_text())
    with open(model_path, "rb") as f:
        base_model = pickle.load(f)

    adj = _build_adj_from_network(base_network)
    bins = list(default_config.delay_bins)

    # Normalize swaps input
    norm_swaps: List[SwapOption] = []
    for s in swaps:
        if isinstance(s, SwapOption):
            norm_swaps.append(s)
        else:
            norm_swaps.append(SwapOption(**s, role=role))

    # Union scope over all swaps
    union_scope: Set[str] = set()
    for sw in norm_swaps:
        union_scope |= _compute_scope(sw, adj, hop_limit)

    base_infer = VariableElimination(base_model)
    base_exp_cache: Dict[str, float] = {}
    for node in union_scope:
        t_node = f"{node}_t"
        base_exp_cache[t_node] = _expected_delay(base_infer, t_node, bins)

    results: List[Dict[str, object]] = []
    for sw in norm_swaps:
        try:
            new_net, new_model = reroute_network_and_model(base_network, base_model, sw, sw.role)
            # Recompute scope (routes changed, but hop limit on original graph is acceptable for speed)
            scope = _compute_scope(sw, adj, hop_limit)
            infer = VariableElimination(new_model)
            benefit = 0.0
            for n in scope:
                t_node = f"{n}_t"
                base_exp = base_exp_cache.get(t_node, 0.0)
                new_exp = _expected_delay(infer, t_node, bins)
                benefit += max(0.0, base_exp - new_exp)
            results.append({"key": swap_key(sw), "benefit": benefit})
        except Exception as e:
            results.append({"key": swap_key(sw), "benefit": float("-inf"), "error": str(e)})
    return results


def _make_variant_id(base: str, role: str, idx: int, out_dir: Path) -> str:
    stem = f"{base}_{role[:3]}{idx+1:02d}"
    candidate = stem
    counter = 1
    while (out_dir / f"{candidate}_network_opt.json").exists():
        counter += 1
        candidate = f"{stem}_{counter}"
    return candidate


def materialize_swap_variant(
    model_name: str,
    swap: SwapOption,
    idx: int,
    base_network: Dict,
    base_cpts: Dict,
    out_dir: Path,
    base_model: Optional[object] = None,
    base_exp_cache: Optional[Dict[str, float]] = None,
    adj: Optional[Dict[str, List[str]]] = None,
    bins: Optional[List[int]] = None,
    hop_limit: int = HOP_LIMIT_DEFAULT,
) -> Dict[str, object]:
    """
    Apply a single swap, emit network/cpts/model artifacts to out_dir, and return metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_id = _make_variant_id(model_name, swap.role, idx, out_dir)

    swapped_network = apply_swap_to_network_data(base_network, swap, swap.role)
    swapped_cpts = apply_swap_to_cpts_data(base_cpts, swap, swap.role)

    net_path = out_dir / f"{variant_id}_network_opt.json"
    cpt_path = out_dir / f"{variant_id}_cpts.json"
    model_path = out_dir / f"{variant_id}_model.pkl"

    net_path.write_text(json.dumps(swapped_network, indent=2))
    cpt_path.write_text(json.dumps(swapped_cpts, indent=2))

    # Build and persist the BN so it is ready for propagation checks
    if base_model is not None:
        model = copy.deepcopy(base_model)
        _swap_model_edges_and_cpds(model, swap, swap.role)
    else:
        from dptbn.bn_pgmpy import build_model

        model = build_model(str(cpt_path))

    # Ensure DAG validity; if cycle introduced, bail early
    try:
        if not nx.is_directed_acyclic_graph(model.to_networkx()):
            raise ValueError("Swap introduced cycle")
    except Exception:
        # Persist baseline artifacts but mark benefit as 0 and skip pickling mutated model
        return {
            "key": swap_key(swap),
            "variant_id": variant_id,
            "benefit": float("-inf"),
            "paths": {"network": str(net_path), "cpts": str(cpt_path), "model": ""},
        }

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    benefit = 0.0
    if base_exp_cache is not None and adj is not None and bins is not None:
        try:
            infer = VariableElimination(model)
            scope = _compute_scope(swap, adj, hop_limit)
            for node in scope:
                t_node = f"{node}_t"
                base_exp = base_exp_cache.get(t_node, 0.0)
                new_exp = _expected_delay(infer, t_node, bins)
                benefit += max(0.0, base_exp - new_exp)
        except Exception:
            benefit = 0.0

    return {
        "key": swap_key(swap),
        "variant_id": variant_id,
        "benefit": benefit,
        "paths": {"network": str(net_path), "cpts": str(cpt_path), "model": str(model_path)},
    }


def run_swaps_and_materialize(
    model_name: str,
    swaps: List[SwapOption],
    role: Literal["aircraft", "pilot", "crew"],
    out_dir: Optional[Path] = None,
    hop_limit: int = HOP_LIMIT_DEFAULT,
) -> List[Dict[str, object]]:
    """Run all swaps, emit artifacts, and return metadata for each variant."""
    # Normalize swaps input
    norm_swaps: List[SwapOption] = []
    # Normalize swaps input
    for s in swaps:
        if isinstance(s, SwapOption):
            norm_swaps.append(s)
        else:
            norm_swaps.append(SwapOption(**s))

    net_path = _locate_network_opt(model_name)
    cpt_path = _locate_cpts(model_name)
    model_path = _locate_model_pkl(model_name)
    out_root = out_dir or net_path.parent / "swap_runs"

    base_network = json.loads(net_path.read_text())
    base_cpts = json.loads(cpt_path.read_text())
    base_model = None
    if model_path and model_path.exists():
        try:
            with open(model_path, "rb") as f:
                base_model = pickle.load(f)
        except Exception:
            base_model = None

    adj = _build_adj_from_network(base_network)
    bins = list(default_config.delay_bins)

    # Precompute baseline expectations for union of scopes
    union_scope: Set[str] = set()
    for sw in norm_swaps:
        union_scope |= _compute_scope(sw, adj, hop_limit)

    base_exp_cache: Dict[str, float] = {}
    if base_model is not None:
        try:
            base_infer = VariableElimination(base_model)
            for node in union_scope:
                t_node = f"{node}_t"
                base_exp_cache[t_node] = _expected_delay(base_infer, t_node, bins)
        except Exception:
            base_exp_cache = {}

    results = []
    for idx, swap in enumerate(norm_swaps):
        swap.role = role  # enforce role
        if base_model is not None and not _feasible_swap(base_model, swap, role):
            results.append(
                {
                    "key": swap_key(swap),
                    "variant_id": "",
                    "benefit": float("-inf"),
                    "paths": {},
                    "error": "Feasibility failed",
                }
            )
            continue
        result = materialize_swap_variant(
            model_name,
            swap,
            idx,
            base_network,
            base_cpts,
            out_root,
            base_model=base_model,
            base_exp_cache=base_exp_cache,
            adj=adj,
            bins=bins,
            hop_limit=hop_limit,
        )
        results.append(result)
    return results



def score_swaps_fast(
    model_name: str,
    swaps: List[SwapOption],
    role: Literal["aircraft", "pilot", "crew"],
    hop_limit: int = HOP_LIMIT_DEFAULT,
) -> List[Dict[str, object]]:
    """
    Score swaps quickly:
      - no artifact writing
      - no deepcopy per swap
      - apply swap in-place -> infer -> compute -> undo
    Returns list of dicts with:
      key, benefit (=net_delta), pos_benefit, improved, worsened, worst_regression
    """
    net_path = _locate_network_opt(model_name)
    model_path = _locate_model_pkl(model_name)
    if not model_path or not model_path.exists():
        raise FileNotFoundError("Base model.pkl not found; need it for fast scoring")

    base_network = json.loads(net_path.read_text())
    with open(model_path, "rb") as f:
        base_model = pickle.load(f)

    # feasibility precheck + scope precompute
    adj = _build_adj_from_network(base_network)
    bins = list(default_config.delay_bins)

    norm = []
    for s in swaps:
        if not isinstance(s, SwapOption):
            s = SwapOption(**s)
        s.role = role
        if _feasible_swap(base_model, s, role):
            norm.append(s)
        else:
            norm.append(s)  # keep, but will return -inf

    # union scope on flight ids (your hop-limited adjacency)
    union_scope: Set[str] = set()
    per_swap_scope: Dict[str, Set[str]] = {}
    for sw in norm:
        sc = _compute_scope(sw, adj, hop_limit)
        per_swap_scope[swap_key(sw)] = sc
        union_scope |= sc

    # baseline cache once
    base_infer = VariableElimination(base_model)
    base_exp_cache: Dict[str, float] = {}
    for f_id in union_scope:
        t_node = f"{f_id}_t"
        if t_node in base_model.nodes():
            base_exp_cache[t_node] = _expected_delay(base_infer, t_node, bins)

    # build a reusable nx graph view for cycle checks
    g = _as_nx_digraph(base_model)

    results: List[Dict[str, object]] = []
    for sw in norm:
        key = swap_key(sw)

        if not _feasible_swap(base_model, sw, role):
            results.append({"key": key, "benefit": float("-inf"), "pos_benefit": float("-inf"), "error": "Feasibility failed"})
            continue

        # quick cycle check BEFORE mutating (just for the new edges)
        res_key = {"aircraft": "k", "pilot": "q", "crew": "c"}[role]
        res1 = f"{sw.succ1}_{res_key}"
        res2 = f"{sw.succ2}_{res_key}"
        new1 = (f"{sw.pred2}_t", res1)
        new2 = (f"{sw.pred1}_t", res2)

        # simulate cycle check by temporarily adding edges to g (cheap)
        if _would_create_cycle(g, new1[0], new1[1]) or _would_create_cycle(g, new2[0], new2[1]):
            results.append({"key": key, "benefit": float("-inf"), "pos_benefit": float("-inf"), "error": "cycle detected"})
            continue

        # apply swap in-place, infer, score, undo
        undo = _apply_swap_in_place_single_parent(base_model, sw, role)

        try:
            base_model.check_model()
            infer = VariableElimination(base_model)

            scope = per_swap_scope[key]
            net_delta = 0.0
            pos_benefit = 0.0
            improved = 0
            worsened = 0
            worst_regression = 0.0

            for f_id in scope:
                t_node = f"{f_id}_t"
                if t_node not in base_model.nodes():
                    continue
                e0 = base_exp_cache.get(t_node, 0.0)
                e1 = _expected_delay(infer, t_node, bins)
                d = e0 - e1

                net_delta += d
                if d > 0:
                    pos_benefit += d
                    improved += 1
                elif d < 0:
                    worsened += 1
                    worst_regression = min(worst_regression, d)

            results.append({
                "key": key,
                "benefit": net_delta,          # <-- UI uses this
                "net_delta": net_delta,
                "pos_benefit": pos_benefit,
                "improved": improved,
                "worsened": worsened,
                "worst_regression": worst_regression,
            })

        finally:
            undo()

    return results



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="List swap options for a model/network.")
    parser.add_argument("model", help="Model name, e.g., v3 or v3_act1")
    parser.add_argument("--role", choices=["aircraft", "pilot", "crew"], default="aircraft")
    parser.add_argument("--min_ct", type=int, default=MIN_CT_DEFAULT)
    parser.add_argument("--horizon", type=int, default=HORIZON_DEFAULT)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of swaps")
    args = parser.parse_args()

    try:
        swap_list = list_swaps(args.model, role=args.role, min_ct=args.min_ct, horizon=args.horizon, limit=args.limit)
        print(f"Found {len(swap_list)} swaps for role={args.role}")
        for s in swap_list:
            print(f"{s.chain1}: {s.pred1}->{s.succ1}   ||   {s.chain2}: {s.pred2}->{s.succ2}   (benefit={s.benefit})")
    except Exception as e:
        print(f"Error: {e}")
