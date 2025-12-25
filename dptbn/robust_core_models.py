"""
JOINT robust coupled arc-based model for aircraft, pilot, and crew.

Docplex MILP:
- Each resource r ∈ {A,P,C} forms a path-cover over all flights (cover each flight once).
- Objective: minimize total starts (resources used) while maximizing:
    * 2-chain coupling: shared arcs (i->j) used by all A,P,C
    * 3-chain coupling: shared triples (i->j->k) used by all A,P,C

Meant to be run in cplexenv (docplex installed).
"""
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from docplex.mp.model import Model
except ImportError:  # pragma: no cover - only available in cplexenv
    Model = None

from dptbn.network import SyntheticNetwork
from dptbn.config import NetworkDesign

Arc = Tuple[str, str]
Triple = Tuple[str, str, str]


def _ensure_docplex():
    if Model is None:
        raise ImportError("docplex is required; please run in the cplexenv.")


def _build_feasible_arcs(
    flights,
    min_ct: int,
    max_ct: int,
) -> Dict[str, List[str]]:
    """Adjacency list of feasible arcs i -> j given CT window and station continuity."""
    arcs: Dict[str, List[str]] = {f.flight_id: [] for f in flights}
    flights_sorted = sorted(flights, key=lambda f: f.scheduled_arrival)
    for i, fin in enumerate(flights_sorted):
        for fout in flights_sorted[i + 1 :]:
            if fin.arrival_airport != fout.departure_airport:
                continue
            gap = fout.scheduled_departure - fin.scheduled_arrival
            if min_ct <= gap <= max_ct:
                arcs[fin.flight_id].append(fout.flight_id)
    return arcs


def _all_arcs_from_adj(flight_ids: List[str], adj: Dict[str, List[str]]) -> Tuple[List[Arc], List[Arc], List[Arc], List[Arc]]:
    """Return (start_arcs, mid_arcs, end_arcs, all_arcs) for a resource."""
    start_arcs: List[Arc] = [("SRC", fid) for fid in flight_ids]
    end_arcs: List[Arc] = [(fid, "SNK") for fid in flight_ids]
    mid_arcs: List[Arc] = [(i, j) for i, outs in adj.items() for j in outs]
    all_arcs: List[Arc] = start_arcs + mid_arcs + end_arcs
    return start_arcs, mid_arcs, end_arcs, all_arcs


def _extract_routes(x, flight_ids: List[str]) -> List[List[str]]:
    """Extract routes from solved x over arcs including SRC/SNK."""
    succ: Dict[str, str] = {}
    for (i, j), var in x.items():
        if i in ("SRC", "SNK") or j in ("SRC", "SNK"):
            continue
        if var.solution_value > 0.5:
            succ[i] = j

    starts = [j for (i, j), var in x.items() if i == "SRC" and var.solution_value > 0.5]
    routes: List[List[str]] = []
    for s in starts:
        route = [s]
        while route[-1] in succ:
            route.append(succ[route[-1]])
        routes.append(route)

    covered = {f for r in routes for f in r}
    if len(covered) != len(flight_ids):
        missing = set(flight_ids) - covered
        extra = covered - set(flight_ids)
        raise RuntimeError(f"Route extraction mismatch; missing={missing}, extra={extra}")
    return routes


def solve_joint_robust_coupled_with_log(
    net: SyntheticNetwork,
    design: Optional[NetworkDesign] = None,
    *,
    start_weight: float = 1.0,
    w2: float = 0.0,
    w3: float = 0.0,
    max_triples: Optional[int] = None,
    log_output: bool = True,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """
    Solve a JOINT model for aircraft, pilot, crew simultaneously.

    Returns:
        (routes_aircraft, routes_pilot, routes_crew)
    """
    _ensure_docplex()
    d = design or net.design
    if d is None:
        raise ValueError("NetworkDesign required for robust coupled model.")

    # Strict enable flags (treat tiny weights as disabled)
    enable_w2 = abs(w2) > 1e-12
    enable_w3 = abs(w3) > 1e-12

    flights = net.flights
    flight_ids = [f.flight_id for f in flights]

    # Build feasible arcs per resource
    adj_A = _build_feasible_arcs(flights, min_ct=d.tat_aircraft, max_ct=d.max_ct_aircraft)
    adj_P = _build_feasible_arcs(flights, min_ct=d.mct_pilot, max_ct=d.max_ct_pilot)
    adj_C = _build_feasible_arcs(flights, min_ct=d.mct_crew, max_ct=d.max_ct_crew)

    start_A, mid_A, end_A, all_A = _all_arcs_from_adj(flight_ids, adj_A)
    start_P, mid_P, end_P, all_P = _all_arcs_from_adj(flight_ids, adj_P)
    start_C, mid_C, end_C, all_C = _all_arcs_from_adj(flight_ids, adj_C)

    mid_set_A = set(mid_A)
    mid_set_P = set(mid_P)
    mid_set_C = set(mid_C)
    mid_intersection: List[Arc] = sorted(list(mid_set_A & mid_set_P & mid_set_C))

    # Only enumerate triples if w3 is enabled
    triples: List[Triple] = []
    if enable_w3 and mid_intersection:
        # successor adjacency over intersection arcs only
        succ_int: Dict[str, List[str]] = {fid: [] for fid in flight_ids}
        for (i, j) in mid_intersection:
            succ_int[i].append(j)

        for i in flight_ids:
            outs_ij = succ_int.get(i, [])
            if not outs_ij:
                continue
            for j in outs_ij:
                outs_jk = succ_int.get(j, [])
                if not outs_jk:
                    continue
                for k in outs_jk:
                    triples.append((i, j, k))

        if max_triples is not None and max_triples > 0 and len(triples) > max_triples:
            triples = triples[:max_triples]

    mdl = Model(name="robust_joint_coupled")

    # Decision vars for each resource
    xA = mdl.binary_var_dict(all_A, name="xA")
    xP = mdl.binary_var_dict(all_P, name="xP")
    xC = mdl.binary_var_dict(all_C, name="xC")

    # Position vars for max-legs constraints (DAG order)
    posA = mdl.integer_var_dict(flight_ids, lb=1, ub=d.max_aircraft_stints, name="posA")
    posP = mdl.integer_var_dict(flight_ids, lb=1, ub=d.max_pilot_stints, name="posP")
    posC = mdl.integer_var_dict(flight_ids, lb=1, ub=d.max_crew_stints, name="posC")

    def add_path_cover_constraints(x, mid_arcs: Sequence[Arc], tag: str):
        # inbound exactly once
        for fid in flight_ids:
            incoming = [("SRC", fid)] + [(i, fid) for (i, j) in mid_arcs if j == fid]
            mdl.add_constraint(mdl.sum(x[a] for a in incoming) == 1, ctname=f"{tag}_in_{fid}")

        # flow: inbound == outbound
        for fid in flight_ids:
            outgoing = [(fid, "SNK")] + [(fid, j) for (i, j) in mid_arcs if i == fid]
            incoming = [("SRC", fid)] + [(i, fid) for (i, j) in mid_arcs if j == fid]
            mdl.add_constraint(
                mdl.sum(x[a] for a in outgoing) == mdl.sum(x[a] for a in incoming),
                ctname=f"{tag}_flow_{fid}",
            )

    add_path_cover_constraints(xA, mid_A, "A")
    add_path_cover_constraints(xP, mid_P, "P")
    add_path_cover_constraints(xC, mid_C, "C")

    def add_max_legs_constraints(x, pos, mid_arcs: Sequence[Arc], max_legs: int, tag: str):
        for (i, j) in mid_arcs:
            mdl.add_constraint(
                pos[j] >= pos[i] + 1 - max_legs * (1 - x[(i, j)]),
                ctname=f"{tag}_order_{i}_{j}",
            )
        for fid in flight_ids:
            mdl.add_constraint(pos[fid] <= max_legs, ctname=f"{tag}_cap_{fid}")
            mdl.add_constraint(
                pos[fid] >= 1 - max_legs * (1 - x[("SRC", fid)]),
                ctname=f"{tag}_startpos_{fid}",
            )

    add_max_legs_constraints(xA, posA, mid_A, d.max_aircraft_stints, "A")
    add_max_legs_constraints(xP, posP, mid_P, d.max_pilot_stints, "P")
    add_max_legs_constraints(xC, posC, mid_C, d.max_crew_stints, "C")

    # ---- 2-chain coupling vars/constraints (only if enabled) ----
    z2: Dict[Arc, any] = {}
    if enable_w2 and mid_intersection:
        z2 = mdl.binary_var_dict(mid_intersection, name="z2")
        for (i, j) in mid_intersection:
            mdl.add_constraint(z2[(i, j)] <= xA[(i, j)], ctname=f"z2_le_A_{i}_{j}")
            mdl.add_constraint(z2[(i, j)] <= xP[(i, j)], ctname=f"z2_le_P_{i}_{j}")
            mdl.add_constraint(z2[(i, j)] <= xC[(i, j)], ctname=f"z2_le_C_{i}_{j}")
            mdl.add_constraint(
                z2[(i, j)] >= xA[(i, j)] + xP[(i, j)] + xC[(i, j)] - 2,
                ctname=f"z2_ge_sum_{i}_{j}",
            )

    # ---- 3-chain coupling vars/constraints (only if enabled) ----
    z3: Dict[Triple, any] = {}
    if enable_w3 and triples:
        z3 = mdl.binary_var_dict(triples, name="z3")
        for (i, j, k) in triples:
            mdl.add_constraint(z3[(i, j, k)] <= xA[(i, j)], ctname=f"z3_le_A1_{i}_{j}_{k}")
            mdl.add_constraint(z3[(i, j, k)] <= xA[(j, k)], ctname=f"z3_le_A2_{i}_{j}_{k}")
            mdl.add_constraint(z3[(i, j, k)] <= xP[(i, j)], ctname=f"z3_le_P1_{i}_{j}_{k}")
            mdl.add_constraint(z3[(i, j, k)] <= xP[(j, k)], ctname=f"z3_le_P2_{i}_{j}_{k}")
            mdl.add_constraint(z3[(i, j, k)] <= xC[(i, j)], ctname=f"z3_le_C1_{i}_{j}_{k}")
            mdl.add_constraint(z3[(i, j, k)] <= xC[(j, k)], ctname=f"z3_le_C2_{i}_{j}_{k}")

            mdl.add_constraint(
                z3[(i, j, k)]
                >= xA[(i, j)] + xA[(j, k)] + xP[(i, j)] + xP[(j, k)] + xC[(i, j)] + xC[(j, k)] - 5,
                ctname=f"z3_ge_sum_{i}_{j}_{k}",
            )

    # Objective: start_weight * total_starts  - w2*sum(z2) - w3*sum(z3)
    total_starts = mdl.sum(xA[a] for a in start_A) + mdl.sum(xP[a] for a in start_P) + mdl.sum(xC[a] for a in start_C)
    obj = start_weight * total_starts
    if enable_w2 and z2:
        obj = obj - w2 * mdl.sum(z2[a] for a in z2)
    if enable_w3 and z3:
        obj = obj - w3 * mdl.sum(z3[t] for t in z3)

    mdl.minimize(obj)

    mdl.solve(log_output=log_output)
    if mdl.solution is None:
        raise RuntimeError("No feasible robust coupled solution found.")

    routes_A = _extract_routes(xA, flight_ids)
    routes_P = _extract_routes(xP, flight_ids)
    routes_C = _extract_routes(xC, flight_ids)
    return routes_A, routes_P, routes_C
