"""
Arc-based feasibility models for aircraft, pilot, and crew assignment.

Uses docplex to minimize the number of resources (starts) while covering each
flight exactly once, with connection-time and max-legs constraints.

These functions are meant to be run in the cplexenv (docplex available) and
consume the network JSON generated in Phase 2.
"""
from typing import Dict, List, Sequence, Tuple, Optional

try:
    from docplex.mp.model import Model
except ImportError:  # pragma: no cover - only available in cplexenv
    Model = None

from dptbn.network import SyntheticNetwork
from dptbn.config import NetworkDesign


Arc = Tuple[str, str]


def _ensure_docplex():
    if Model is None:
        raise ImportError("docplex is required; please run in the cplexenv.")


def _build_feasible_arcs(
    flights,
    min_ct: int,
    max_ct: int,
) -> Dict[str, List[str]]:
    """Return adjacency list of feasible arcs i -> j given CT window."""
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


def _build_model(
    net: SyntheticNetwork,
    design: NetworkDesign,
    min_ct: int,
    max_ct: int,
    max_legs: int,
    name: str,
    overlap_penalty: Optional[Dict[Tuple[str, str], float]] = None,
):
    _ensure_docplex()
    mdl = Model(name=name)
    flights = net.flights
    flight_ids = [f.flight_id for f in flights]
    arcs_adj = _build_feasible_arcs(flights, min_ct=min_ct, max_ct=max_ct)

    start_arcs: List[Arc] = [("SRC", fid) for fid in flight_ids]
    end_arcs: List[Arc] = [(fid, "SNK") for fid in flight_ids]
    mid_arcs: List[Arc] = [(i, j) for i, outs in arcs_adj.items() for j in outs]
    all_arcs: Sequence[Arc] = start_arcs + mid_arcs + end_arcs

    x = mdl.binary_var_dict(all_arcs, name="x")
    pos = mdl.integer_var_dict(flight_ids, lb=1, ub=max_legs, name="pos")

    # Cover each flight exactly once (inbound).
    for fid in flight_ids:
        incoming = [("SRC", fid)] + [(i, fid) for i, j in mid_arcs if j == fid]
        mdl.add_constraint(mdl.sum(x[a] for a in incoming) == 1, ctname=f"in_{fid}")

    # Flow conservation: inbound == outbound.
    for fid in flight_ids:
        outgoing = [(fid, "SNK")] + [(fid, j) for i, j in mid_arcs if i == fid]
        incoming = [("SRC", fid)] + [(i, fid) for i, j in mid_arcs if j == fid]
        mdl.add_constraint(mdl.sum(x[a] for a in outgoing) == mdl.sum(x[a] for a in incoming), ctname=f"flow_{fid}")

    # Path length (max legs) on DAG using position variables.
    for i, j in mid_arcs:
        mdl.add_constraint(pos[j] >= pos[i] + 1 - max_legs * (1 - x[(i, j)]), ctname=f"order_{i}_{j}")
    for fid in flight_ids:
        mdl.add_constraint(pos[fid] <= max_legs, ctname=f"cap_{fid}")
        mdl.add_constraint(pos[fid] >= 1 - max_legs * (1 - x[('SRC', fid)]), ctname=f"startpos_{fid}")

    # Objective: minimize number of starts (resources).
    mdl.minimize(mdl.sum(x[a] for a in start_arcs))
    if overlap_penalty:
        mdl.minimize(mdl.objective_expr + mdl.sum(overlap_penalty.get((i, j), 0.0) * x[(i, j)] for (i, j) in mid_arcs))

    return mdl, x, mid_arcs


def _extract_routes(x, mid_arcs: Sequence[Arc], flight_ids: List[str]) -> List[List[str]]:
    # Build successor map from solution.
    succ: Dict[str, str] = {}
    for (i, j), var in x.items():
        if i in ("SRC", "SNK") or j in ("SRC", "SNK"):
            continue
        if var.solution_value > 0.5:
            succ[i] = j

    # Identify starts: arcs SRC->fid selected.
    starts = [j for (i, j), var in x.items() if i == "SRC" and var.solution_value > 0.5]
    routes: List[List[str]] = []
    for s in starts:
        route = [s]
        while route[-1] in succ:
            route.append(succ[route[-1]])
        routes.append(route)
    # Sanity: ensure every flight is covered exactly once
    covered = {f for route in routes for f in route}
    if len(covered) != len(flight_ids):
        missing = set(flight_ids) - covered
        extra = covered - set(flight_ids)
        raise RuntimeError(f"Route extraction mismatch; missing={missing}, extra={extra}")
    return routes


def solve_aircraft_feasibility(net: SyntheticNetwork, design: Optional[NetworkDesign] = None) -> List[List[str]]:
    return solve_aircraft_feasibility_with_log(net, design=design, log_output=False)


def solve_aircraft_feasibility_with_log(
    net: SyntheticNetwork, design: Optional[NetworkDesign] = None, log_output: bool = False
) -> List[List[str]]:
    d = design or net.design
    if d is None:
        raise ValueError("NetworkDesign required for aircraft feasibility.")
    mdl, x, mid_arcs = _build_model(
        net=net,
        design=d,
        min_ct=d.tat_aircraft,
        max_ct=d.max_ct_aircraft,
        max_legs=d.max_aircraft_stints,
        name="aircraft_feasibility",
        overlap_penalty=None,
    )
    mdl.solve(log_output=log_output)
    if mdl.solution is None:
        raise RuntimeError("No feasible aircraft routing found.")
    return _extract_routes(x, mid_arcs, [f.flight_id for f in net.flights])


def solve_pilot_feasibility(net: SyntheticNetwork, design: Optional[NetworkDesign] = None) -> List[List[str]]:
    return solve_pilot_feasibility_with_log(net, design=design, log_output=False)


def solve_pilot_feasibility_with_log(
    net: SyntheticNetwork,
    design: Optional[NetworkDesign] = None,
    log_output: bool = False,
    overlap_weights: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[List[str]]:
    d = design or net.design
    if d is None:
        raise ValueError("NetworkDesign required for pilot feasibility.")
    mdl, x, mid_arcs = _build_model(
        net=net,
        design=d,
        min_ct=d.mct_pilot,
        max_ct=d.max_ct_pilot,
        max_legs=d.max_pilot_stints,
        name="pilot_feasibility",
        overlap_penalty=overlap_weights,
    )
    mdl.solve(log_output=log_output)
    if mdl.solution is None:
        raise RuntimeError("No feasible pilot pairing found.")
    return _extract_routes(x, mid_arcs, [f.flight_id for f in net.flights])


def solve_crew_feasibility(net: SyntheticNetwork, design: Optional[NetworkDesign] = None) -> List[List[str]]:
    return solve_crew_feasibility_with_log(net, design=design, log_output=False)


def solve_crew_feasibility_with_log(
    net: SyntheticNetwork,
    design: Optional[NetworkDesign] = None,
    log_output: bool = False,
    overlap_weights: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[List[str]]:
    d = design or net.design
    if d is None:
        raise ValueError("NetworkDesign required for crew feasibility.")
    mdl, x, mid_arcs = _build_model(
        net=net,
        design=d,
        min_ct=d.mct_crew,
        max_ct=d.max_ct_crew,
        max_legs=d.max_crew_stints,
        name="crew_feasibility",
        overlap_penalty=overlap_weights,
    )
    mdl.solve(log_output=log_output)
    if mdl.solution is None:
        raise RuntimeError("No feasible crew pairing found.")
    return _extract_routes(x, mid_arcs, [f.flight_id for f in net.flights])


def validate_routes(
    flights,
    routes: List[List[str]],
    min_ct: int,
    max_ct: int,
    label: str,
) -> None:
    """Raise if coverage or CT constraints are violated."""
    flight_ids = [f.flight_id for f in flights]
    flat = [fid for r in routes for fid in r]
    if len(flat) != len(flight_ids):
        raise ValueError(f"{label}: coverage mismatch; expected {len(flight_ids)} legs, got {len(flat)}")
    if len(set(flat)) != len(flight_ids):
        raise ValueError(f"{label}: duplicate legs detected")

    by_id = {f.flight_id: f for f in flights}
    for route in routes:
        for i in range(1, len(route)):
            prev = by_id[route[i - 1]]
            cur = by_id[route[i]]
            if prev.arrival_airport != cur.departure_airport:
                raise ValueError(f"{label}: airport mismatch {prev.flight_id}->{cur.flight_id}")
            gap = cur.scheduled_departure - prev.scheduled_arrival
            if gap < min_ct or gap > max_ct:
                raise ValueError(f"{label}: CT window violated {prev.flight_id}->{cur.flight_id} gap={gap}")
