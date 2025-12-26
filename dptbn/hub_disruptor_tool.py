import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from dptbn.config import default_config
from dptbn.graph_algo import _find_reachable_subgraph, _batch_query
from dptbn.math_engine import calculate_expected_delay, calculate_prob_delay
from dptbn.schemas import (
    EdgeStyle,
    FlightState,
    MetaInfo,
    NodeStyle,
    SummaryPanel,
    GlobalResponse,
    Visuals,
)


def _locate_network_file(model_name: str) -> Optional[Path]:
    """
    Look for the network JSON (needed for hub/timebank filtering).
    Only searches in bn-viz/public/data.
    """
    candidate = Path("bn-viz") / "public" / "data" / f"{model_name}_network_opt.json"
    return candidate if candidate.exists() else None


def _select_flights_from_timebank(
    model_name: str, hub: Optional[str], start_min: Optional[int], end_min: Optional[int]
) -> List[str]:
    """
    Load the network JSON and collect flights that depart from `hub`
    within [start_min, end_min).
    Returns a list of flight_t node names (e.g., F001_t).
    """
    net_path = _locate_network_file(model_name)
    if not net_path:
        raise HTTPException(404, f"Network file for {model_name} not found to resolve timebank flights.")

    try:
        data = json.loads(net_path.read_text())
    except Exception as exc:
        raise HTTPException(500, f"Failed to read network file {net_path}: {exc}")

    flights = data.get("flights", [])
    selected: List[str] = []
    for f in flights:
        fid = f.get("flight_id")
        dep_airport = f.get("departure_airport")
        dep_min = f.get("scheduled_departure")
        if not fid or dep_airport is None or dep_min is None:
            continue
        if hub and dep_airport != hub:
            continue
        if start_min is not None and end_min is not None:
            if not (start_min <= dep_min < end_min):
                continue
        selected.append(f"{fid}_t")

    return selected


def _format_evidence(node_states: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure keys carry suffixes expected by pgmpy (append _t for flights).
    """
    formatted: Dict[str, Any] = {}
    for k, v in node_states.items():
        key = k if "_" in k else f"{k}_t"
        formatted[key] = v
    return formatted


def _delta_to_color(delta_minutes: float) -> str:
    """
    Diverging palette: greens for improvement, reds for harm.
    """
    if delta_minutes <= -8:
        return "#0f7a39"  # strong improvement
    if delta_minutes <= -3:
        return "#1fa455"
    if delta_minutes <= -1:
        return "#7fd09a"
    if delta_minutes < 1:
        return "#f5f5f5"
    if delta_minutes < 3:
        return "#f6c343"
    if delta_minutes < 8:
        return "#f28b2f"
    return "#c03532"


def _expected_cost(dist: Dict[Any, float], cost_map: Dict[int, float]) -> float:
    return sum(float(prob) * cost_map.get(int(bin_idx), 0.0) for bin_idx, prob in dist.items())


def _tint_color(hex_color: str, factor: float) -> str:
    """
    Darken/lighten a hex color by factor in [0,1]. 1 = original, <1 lighter.
    """
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    try:
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
    except Exception:
        return hex_color
    mix = lambda c: int((255 - c) * (1 - factor) + c * factor)
    return "#{:02x}{:02x}{:02x}".format(mix(r), mix(g), mix(b))


def run_hub_disruptor(req, model_wrapper, infer_engine) -> GlobalResponse:
    """
    Hub Disruptor Tool:
      - Select departures at a hub within a timebank.
      - Force them to Bin 4 via do-intervention.
      - Propagate and report cost / impact deltas.
    """
    t0 = time.time()
    model = model_wrapper.model
    bins = list(default_config.delay_bins)

    hub = getattr(req, "hub", None) or getattr(req, "hub_code", None)
    time_start = getattr(req, "timebank_start", None)
    time_end = getattr(req, "timebank_end", None)
    time_label = getattr(req, "timebank_label", None)

    # 1) Identify root flights from timebank + hub
    root_flights: List[str] = []
    try:
        root_flights = _select_flights_from_timebank(req.model, hub, time_start, time_end)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Failed to select flights for hub disruptor: {exc}")

    if not root_flights:
        raise HTTPException(400, "No flights found for the selected hub/timebank.")

    # Force Bin 4 (or max state available) for selected flights
    evidence: Dict[str, int] = {}
    for node in root_flights:
        try:
            max_state = model.get_cardinality(node) - 1
            evidence[node] = min(4, max_state)
        except Exception:
            evidence[node] = 4

    # Merge any user-supplied evidence (root overrides if conflict)
    merged_evidence: Dict[str, Any] = {}
    merged_evidence.update(req.evidence or {})
    merged_evidence.update(evidence)
    formatted_evidence = _format_evidence(merged_evidence)

    # 2) Reachability from all roots (forward)
    reachable_nodes, reachable_edges = _find_reachable_subgraph(model, root_flights, hop_limit=getattr(req, "hop_limit", 5))

    # 3) Query scope: reachable flights + their resources
    resource_nodes = [e["resource"] for e in reachable_edges]
    query_nodes = list(set(list(reachable_nodes) + resource_nodes))
    # Avoid querying variables that are clamped in evidence (pgmpy forbids target ∩ evidence)
    query_nodes_no_evidence = [q for q in query_nodes if q not in formatted_evidence]

    if not infer_engine:
        raise HTTPException(500, "Inference engine not ready")

    # 4) Baseline
    base_results = _batch_query(infer_engine, query_nodes, evidence={}, chunk_size=50)

    # 5) Intervention (do)
    from pgmpy.inference import VariableElimination

    try:
        active_model = model.do(list(formatted_evidence.keys()))
        infer_do = VariableElimination(active_model)
        shock_results = _batch_query(infer_do, query_nodes_no_evidence, evidence=formatted_evidence, chunk_size=30)
    except Exception as exc:
        print(f"HubDisruptor do() failed, falling back to observe: {exc}")
        shock_results = _batch_query(infer_engine, query_nodes_no_evidence, evidence=formatted_evidence, chunk_size=30)

    # 6) Node states + visuals
    net_state: Dict[str, FlightState] = {}
    visual_nodes: Dict[str, NodeStyle] = {}

    # cost per bin
    cost_map = {0: 0.0, 1: 5000.0, 2: 20000.0, 3: 40000.0, 4: 50000.0}

    baseline_total_cost = 0.0
    shock_total_cost = 0.0

    for node in reachable_nodes:
        p_base = base_results.get(node)
        if p_base is None:
            continue
        if node in formatted_evidence:
            dist_shock = {formatted_evidence[node]: 1.0}
        else:
            dist_shock = shock_results.get(node)
        if dist_shock is None:
            continue

        exp_base = calculate_expected_delay(p_base, bins)
        exp_shock = calculate_expected_delay(dist_shock, bins)
        delta_e = exp_shock - exp_base

        prob_delay = calculate_prob_delay(dist_shock, threshold_bin_idx=1)
        ml_state = max(dist_shock, key=dist_shock.get)
        ml_prob = dist_shock[ml_state]

        # costs
        baseline_total_cost += _expected_cost(p_base, cost_map)
        shock_total_cost += _expected_cost(dist_shock, cost_map)

        net_state[node] = FlightState(
            prob_delay=prob_delay,
            expected_delay=exp_shock,
            most_likely_bin=int(ml_state) if str(ml_state).isdigit() else 0,
            most_likely_bin_prob=ml_prob,
            is_affected=True,
        )

        fid = node.replace("_t", "")
        visual_nodes[fid] = NodeStyle(
            color=_delta_to_color(delta_e),
            border="2px solid #111111" if node in root_flights else None,
            # tooltip=f"Base: {exp_base:.1f}m | Shock: {exp_shock:.1f}m | Δ={delta_e:+.1f}m\nP(delay>=bin1): {prob_delay:.0%}",
            tooltip=f"Base: {exp_base:.1f}m | Shock: {exp_shock:.1f}m | Δ={delta_e:+.1f}m",
        )

    # 7) Edges
    visual_edges: List[EdgeStyle] = []
    for edge in reachable_edges:
        res_node = edge["resource"]
        p_base_res = base_results.get(res_node)
        p_shock_res = shock_results.get(res_node) or {}

        prob_bad_shock = calculate_prob_delay(p_shock_res, threshold_bin_idx=1) if p_shock_res else 0.0
        prob_bad_base = calculate_prob_delay(p_base_res, threshold_bin_idx=1) if p_base_res else 0.0
        delta_prob = prob_bad_shock - prob_bad_base

        child_t = f"{edge['child']}_t"
        delta_child = 0.0
        if child_t in net_state and child_t in base_results:
            delta_child = net_state[child_t].expected_delay - calculate_expected_delay(base_results[child_t], bins)

        significant = (abs(delta_child) >= 3.0) and (prob_bad_shock >= 0.30)

        color = "#d3d3d3"
        width = 1
        tooltip = "Minor effect"
        etype = edge.get("type", "other")
        if etype == "other":
            if res_node.endswith("_k"):
                etype = "ac"
            elif res_node.endswith("_q"):
                etype = "pilot"
            elif res_node.endswith("_c"):
                etype = "cabin"
            elif "_p_" in res_node:
                etype = "pax"

        if significant or res_node in formatted_evidence:
            if etype == "ac":
                color = "#2563eb"
            elif etype == "pilot":
                color = "#3b82f6"
            elif etype == "cabin":
                color = "#ef4444"
            elif etype == "pax":
                color = "#8b5cf6"
            else:
                color = "#6b7280"
            width = 1 # 2 + int(min(6, abs(delta_child) / 2))
            # intensity scales with |delta_child|
            intensity = min(1.0, 0.3 + min(abs(delta_child) / 10.0, 1.0) * 0.7)
            color = _tint_color(color, intensity)
            tooltip = f"Resource risk {prob_bad_shock:.0%} (Δ{delta_prob:+.0%}) | Impact: {delta_child:+.1f}m"

        visual_edges.append(
            EdgeStyle(
                **{"from": edge["parent"], "to": edge["child"]},
                color=color,
                thickness=width,
                tooltip=tooltip,
                type=etype,
            )
        )

    calc_ms = (time.time() - t0) * 1000.0
    total_roots = len(root_flights)
    savings = baseline_total_cost - shock_total_cost
    summary_stats = [
        {"label": "Hub", "value": hub or "N/A"},
        {"label": "Timebank", "value": time_label or f"{time_start}-{time_end}"},
        {"label": "Root flights", "value": total_roots},
        {"label": "Baseline cost (USD)", "value": round(baseline_total_cost, 2)},
        {"label": "Post-shock cost (USD)", "value": round(shock_total_cost, 2)},
        {"label": "Savings (USD)", "value": round(savings, 2)},
    ]

    visuals = Visuals(
        heatmap_mode="delta_delay",
        edges=visual_edges,
        nodes=visual_nodes,
        summary_panel=SummaryPanel(title="Hub Disruptor", stats=summary_stats),
    )

    return GlobalResponse(
        meta_info=MetaInfo(tool_exec="hub_disruptor", calc_time_ms=calc_ms, model_name=req.model),
        network_state=net_state,
        visuals=visuals,
    )
