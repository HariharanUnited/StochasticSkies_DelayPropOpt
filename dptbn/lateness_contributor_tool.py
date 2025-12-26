import time
from functools import reduce
from typing import Any, Dict, List

from fastapi import HTTPException

from dptbn.config import default_config
from dptbn.graph_algo import _batch_query
from dptbn.math_engine import calculate_expected_delay, calculate_prob_delay, get_bin_midpoints
from dptbn.schemas import (
    EdgeStyle,
    FlightState,
    GlobalResponse,
    MetaInfo,
    NodeStyle,
    SummaryPanel,
    Visuals,
)


def _format_evidence(ev: Dict[str, Any]) -> Dict[str, Any]:
    formatted: Dict[str, Any] = {}
    for k, v in ev.items():
        key = k if "_" in k else f"{k}_t"
        formatted[key] = v
    return formatted


def _mix_with_white(color_hex: str, strength: float) -> str:
    """
    Simple linear blend of a hex color with white.
    strength in [0,1], where 1 => original color, 0 => white.
    """
    color_hex = color_hex.lstrip("#")
    if len(color_hex) != 6:
        return f"#{color_hex}"
    try:
        r = int(color_hex[0:2], 16)
        g = int(color_hex[2:4], 16)
        b = int(color_hex[4:6], 16)
    except ValueError:
        return f"#{color_hex}"

    r_new = int(255 - (255 - r) * strength)
    g_new = int(255 - (255 - g) * strength)
    b_new = int(255 - (255 - b) * strength)
    return "#{:02x}{:02x}{:02x}".format(r_new, g_new, b_new)


def _dominant_color(contributor: str, intensity: float) -> str:
    base = {
        "k": "#ef4444",      # aircraft
        "q": "#3b82f6",      # pilot
        "c": "#eab308",      # crew
        "p": "#8b5cf6",      # pax
        "none": "#d1d5db",
    }.get(contributor, "#d1d5db")
    return _mix_with_white(base, intensity)


def _get_parents(model, flight_t: str) -> List[str]:
    try:
        parents = model.get_parents(flight_t)
    except Exception:
        parents = []
    return [p for p in parents if p.endswith(("_k", "_q", "_c")) or "_p_" in p]


def _expected_flight_delay(infer_engine, flight_t: str, evidence: Dict[str, Any], bins) -> float:
    """
    Compute E[D] for flight_t under evidence.
    If the flight itself is in evidence, drop that entry so we can see the modeled effect.
    """
    ev = {k: v for k, v in evidence.items() if k != flight_t}
    res = _batch_query(infer_engine, [flight_t], evidence=ev, chunk_size=1)
    dist = res.get(flight_t, {})
    if not dist:
        return 0.0
    return calculate_expected_delay(dist, bins)


def run_lateness_contributor(req, model_wrapper, infer_engine) -> GlobalResponse:
    """
    Lateness Contributor Tool:
      - For each flight, condition on the most likely delayed state.
      - Aggregate contributor shares (k/q/c/p) excluding turnaround/ground.
    """
    t0 = time.time()
    model = model_wrapper.model
    bins = default_config.delay_bins

    formatted_ev = _format_evidence(req.evidence or {})

    # Determine target flights
    if getattr(req, "target", None):
        target_flights = [t if t.endswith("_t") else f"{t}_t" for t in req.target]
    else:
        target_flights = [n for n in model.nodes() if n.endswith("_t")]

    if not target_flights:
        raise HTTPException(400, "No target flights specified or found for lateness contributor tool.")

    # Baseline distributions for targets under provided evidence
    base_results = _batch_query(infer_engine, target_flights, evidence=formatted_ev, chunk_size=30)

    net_state: Dict[str, FlightState] = {}
    visual_nodes: Dict[str, NodeStyle] = {}
    visual_edges: List[EdgeStyle] = []

    for flight_t in target_flights:
        base_dist = base_results.get(flight_t)
        if base_dist is None:
            continue

        prob_late = calculate_prob_delay(base_dist, threshold_bin_idx=1)
        most_likely = max(base_dist, key=base_dist.get)
        ml_prob = base_dist[most_likely]
        exp_delay = calculate_expected_delay(base_dist, bins)

        # Choose a delayed state to condition on (argmax over bins >= 1)
        delayed_state = None
        for state, prob in sorted(base_dist.items(), key=lambda kv: kv[1], reverse=True):
            if int(state) >= 1:
                delayed_state = state
                break

        parents = _get_parents(model, flight_t)
        cond_ev = dict(formatted_ev)
        if delayed_state is not None:
            cond_ev[flight_t] = int(delayed_state)

        parent_results = _batch_query(infer_engine, parents, evidence=cond_ev, chunk_size=15) if parents else {}

        # Contribution math: probability parent is bad * marginal effect on flight when forcing bad vs good
        scores_raw = {"k": 0.0, "q": 0.0, "c": 0.0, "p": 0.0}
        base_exp = _expected_flight_delay(infer_engine, flight_t, formatted_ev, bins)

        for p in parents:
            prob_bad = calculate_prob_delay(parent_results.get(p, {}), threshold_bin_idx=1)
            try:
                card = model.get_cardinality(p)
            except Exception:
                card = 5
            good_state = 0
            bad_state = max(0, card - 1)
            exp_good = _expected_flight_delay(infer_engine, flight_t, {**formatted_ev, p: good_state}, bins)
            exp_bad = _expected_flight_delay(infer_engine, flight_t, {**formatted_ev, p: bad_state}, bins)
            effect = max(0.0, exp_bad - exp_good)
            score = prob_bad * effect
            if p.endswith("_k"):
                scores_raw["k"] = max(scores_raw["k"], score)
            elif p.endswith("_q"):
                scores_raw["q"] = max(scores_raw["q"], score)
            elif p.endswith("_c"):
                scores_raw["c"] = max(scores_raw["c"], score)
            elif "_p_" in p:
                scores_raw["p"] += score  # sum pax contributions

        total_score = sum(scores_raw.values())
        if total_score <= 0 or prob_late < 0.1:
            shares = {"Aircraft (k)": 0.0, "Pilot (q)": 0.0, "Crew (c)": 0.0, "Pax (p)": 0.0}
            dominant_key = "none"
        else:
            shares = {
                "Aircraft (k)": scores_raw["k"] / total_score,
                "Pilot (q)": scores_raw["q"] / total_score,
                "Crew (c)": scores_raw["c"] / total_score,
                "Pax (p)": scores_raw["p"] / total_score,
            }
            dominant = max(shares.items(), key=lambda kv: kv[1])[0] if shares else "none"
            if dominant.startswith("Aircraft"):
                dominant_key = "k"
            elif dominant.startswith("Pilot"):
                dominant_key = "q"
            elif dominant.startswith("Crew"):
                dominant_key = "c"
            elif dominant.startswith("Pax"):
                dominant_key = "p"
            else:
                dominant_key = "none"

        if dominant_key == "k":
            dominant_label = "Aircraft (k)"
        elif dominant_key == "q":
            dominant_label = "Pilot (q)"
        elif dominant_key == "c":
            dominant_label = "Crew (c)"
        elif dominant_key == "p":
            dominant_label = "Pax (p)"
        else:
            dominant_label = "None"

        dominant_share = shares.get(dominant_label, 0.0)
        intensity = max(0.2, min(1.0, prob_late / 0.8))
        color = _dominant_color(dominant_key, intensity)

        fid = flight_t.replace("_t", "")
        net_state[flight_t] = FlightState(
            prob_delay=prob_late,
            expected_delay=exp_delay,
            most_likely_bin=int(most_likely) if str(most_likely).isdigit() else 0,
            most_likely_bin_prob=ml_prob,
            is_affected=True,
            cause_breakdown={k: round(v, 3) for k, v in shares.items()},
        )

        breakdown_str = ", ".join([f"{k}: {shares.get(k,0):.0%}" for k in ["Aircraft (k)", "Pilot (q)", "Crew (c)", "Pax (p)"]])

        visual_nodes[fid] = NodeStyle(
            color=color,
            tooltip=(
                f"Dominant: {dominant_label} ({dominant_share:.0%})\n"
                f"Breakdown: {breakdown_str}"
            ),
        )

        # Edge highlighting for dominant contributor
        if parents and dominant_key != "none":
            best_parent = None
            best_score = -1.0
            for p in parents:
                if dominant_key == "p" and "_p_" not in p:
                    continue
                if dominant_key == "k" and not p.endswith("_k"):
                    continue
                if dominant_key == "q" and not p.endswith("_q"):
                    continue
                if dominant_key == "c" and not p.endswith("_c"):
                    continue
                sc = calculate_prob_delay(parent_results.get(p, {}), threshold_bin_idx=1)
                if sc > best_score:
                    best_score = sc
                    best_parent = p

            if best_parent:
                # parent flight id
                try:
                    src_flight = next((pr for pr in model.get_parents(best_parent) if pr.endswith("_t")), None)
                except Exception:
                    src_flight = None
                src = src_flight.replace("_t", "") if src_flight else best_parent
                thickness = 2 + int(6 * min(1.0, shares.get(dominant_label, 0.0)))
                visual_edges.append(
                    EdgeStyle(
                        **{"from": src, "to": fid},
                        color=_dominant_color(dominant_key, 1.0),
                        thickness=thickness,
                        tooltip=f"Top contributor edge via {best_parent}: {best_score:.0%}",
                        type=dominant_key,
                    )
                )

    calc_ms = (time.time() - t0) * 1000.0
    summary = SummaryPanel(
        title="Lateness Contributors",
        stats=[
            {"label": "Flights analyzed", "value": len(net_state)},
            {"label": "Evidence provided", "value": len(formatted_ev)},
        ],
    )

    visuals = Visuals(
        heatmap_mode="prob_delay",
        edges=visual_edges,
        nodes=visual_nodes,
        summary_panel=summary,
    )

    return GlobalResponse(
        meta_info=MetaInfo(tool_exec="lateness_contributor", calc_time_ms=calc_ms, model_name=req.model),
        network_state=net_state,
        visuals=visuals,
    )
