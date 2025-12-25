import time
from typing import Dict, List, Any, Tuple, Set

from fastapi import HTTPException

from dptbn.schemas import (
    GlobalResponse, MetaInfo, DMStats, Scorecard, TopKItem,
    FlightState, Visuals, NodeStyle
)
from dptbn.math_engine import calculate_expected_delay, get_bin_midpoints
from dptbn.config import default_config, default_network_design

# IMPORTANT: Your _batch_query should return {node: {bin_idx: prob, ...}}
# (same as Tool1’s helper in server.py) :contentReference[oaicite:2]{index=2}
from dptbn.graph_algo import _batch_query

ROOT_DELAY_MIN = 60.0   # Dr in minutes (paper imposes 60-min root delay)
HOP_LIMIT = 10         # Increased for paper-faithful propagation
DEFAULT_BUFFER = 15.0   # naive slack absorption for DMc

def _to_dist_map(x: Any) -> Dict[int, float]:
    """
    Accepts either:
      - dict-like {0: p0, 1: p1, ...}  (your _batch_query output)
      - pgmpy DiscreteFactor-like with .values
    Returns {int_bin: float_prob}
    """
    if x is None:
        return {}
    if isinstance(x, dict):
        return {int(k): float(v) for k, v in x.items()}
    # pgmpy factor case
    if hasattr(x, "values"):
        vals = list(x.values)  # factor.values is typically ndarray-like
        return {i: float(v) for i, v in enumerate(vals)}
    return {}

def _exp_delay_nonneg(dist: Dict[int, float], bins: List[int]) -> float:
    """
    For multipliers, delays should not go negative.
    Your midpoint for bin0 is -5 :contentReference[oaicite:3]{index=3}, so clamp to 0.
    """
    e = calculate_expected_delay(dist, bins)
    return max(0.0, float(e))

def _descendants_with_hops(model, root_t: str, hop_limit: int) -> Set[str]:
    """
    Follow children to find downstream flight nodes up to hop_limit.
    Mirrors your Scenario B forward-reachability idea (but per-root).
    """
    descendants: Set[str] = set()
    queue: List[Tuple[str, int]] = [(root_t, 0)]
    visited: Set[str] = {root_t}

    while queue:
        curr, depth = queue.pop(0)
        if depth >= hop_limit:
            continue

        for child in model.get_children(curr):
            if child.endswith("_t"):
                if child not in visited:
                    visited.add(child)
                    descendants.add(child)
                    queue.append((child, depth + 1))
            else:
                # resource bridge
                for grand in model.get_children(child):
                    if grand.endswith("_t") and grand not in visited:
                        visited.add(grand)
                        descendants.add(grand)
                        queue.append((grand, depth + 1))

    return descendants

def run_dm_calculations(req, model_wrapper, infer_engine) -> GlobalResponse:
    """
    Tool 3: Delay Multipliers (DMc, DMe, DMx) + Scorecard.
    Fixes:
      - parse _batch_query outputs correctly (dict not factor)
      - clamp expected delays to >= 0 for multiplier math
      - supports do-intervention per root (recommended)
    """
    print(f"\n📨 DM Analysis Request: {req.dict()}")
    t0 = time.time()

    model = model_wrapper.model
    bins = default_config.delay_bins
    mids = get_bin_midpoints(bins)

    # -------------------------
    # 0) Identify flight nodes
    # -------------------------
    all_flights = sorted([n for n in model.nodes() if n.endswith("_t")])
    if not all_flights:
        raise HTTPException(400, "No flight nodes (*_t) found in model.")

    # -------------------------
    # 0b) Tail + Hub metadata
    # -------------------------
    flight_to_tail: Dict[str, str] = {}
    flight_to_origin: Dict[str, str] = {}
    tails: Dict[str, List[str]] = {}

    # Try mapping origin from network JSON (Reliable)
    import json
    from pathlib import Path
    try:
        net_path = Path("data") / f"{req.model}_network.json"
        if net_path.exists():
             with open(net_path, "r") as f:
                 net_data = json.load(f)
                 for flt in net_data.get("flights", []):
                     fid = flt.get("flight_id")
                     orig = flt.get("departure_airport")
                     if fid and orig:
                         flight_to_origin[f"{fid}_t"] = orig
                         # Also map raw ID just in case
                         flight_to_origin[fid] = orig
    except Exception as e:
        print(f"Failed to load network layout: {e}")

    # Fallback to _g parent if not found
    for f in all_flights:
        if f not in flight_to_origin:
            for p in model.get_parents(f):
                if p.endswith("_g"):
                    flight_to_origin[f] = p[:-2]

    # tail chains via _k nodes: Fprev_t -> Fchild_k -> Fchild_t
    tail_adj: Dict[str, List[str]] = {f: [] for f in all_flights}
    has_incoming_k: Dict[str, bool] = {f: False for f in all_flights}

    k_nodes = [n for n in model.nodes() if n.endswith("_k")]
    for kn in k_nodes:
        # child flight
        f_child = next((c for c in model.get_children(kn) if c.endswith("_t")), None)
        # upstream flight
        f_prev = next((p for p in model.get_parents(kn) if p.endswith("_t")), None)
        if f_child and f_prev:
            tail_adj[f_prev].append(f_child)
            has_incoming_k[f_child] = True

    tail_counter = 1
    for f in all_flights:
        if not has_incoming_k[f]:
            tail_id = f"N{tail_counter:03d}"
            chain: List[str] = []
            curr = f
            seen = set()
            while curr and curr not in seen:
                seen.add(curr)
                chain.append(curr)
                flight_to_tail[curr] = tail_id
                nxts = tail_adj.get(curr, [])
                curr = nxts[0] if nxts else None
            tails[tail_id] = chain
            tail_counter += 1

    print(f"Identified {len(tails)} aircraft tails.")

    # -------------------------
    # 1) Baseline expected delay E[D_i] for all flights
    # -------------------------
    baseline_delays: Dict[str, float] = {}
    try:
        base_res = _batch_query(infer_engine, all_flights, evidence={}, chunk_size=1)
        for f in all_flights:
            dist = _to_dist_map(base_res.get(f))
            baseline_delays[f] = _exp_delay_nonneg(dist, bins)
    except Exception as e:
        print(f"Baseline inference failed: {e}")
        baseline_delays = {f: 0.0 for f in all_flights}

    # -------------------------
    # 2) Main loop per root flight
    # -------------------------
    dm_results: Dict[str, Dict[str, float]] = {}

    # Choose which bin corresponds to the imposed 60 minutes:
    # safest: pick the bin whose midpoint is closest to 60
    shock_bin = int(min(range(len(mids)), key=lambda i: abs(mids[i] - ROOT_DELAY_MIN)))

    # If you want the "60+" bin always, uncomment this instead:
    # shock_bin = min(4, max(0, model.get_cardinality(root_t) - 1))

    shock_mode = getattr(req, "shock_mode", "do")  # "do" or "observe"

    from pgmpy.inference import VariableElimination

    total = len(all_flights)
    for idx, root_t in enumerate(all_flights):
        fid = root_t[:-2]
        if idx % 10 == 0:
            print(f"  Processing {idx}/{total}: {fid}")

        # A) affected set I(R)
        affected = _descendants_with_hops(model, root_t, hop_limit=HOP_LIMIT)
        if not affected:
            dm_results[fid] = {"dmc": 1.0, "dme": 1.0, "dmx": 1.0}
            continue

        affected_list = sorted(list(affected))

        # B) Posterior expected delays under imposed root delay
        sum_post = 0.0
        sum_induced_pos = 0.0 # Only positive induced delay (propagation)
        
        # We need E_prior sum for DME but we iterate individually for DMx
        
        # Build shock engine
        try:
            if shock_mode == "do":
                # mutilate just the root node (cut its parents), then set its value by evidence
                shock_model = model.do([root_t])
                infer_shock = VariableElimination(shock_model)
            else:
                infer_shock = infer_engine

            shock_ev = {root_t: min(shock_bin, model.get_cardinality(root_t) - 1)}

            shock_res = _batch_query(infer_shock, affected_list, evidence=shock_ev, chunk_size=1)

            for a in affected_list:
                # Per-node calculation
                prior = baseline_delays.get(a, 0.0)
                
                dist_a = _to_dist_map(shock_res.get(a))
                post = _exp_delay_nonneg(dist_a, bins)
                
                sum_post += post
                
                # DMx (Positive-Only): max(0, post - prior)
                # Matches paper definition of "sensitive propagators"
                delta = post - prior
                if delta > 0:
                     sum_induced_pos += delta

        except Exception as e:
            # Hard fallback: treat as no propagation
            # (better than silently turning everything to 1 because baseline=0)
            print(f"  Root {fid}: shock inference failed: {e}")
            sum_induced_pos = 0.0
            sum_post = sum(baseline_delays.get(a, 0.0) for a in affected_list)

        # DMe = (Dr + Σ E[D'_i]) / Dr
        dme = (ROOT_DELAY_MIN + sum_post) / ROOT_DELAY_MIN

        # DMx = (Dr + Σ max(0, E[D'_i] - E[D_i])) / Dr
        dmx = (ROOT_DELAY_MIN + sum_induced_pos) / ROOT_DELAY_MIN

        # C) DMc (classic naive deterministic)
        # deterministic forward pass, but with your simplified “buffer-per-hop” rule:
        dmc_sum = 0.0
        q = [(root_t, ROOT_DELAY_MIN, 0)]
        seen = {root_t}

        while q:
            curr, d_in, depth = q.pop(0)
            if depth >= HOP_LIMIT:
                continue
            d_out = max(0.0, d_in - DEFAULT_BUFFER)
            if d_out <= 0:
                continue

            for child in model.get_children(curr):
                if child.endswith("_t"):
                    if child in affected and child not in seen:
                        seen.add(child)
                        dmc_sum += d_out
                        q.append((child, d_out, depth + 1))
                else:
                    for grand in model.get_children(child):
                        if grand.endswith("_t") and grand in affected and grand not in seen:
                            seen.add(grand)
                            dmc_sum += d_out
                            q.append((grand, d_out, depth + 1))

        dmc = (ROOT_DELAY_MIN + dmc_sum) / ROOT_DELAY_MIN

        dm_results[fid] = {
            "dmc": round(float(dmc), 3),
            "dme": round(float(dme), 3),
            "dmx": round(float(dmx), 3),
        }

    # -------------------------
    # 3) Top-K + scorecard
    # -------------------------
    def top_k(metric: str, k: int = 10) -> List[TopKItem]:
        items = sorted(dm_results.items(), key=lambda kv: kv[1][metric], reverse=True)
        return [TopKItem(id=fid, val=vals[metric]) for fid, vals in items[:k]]

    top_k_dmx = top_k("dmx", 10)
    top_k_dme = top_k("dme", 10)
    top_k_dmc = top_k("dmc", 10)

    # Tail fragility: max(DMx of first 2 flights)
    tail_scores: List[Tuple[str, float]] = []
    for tail_id, chain in tails.items():
        early = chain[:2]
        vals = []
        for f in early:
            f_id = f[:-2]
            if f_id in dm_results:
                vals.append(dm_results[f_id]["dmx"])
        if vals:
            tail_scores.append((tail_id, max(vals)))
    tail_scores.sort(key=lambda x: x[1], reverse=True)
    top_fragile = [f"{tid} ({score:.2f})" for tid, score in tail_scores[:3]]

    # Hub risk: mean(top 10% DMx of departures at hub)
    hub_risks: List[Tuple[str, float]] = []
    for hub in getattr(default_network_design, "hub_codes", []):
        hub_flights = [f for f, orig in flight_to_origin.items() if orig == hub]
        vals = []
        for f in hub_flights:
            f_id = f[:-2]
            if f_id in dm_results:
                vals.append(dm_results[f_id]["dmx"])
        if not vals:
            continue
        vals.sort(reverse=True)
        take = max(1, int(0.1 * len(vals)))
        hub_risks.append((hub, sum(vals[:take]) / take))
    hub_risks.sort(key=lambda x: x[1], reverse=True)
    top_hubs = [f"{hub} ({risk:.2f})" for hub, risk in hub_risks[:3]]

    # Robustness score (0..100), higher = better
    all_dmx = sorted([v["dmx"] for v in dm_results.values()])
    n = len(all_dmx)
    p95 = all_dmx[min(n - 1, int(0.95 * n))] if n else 1.0
    r_net = max(0.0, min(1.0, (p95 - 1.0) / 0.5))

    avg_early = (sum(s for _, s in tail_scores) / len(tail_scores)) if tail_scores else 1.0
    r_early = max(0.0, min(1.0, (avg_early - 1.0) / 0.5))

    max_hub = hub_risks[0][1] if hub_risks else 1.0
    r_hub = max(0.0, min(1.0, (max_hub - 1.0) / 0.5))

    risk_total = 0.50 * r_net + 0.35 * r_early + 0.15 * r_hub
    robustness_score = round(100 * (1.0 - risk_total))

    scorecard = Scorecard(
        robustness_score=robustness_score,
        top_k_dmx=top_k_dmx,
        top_k_dme=top_k_dme,
        top_k_dmc=top_k_dmc,
        risky_hubs=top_hubs,
        fragile_tails=top_fragile
    )

    # -------------------------
    # 4) Response payload
    # -------------------------
    net_state: Dict[str, FlightState] = {}
    visual_nodes: Dict[str, NodeStyle] = {}

    for fid, m in dm_results.items():
        net_state[f"{fid}_t"] = FlightState(
            prob_delay=0.0,
            expected_delay=0.0,
            most_likely_bin=0,
            is_affected=True,
            dm_metrics=DMStats(**m),
        )
        visual_nodes[fid] = NodeStyle(
            color="#ffffff",
            tooltip=f"DMc: {m['dmc']}\nDMe: {m['dme']}\nDMx: {m['dmx']}"
        )

    calc_ms = (time.time() - t0) * 1000.0
    return GlobalResponse(
        meta_info=MetaInfo(tool_exec="tool_3_dm", calc_time_ms=calc_ms, model_name=req.model),
        network_state=net_state,
        visuals=Visuals(nodes=visual_nodes, edges=[]),
        scorecard=scorecard
    )
