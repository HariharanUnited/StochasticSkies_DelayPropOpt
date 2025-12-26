
import time
from fastapi import HTTPException
from typing import Dict, Any, List
from dptbn.schemas import GlobalResponse, MetaInfo, FlightState, Visuals, EdgeStyle, NodeStyle, SummaryPanel, FlightMetrics
from dptbn.math_engine import calculate_expected_delay, calculate_prob_delay, calculate_dmx, get_bin_midpoints
from dptbn.config import default_config
from dptbn.graph_algo import _batch_query

def _build_global_response_legacy(
    tool_name: str, 
    model_name: str, 
    start_time: float, 
    raw_results: Dict[str, Dict[int, float]], 
    evidence: Dict[str, Any],
    targets: List[str],
    model: Any = None
) -> GlobalResponse:
    """
    Helper for legacy/extra tools that still rely on this mapping logic.
    """
    calc_time = (time.time() - start_time) * 1000
    net_state = {}
    bins = default_config.delay_bins
    
    for node_name, dist_map in raw_results.items():
        if not node_name.endswith("_t"): continue
        fid = node_name[:-2]
        
        prob_delay = calculate_prob_delay(dist_map)
        exp_delay = calculate_expected_delay(dist_map, bins)
        
        best_bin = 0
        max_p = -1.0
        for b, p in dist_map.items():
            if p > max_p:
                max_p = p
                best_bin = int(b)
                
        metrics = FlightMetrics()
        net_state[fid] = FlightState(
            prob_delay=prob_delay,
            expected_delay=exp_delay,
            most_likely_bin=best_bin,
            most_likely_bin_prob=max_p,
            is_affected=prob_delay > 0.1,
            metrics=metrics
        )
        
    visuals = Visuals(
        heatmap_mode="prob_delay",
        edges=[], 
        nodes={},
        summary_panel=SummaryPanel(title=f"Result: {tool_name}", stats=[])
    )
    
    return GlobalResponse(
        meta_info=MetaInfo(tool_exec=tool_name, calc_time_ms=calc_time, model_name=model_name),
        network_state=net_state,
        visuals=visuals
    )

def run_multipliers(req, model, infer):
    t0 = time.time()
    # 1. Baseline
    target_nodes = [n for n in model.nodes() if n.endswith("_t")]
    q_base = _batch_query(infer, target_nodes, evidence={}, chunk_size=50)

    # 2. Shock
    pgmpy_evidence = {k if "_" in k else f"{k}_t": v for k, v in req.evidence.items()}
    q_shock = _batch_query(infer, target_nodes, evidence=pgmpy_evidence, chunk_size=20)
    
    response = _build_global_response_legacy("T5_DelayMultipliers", req.model, t0, q_shock, req.evidence, req.target, model)
    
    bins = default_config.delay_bins
    
    # Root calc
    root_delay_duration = 0.0
    mids = get_bin_midpoints(bins)
    for k, v in req.evidence.items():
        if hasattr(mids, "__getitem__") and  0 <= v < len(mids):
            val = mids[v]
            if val > root_delay_duration: root_delay_duration = val
            
    for fid, fstate in response.network_state.items():
        node_t = f"{fid}_t"
        if node_t in q_base:
            base_exp = calculate_expected_delay(q_base[node_t], bins)
            dmx = calculate_dmx(fstate.expected_delay, base_exp, root_delay_duration)
            fstate.metrics.DMe = fstate.expected_delay
            fstate.metrics.DMx = dmx
            
    response.visuals.heatmap_mode = "metrics.DMx"
    return response

def run_weak_links(req, model, infer):
    t0 = time.time()
    target_nodes = [n for n in model.nodes() if n.endswith("_t")]
    q_base = _batch_query(infer, target_nodes, evidence={}, chunk_size=1)
    
    bins = default_config.delay_bins
    mids = get_bin_midpoints(bins)
    
    base_total = sum(calculate_expected_delay(dist, bins) for dist in q_base.values())
        
    all_flights = sorted([n[:-2] for n in target_nodes])
    candidates = all_flights[:20] 
    
    net_state = {}
    
    for fid in candidates:
        ev = {f"{fid}_t": 2} # Shock (Bin 2)
        root_val = mids[2] 
        q_shock = _batch_query(infer, target_nodes, evidence=ev, chunk_size=1)
        shock_total = sum(calculate_expected_delay(dist, bins) for dist in q_shock.values())
            
        final_dmx = calculate_dmx(shock_total, base_total, root_val)
        
        net_state[fid] = FlightState(
            prob_delay=0.0, expected_delay=0.0, most_likely_bin=0, is_affected=True,
            metrics=FlightMetrics(DMx=final_dmx)
        )

    visuals = Visuals(
        heatmap_mode="metrics.DMx",
        summary_panel=SummaryPanel(
            title="Weak-Link Analysis",
            stats=[{"label": "Flights Scanned", "value": len(candidates)}]
        )
    )

    return GlobalResponse(
        meta_info=MetaInfo(tool_exec="T6_WeakLink", calc_time_ms=(time.time()-t0)*1000, model_name=req.model),
        network_state=net_state,
        visuals=visuals
    )

def run_factors(req, model, infer):
    t0 = time.time()
    if not req.target: raise HTTPException(400, "Tool 7 requires target.")
    fid = req.target[0]
    if fid.endswith("_t"): fid = fid[:-2]
    target_node = f"{fid}_t"
    
    try: parents = model.get_parents(target_node)
    except: parents = []
    
    query_vars = [target_node]
    label_map = {}
    for p in parents:
        query_vars.append(p)
        if p.endswith("_k"): label_map[p] = "Aircraft (k)"
        elif p.endswith("_q"): label_map[p] = "Pilot (q)"
        elif p.endswith("_c"): label_map[p] = "Crew (c)"
        elif "_p_" in p: 
            parts = p.split("_p_")
            label_map[p] = f"Pax ({parts[1]})" if len(parts)>1 else "Pax"
        else: label_map[p] = p
        
    q_res = _batch_query(infer, query_vars, evidence=req.evidence, chunk_size=1)
    
    breakdown = {}
    for node, dist in q_res.items():
        if node == target_node: continue
        prob_bad = calculate_prob_delay(dist, threshold_bin_idx=1) 
        readable_label = label_map.get(node, node)
        breakdown[readable_label] = round(prob_bad, 3)
    
    net_state = {}
    if target_node in q_res:
        dist = q_res[target_node]
        net_state[fid] = FlightState(
            prob_delay=calculate_prob_delay(dist),
            expected_delay=calculate_expected_delay(dist, default_config.delay_bins),
            most_likely_bin=0, is_affected=True,
            cause_breakdown=breakdown
        )
        
    return GlobalResponse(
        meta_info=MetaInfo(tool_exec="T7_RiskFactors", calc_time_ms=(time.time()-t0)*1000, model_name=req.model),
        network_state=net_state,
        visuals=Visuals(heatmap_mode="prob_delay", nodes={})
    )

def run_intervention(req, model, infer):
    t0 = time.time()
    do_dict = {}
    for k, v in req.evidence.items():
        node = k if "_" in k else f"{k}_t"
        do_dict[node] = v
        
    try:
        mutilated_model = model.do(list(do_dict.keys()))
    except Exception as e:
         raise HTTPException(500, f"Do-operator failed: {e}")
         
    from pgmpy.inference import VariableElimination
    infer_new = VariableElimination(mutilated_model)
    query_targets = req.target if req.target else [n for n in mutilated_model.nodes() if n.endswith("_t")]
    results = _batch_query(infer_new, query_targets, evidence=do_dict, chunk_size=20)
    
    return _build_global_response_legacy("T12_Intervention", req.model, t0, results, req.evidence, req.target, mutilated_model)
