
import time
from fastapi import HTTPException
from typing import Dict, Any, List
from dptbn.schemas import GlobalResponse, MetaInfo, FlightState, Visuals, EdgeStyle, NodeStyle, SummaryPanel, FlightMetrics
from dptbn.math_engine import calculate_expected_delay, calculate_prob_delay
from dptbn.graph_algo import _find_reachable_subgraph, _batch_query

def run_propagation(req, model_wrapper, infer_engine) -> GlobalResponse:
    """
    Tool 1: Generalized Inference (Scenario B - Shock Propagation).
    Features: "Do" Operator, Reachability Pruning, Delta-based Coloring.
    """
    print(f"\n📨 Inference Request: {req.dict()}")
    start_time = time.time()
    original_model = model_wrapper.model
    
    # 1. Parse Roots (Shock Targets)
    # 1. Parse Roots (Shock Targets)
    evidence_roots = []
    formatted_evidence = {}
    
    for k, v in req.evidence.items():
        if k == "ALL": continue
        
        # Populate formatted_evidence (Standardization)
        # Assuming input is "F001" or "F001_c".
        # pgmpy needs exact node names. 
        # If input is "F001", model likely expects "F001_t".
        # Only suffix flight IDs if they don't have suffix.
        key_std = k if "_" in k else f"{k}_t"
        formatted_evidence[key_std] = v
        
        # Identification Logic for Roots
        if k.endswith("_t"):
            evidence_roots.append(k)
        elif "_" in k:
            if "_p_" in k:
                parts = k.split("_p_")
                if len(parts) > 0: evidence_roots.append(f"{parts[0]}_t")
            else:
                fid = k.split("_")[0]
                evidence_roots.append(f"{fid}_t")
                
    evidence_roots = list(set(evidence_roots))

    if not evidence_roots and req.target:
        # If targets specified but not in evidence
        # Filter "ALL"
        clean_targets = [t for t in req.target if t != "ALL"]
        if clean_targets:
            evidence_roots.extend([t if "_t" in t else f"{t}_t" for t in clean_targets])
            
    if not evidence_roots:
         # Check if explicit "ALL" requested without evidence
         if "ALL" in req.target:
             # Global Scan requested. Return all nodes from model?
             # For Propagator, "Global Scan" usually means "Base State" (no shock propagation).
             # But if we want to show visual graph, we need nodes.
             # Let's fetch all T-nodes.
             try:
                evidence_roots = [n for n in original_model.nodes() if n.endswith("_t")]
             except:
                evidence_roots = []
         else:
             return GlobalResponse(
                meta_info=MetaInfo(tool_exec="tool_1_prop", calc_time_ms=0, model_name=req.model),
                network_state={},
                visuals=Visuals(nodes={}, edges=[])
            )

    # 2. Reachability BFS (Forward)
    reachable_nodes, reachable_edges = _find_reachable_subgraph(original_model, evidence_roots, hop_limit=req.hop_limit)
    
    # 3. Define Query (Flights + Intermediary Resources)
    query_vars = list(reachable_nodes)
    resource_vars = [e["resource"] for e in reachable_edges]
    full_query = list(set(query_vars + resource_vars))
    
    # 4. Inference
    if not infer_engine:
        raise HTTPException(status_code=500, detail="Inference engine not ready")
        
    # A. Baseline Run (Reference)
    # Use infer_engine as base
    infer_base = infer_engine
    res_base = _batch_query(infer_base, full_query, evidence={}, chunk_size=50)
    
    # B. Shock Run (Intervention or Observation)
    res_shock = {}
    
    if req.shock_mode == "do" and formatted_evidence:
        try:
             # DO-Operator
            active_model = original_model.do(list(formatted_evidence.keys()))
            from pgmpy.inference import VariableElimination
            infer_shock = VariableElimination(active_model)
            res_shock = _batch_query(infer_shock, full_query, evidence=formatted_evidence, chunk_size=20)
        except Exception as e:
            print(f"Do-op failed: {e}. Falling back to Observation.")
            res_shock = _batch_query(infer_base, full_query, evidence=formatted_evidence, chunk_size=20)
            
    else:
        # Observation Mode
        res_shock = _batch_query(infer_base, full_query, evidence=formatted_evidence, chunk_size=20)

    # 5. Delta Processing
    net_state = {}
    visual_nodes = {}
    
    BINS = [0, 15, 30, 60]

    for node in query_vars:
        # Resolve Distribution: Result OR Evidence
        p_base = None
        # Base usually has no evidence, so result is authoritative 
        p_base = res_base.get(node)
        
        p_shock = None
        if node in formatted_evidence: p_shock = {formatted_evidence[node]: 1.0}
        else: p_shock = res_shock.get(node)
        
        if p_base is None or p_shock is None: continue
        
        # State Construction
        best_state = max(p_shock, key=p_shock.get)
        best_prob = p_shock[best_state]
        prob_delay = sum(v for k,v in p_shock.items() if str(k) != '0')
        
        exp_base = calculate_expected_delay(p_base, bins=BINS)
        exp_shock = calculate_expected_delay(p_shock, bins=BINS)
        delta_E = exp_shock - exp_base
        
        net_state[node] = FlightState(
            prob_delay=prob_delay,
            expected_delay=exp_shock,
            most_likely_bin=int(best_state) if str(best_state).isdigit() else 0,
            most_likely_bin_prob=best_prob,
            is_affected=True
        )
        
        # Visuals
        fid = node.replace("_t", "")
        
        color = "#cccccc"
        if delta_E > 15: color = "#8b0000"
        elif delta_E > 5: color = "#ff0000"
        elif delta_E > 1: color = "#ff9900"
        elif delta_E > 0.1: color = "#ffffcc"
        
        border = None
        if node in evidence_roots:
             # Just border, no blue color override
             border = "None" # User disliked blue border? Or just blue node?
             # User said: "The blue colored node is appearing.."
             # Diagnostics tool used border="4px solid blue" previously.
             # I will keep it neutral or use minimal indicator.
             # Let's use standard heatmap color.
             pass
            
        visual_nodes[fid] = NodeStyle(
            color=color,
            tooltip=f"Base: {exp_base:.1f}m -> Shock: {exp_shock:.1f}m\nDelta: +{delta_E:.1f}m"
        )

    # 6. Edges (Rich Visualization)
    visual_edges = []
    for edge in reachable_edges:
        # edge: {parent, child, resource}
        res_node = edge["resource"]
        
        delta_p_bad = 0.0
        prob_issue_shock = 0.0
        
        # Resolve Resource Distribution
        # Check evidence first
        p_s = None
        if res_node in formatted_evidence: p_s = {formatted_evidence[res_node]: 1.0}
        else: p_s = res_shock.get(res_node)
        
        p_b = sorted_base = res_base.get(res_node)

        if p_s and p_b:
            prob_issue_shock = sum(v for k,v in p_s.items() if str(k) != '0')
            prob_issue_base = sum(v for k,v in p_b.items() if str(k) != '0')
            delta_p_bad = prob_issue_shock - prob_issue_base
        
        # Check Child Impact
        child_id = edge["child"] # e.g. F002
        child_t = f"{child_id}_t"
        
        child_delta_E = 0.0
        if child_t in net_state:
             # We use computed expectations from loop above
             child_delta_E = net_state[child_t].expected_delay - calculate_expected_delay(res_base.get(child_t, {0:1.0}), BINS)

        # Significance Rule
        is_sig = (child_delta_E >= 1.0) or (delta_p_bad >= 0.05) or (prob_issue_shock > 0.3)
        
        color = "#e0e0e0"
        width = 1
        opacity = 0.1
        tooltip = "Insignificant"

        # Evidence Highlight Rule (Force)
        if res_node in formatted_evidence:
             is_sig = True
             width = 6
             opacity = 1.0
             delta_p_bad = 1.0 # Force max intensity
             tooltip = f"ROOT CAUSE: {res_node}={formatted_evidence[res_node]}"
        
        if is_sig:
             # Color by type if we can infer it
             etype = "other"
             if res_node.endswith("_k"): etype = "ac"
             elif res_node.endswith("_q"): etype = "pilot"
             elif res_node.endswith("_c"): etype = "cabin"
             elif "_p_" in res_node: etype = "pax"
             
             if etype == "ac": color = "#ef4444"
             elif etype == "pilot": color = "#3b82f6"
             elif etype == "cabin": color = "#eab308"
             elif etype == "pax": color = "#8b5cf6"
             else: color = "#6b7280"
             
             if width == 1: # If not forced
                 width = int(2 + (10 * min(max(delta_p_bad, 0), 0.3) / 0.3))
                 opacity = 1.0
                #  tooltip = f"(+{round(delta_p_bad*100)}%)\nChild Impact: +{child_delta_E:.1f}m"
                 tooltip = f"Link Risk: {round(prob_issue_shock*100)}% (+{round(delta_p_bad*100)}%)\nChild Impact: +{child_delta_E:.1f}m"

        # Infer type for Propagator (since reachable_edges doesn't explicitly have it sometimes?)
        # _find_reachable_subgraph actually returns edge dict with 'type' usually if adapted.
        # But let's infer it safely if missing.
        etype = edge.get("type", "unknown")
        if etype == "unknown":
             if res_node.endswith("_k"): etype = "ac"
             elif res_node.endswith("_q"): etype = "pilot"
             elif res_node.endswith("_c"): etype = "cabin"
             elif "_p_" in res_node: etype = "pax"
             
        visual_edges.append(EdgeStyle(
            **{"from": edge["parent"], "to": edge["child"]},
            color=color,
            thickness=width,
            tooltip=tooltip,
            type=etype
        ))

    calc_time = (time.time() - start_time) * 1000
    
    return GlobalResponse(
        meta_info=MetaInfo(tool_exec="tool_1_prop", calc_time_ms=calc_time, model_name=req.model),
        network_state=net_state,
        visuals=Visuals(nodes=visual_nodes, edges=visual_edges)
    )
