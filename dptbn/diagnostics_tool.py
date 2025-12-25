
import time
from fastapi import HTTPException
from typing import Dict, Any, List
from dptbn.schemas import GlobalResponse, MetaInfo, FlightState, Visuals, EdgeStyle, NodeStyle, SummaryPanel, FlightMetrics
from dptbn.math_engine import calculate_expected_delay, calculate_prob_delay
from dptbn.graph_algo import _find_ancestor_subgraph, _batch_query

def run_diagnostics(req, model_wrapper, infer_engine) -> GlobalResponse:
    """
    Tool 2: Reverse Diagnostics (Scenario C).
    Evidence: Target Flight = Severe Delay (Implicit or Explicit).
    """
    print(f"\n📨 Diagnostic Request: {req.dict()}")
    start_time = time.time()
    original_model = model_wrapper.model
    
    # 1. Parse Roots (Targets for Diagnosis)
    evidence_roots = []
    formatted_evidence = {}
    
    for node, state in req.evidence.items():
        if node == "ALL": continue
        
        # Format Key: "F001" -> "F001_t" if needed
        # But if node is "F001_c", utilize it.
        key = node if "_" in node else f"{node}_t"
        formatted_evidence[key] = state
        
        if key.endswith("_t"):
            evidence_roots.append(key)
        else:
             # Resource Mapping: Map resource to its Flight for BFS start
             if "_p_" in key:
                # FChild_p_FParent -> FChild_t
                parts = key.split("_p_")
                if len(parts) > 0: evidence_roots.append(f"{parts[0]}_t")
             else:
                # F050_k -> F050_t
                fid = key.split("_")[0]
                evidence_roots.append(f"{fid}_t")
    
    evidence_roots = list(set(evidence_roots))
            
    if not evidence_roots:
        return GlobalResponse(
            meta_info=MetaInfo(tool_exec="tool_2_diagnostics", calc_time_ms=0, model_name=req.model),
            network_state={},
            visuals=Visuals(nodes={}, edges=[])
        )

    # 2. Reverse BFS
    ancestor_nodes, ancestor_edges = _find_ancestor_subgraph(original_model, evidence_roots, hop_limit=req.hop_limit)
    
    # 3. Define Query Scope (Ancestors + Resources)
    query_flights = list(ancestor_nodes)
    query_vars = set(query_flights)
    
    for f_node in query_flights:
        pars = original_model.get_parents(f_node)
        relevant_pars = [p for p in pars if p.endswith(("_k", "_q", "_c", "_g")) or "_p_" in p]
        query_vars.update(relevant_pars)
        
    final_query_list = list(query_vars)
    
    # 4. Dual Inference (Baseline vs Diagnostic)
    if not infer_engine:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")

    ev_diag = formatted_evidence.copy()
    # BASELINE: Should be the Nominal parameters (No Evidence / Default CPTs).
    # Previous bug: Filtered against mapped roots, so F032_k leaked into Base.
    # Result: Base == Shock -> Delta == 0 -> Grey Visuals.
    # Fix: Base is empty (or context if separated, but tool treats all input as the scenario).
    ev_base = {}
    
    # Execute Queries
    res_base = _batch_query(infer_engine, final_query_list, evidence=ev_base, chunk_size=50)
    res_diag = _batch_query(infer_engine, final_query_list, evidence=ev_diag, chunk_size=50) 
    
    # 5. Process Metrics
    net_state = {}
    visual_nodes = {}
    results = {} 
    
    BINS = [0, 15, 30, 60]
    
    for node in final_query_list:
        # Resolve Distribution: Result OR Evidence
        p_base = None
        if node in ev_base: p_base = {ev_base[node]: 1.0}
        else: p_base = res_base.get(node)
        
        p_diag = None
        if node in ev_diag: p_diag = {ev_diag[node]: 1.0}
        else: p_diag = res_diag.get(node)
        
        if p_base is None or p_diag is None: continue
        
        results[node] = {"base": p_base, "diag": p_diag} # Store for edges
        
        # Display Diagnostic State
        best_state = max(p_diag, key=p_diag.get)
        best_prob = p_diag[best_state]
        prob_delay = sum(v for k,v in p_diag.items() if str(k) != '0') 
        
        exp_d = 0.0
        if node.endswith("_t"):
             exp_d = calculate_expected_delay(p_diag, bins=BINS)
        
        net_state[node] = FlightState(
            prob_delay=prob_delay,
            expected_delay=exp_d,
            most_likely_bin=int(best_state) if str(best_state).isdigit() else 0,
            most_likely_bin_prob=best_prob,
            is_affected=True
        )

        # Node Visualization
        if node.endswith("_t"):
            exp_base = calculate_expected_delay(p_base, bins=BINS)
            exp_diag = exp_d 
            delta_E = exp_diag - exp_base
            
            fid = node.replace("_t", "")
            
            color = "#cccccc" 
            if delta_E > 15: color = "#8b0000" 
            elif delta_E > 8: color = "#ff0000" 
            elif delta_E > 3: color = "#ff9900" 
            elif delta_E > 1: color = "#ffff00" 
            elif delta_E > 0: color = "#ccffcc" 
            
            # Removed Blue Border override as per feedback
            border = None
             
            visual_nodes[fid] = NodeStyle(color=color, border=border, tooltip=f"Base: {exp_base:.1f}m -> Diag: {exp_diag:.1f}m\nDelta: +{delta_E:.1f}m")

    # 6. Edge Visualization
    visual_edges = []
    
    for edge in ancestor_edges:
        res_node = edge["resource"]
        
        # Default Logic
        color = "#e0e0e0" 
        width = 1 
        opacity = 0.1
        tooltip = "Insignificant"
        
        if res_node in results:
            dist_base = results[res_node]["base"]
            dist_diag = results[res_node]["diag"]
            
            p_bad_base = sum([v for k,v in dist_base.items() if str(k) != '0'])
            p_bad_diag = sum([v for k,v in dist_diag.items() if str(k) != '0'])
            delta_p_bad = p_bad_diag - p_bad_base
            
            child_node = f"{edge['to']}_t"
            child_delta_E = 0
            if child_node in results:
                e_b = calculate_expected_delay(results[child_node]["base"], bins=BINS)
                e_d = calculate_expected_delay(results[child_node]["diag"], bins=BINS)
                child_delta_E = e_d - e_b
            
            is_sig = (child_delta_E >= 1.0) or (delta_p_bad >= 0.01)
            
            if is_sig:
                t = edge["type"]
                if t == "ac": color = "#ef4444"
                elif t == "pilot": color = "#3b82f6"
                elif t == "cabin": color = "#eab308"
                elif t == "pax": color = "#8b5cf6"
                else: color = "#6b7280"
                
                width = 1 #int(2 + (8 * min(max(delta_p_bad, 0), 0.5) / 0.5))
                opacity = 1.0
                tooltip = f"Link Risk (Delta P): +{delta_p_bad:.1%}\nChild Impact: +{child_delta_E:.1f}m"

        visual_edges.append(EdgeStyle(
            **{"from": edge["from"], "to": edge["to"]},
            color=color,
            thickness=width,
            tooltip=tooltip,
            type=edge["type"]
        ))

    calc_time = (time.time() - start_time) * 1000
    
    visuals = Visuals(
        heatmap_mode="delta_delay",
        edges=visual_edges,
        nodes=visual_nodes,
        summary_panel=SummaryPanel(
            title="Diagnostic Inference",
            stats=[
                {"label": "Target", "value": evidence_roots[0] if evidence_roots else "None"},
                {"label": "Ancestors Visited", "value": len(ancestor_nodes)}
            ]
        )
    )
    
    return GlobalResponse(
        meta_info=MetaInfo(tool_exec="diagnostic_v1", calc_time_ms=calc_time, model_name=req.model),
        network_state=net_state,
        visuals=visuals
    )
