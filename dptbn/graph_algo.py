
from typing import List, Dict, Any, Tuple
import time

def _find_reachable_subgraph(model, start_nodes: List[str], hop_limit: int = 5):
    """
    Step B4 & B11: BFS to find reachable flight nodes and build flight-level adjacency.
    Returns:
        reachable_t_nodes: Set of flight nodes to query.
        flight_edges: List of dictionary edges
    """
    queue = [(node, 0) for node in start_nodes]
    visited = set(start_nodes)
    reachable_t_nodes = set(start_nodes)
    flight_edges = []
    
    while queue:
        current_t, depth = queue.pop(0)
        if depth >= hop_limit:
            continue
            
        children = model.get_children(current_t)
        
        for child in children:
            if child.endswith("_t"):
                if child not in visited:
                    visited.add(child)
                    reachable_t_nodes.add(child)
                    queue.append((child, depth + 1))
            else:
                grand_children = model.get_children(child)
                for grand in grand_children:
                    if grand.endswith("_t"):
                        if grand not in visited:
                            visited.add(grand)
                            reachable_t_nodes.add(grand)
                            time.sleep(0) # Yield logic
                            queue.append((grand, depth + 1))
                        
                        flight_edges.append({
                            "parent": current_t.replace("_t", ""),
                            "child": grand.replace("_t", ""),
                            "resource": child
                        })
    
    return reachable_t_nodes, flight_edges

def _find_ancestor_subgraph(model, start_nodes: list, hop_limit: int = 5):
    """
    Reverse BFS to find upstream ancestors (contributories).
    Returns:
       - ancestor_t_nodes (set): Set of Flight_t nodes affecting target
       - flight_edges (list): [{"from": src, "to": dst, "resource": r, "type": t}, ...]
    """
    queue = [(node, 0) for node in start_nodes]   # start_nodes are *_t
    visited = set(start_nodes)
    reachable_t_nodes = set(start_nodes)
    flight_edges = [] 

    while queue:
        current_t, depth = queue.pop(0)
        if depth >= hop_limit:
            continue
        
        parents = model.get_parents(current_t)
        for p in parents:
            if p.endswith(("_k","_q","_c","_g")) or "_p_" in p:
                grandparents = model.get_parents(p)
                for gp in grandparents:
                    if gp.endswith("_t"):
                        if gp not in visited:
                            visited.add(gp)
                            reachable_t_nodes.add(gp)
                            queue.append((gp, depth + 1))

                        etype = "other"
                        if p.endswith("_k"): etype = "ac"
                        elif p.endswith("_q"): etype = "pilot"
                        elif p.endswith("_c"): etype = "cabin"
                        elif "_p_" in p: etype = "pax"
                        elif p.endswith("_g"): etype = "ground"

                        src_id = gp.replace("_t", "")
                        dst_id = current_t.replace("_t", "")
                        
                        edge_data = {
                            "from": src_id,
                            "to": dst_id,
                            "resource": p,
                            "type": etype
                        }
                        flight_edges.append(edge_data)

    return reachable_t_nodes, flight_edges

def _batch_query(infer, targets: List[str], evidence: Dict[str, Any], chunk_size: int = 1) -> Dict[str, Any]:
    """
    Query Inference Engine sequentially.
    """
    # Previously filtered evidence; now we include it to ensure completeness for Visualization.
    clean_targets = targets 
            
    results = {}
    
    for i, t in enumerate(clean_targets):
        try:
             q_res = infer.query([t], evidence=evidence, show_progress=False, joint=False)
             for node, factor in q_res.items():
                 results[node] = {i: float(v) for i, v in enumerate(factor.values)}
        except Exception as e:
            print(f"Query Failed for {t}: {e}")
            
    return results
