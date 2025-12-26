
def _find_ancestor_subgraph(model, start_nodes: list, hop_limit: int = 5):
    """
    Reverse BFS to find upstream ancestors (contributories).
    Returns:
       - ancestor_t_nodes (set): Set of Flight_t nodes affecting target
       - flight_edges (list): [(src_fid, dst_fid, resource_node, edge_type), ...]
    """
    queue = [(node, 0) for node in start_nodes]   # start_nodes are *_t
    visited = set(start_nodes)
    reachable_t_nodes = set(start_nodes)
    flight_edges = []  # (src_fid, dst_fid, resource_node, type)

    while queue:
        current_t, depth = queue.pop(0)
        # If we reached limit, we don't expand further, but we still validly visited current_t
        if depth >= hop_limit:
            continue

        # In BN, Upstream -> Resource -> Current_t
        # So Parent(Current_t) = Resource
        # Parent(Resource) = Upstream_t
        
        parents = model.get_parents(current_t)
        for p in parents:
            # Check if p is a resource node
            if p.endswith(("_k","_q","_c","_g")) or "_p_" in p:
                # Upstream flights are parents of the resource node
                grandparents = model.get_parents(p)
                for gp in grandparents:
                    if gp.endswith("_t"):
                        # Found an upstream flight (gp -> p -> current_t)
                        
                        # Add to queue if new
                        if gp not in visited:
                            visited.add(gp)
                            reachable_t_nodes.add(gp)
                            queue.append((gp, depth + 1))

                        # Record Edge for Visualization
                        # Extract Type
                        etype = "other"
                        if p.endswith("_k"): etype = "ac"
                        elif p.endswith("_q"): etype = "pilot"
                        elif p.endswith("_c"): etype = "cabin"
                        elif "_p_" in p: etype = "pax"
                        elif p.endswith("_g"): etype = "ground" # Should not happen usually for flight-to-flight but robust to have

                        # Src: gp[:-2] (F001), Dst: current_t[:-2] (F002)
                        # We store IDs without _t suffix for cleaner edge mapping
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
