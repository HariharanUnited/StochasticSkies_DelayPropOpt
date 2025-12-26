"""
Build pgmpy BayesianModel from CPT JSON (inter-flight structure) and run inference.
"""
from itertools import product
from typing import Dict, List, Any

import importlib
import json

# Lazy imports
ModelClass = None
TabularCPD = None
VariableElimination = None


def _ensure_pgmpy():
    global ModelClass, TabularCPD, VariableElimination
    if ModelClass is not None:
        return
    models_mod = importlib.import_module("pgmpy.models")
    ModelClass = (
        getattr(models_mod, "DiscreteBayesianNetwork", None)
        or getattr(models_mod, "BayesianModel", None)
        or getattr(models_mod, "BayesianNetwork", None)
    )
    if ModelClass is None:
        raise ImportError("No Bayesian model class found in pgmpy.models")
    TabularCPD = importlib.import_module("pgmpy.factors.discrete").TabularCPD
    VariableElimination = importlib.import_module("pgmpy.inference").VariableElimination


def _infer_g_states_global(cpts_raw):
    g_set = set()
    for table in cpts_raw.values():
        for key in table.keys():
            g_set.add(key.split("|")[-1])
    return sorted(g_set)


def _probabilistic_cpd(child: str, parent: str, mapping: Dict[int, Dict[int, float]], num_bins: int) -> TabularCPD:
    values = [[0.0] * num_bins for _ in range(num_bins)]
    for tbin, dist in mapping.items():
        for out_bin, p in dist.items():
            values[int(out_bin)][int(tbin)] = float(p)
        # normalize column
        col_sum = sum(values[r][int(tbin)] for r in range(num_bins))
        if col_sum == 0:
            for r in range(num_bins):
                values[r][int(tbin)] = 1.0 / num_bins
        else:
            for r in range(num_bins):
                values[r][int(tbin)] /= col_sum
    return TabularCPD(
        variable=child,
        variable_card=num_bins,
        values=values,
        evidence=[parent],
        evidence_card=[num_bins],
        state_names={child: list(range(num_bins)), parent: list(range(num_bins))},
    )


import numpy as np

def _expand_t_cpd_wrapper(args):
    """Wrapper for ProcessPoolExecutor"""
    from dptbn.bn_pgmpy import _expand_t_cpd, _ensure_pgmpy
    _ensure_pgmpy()
    return _expand_t_cpd(*args)

def _expand_t_cpd(flight_id: str, table: Dict[str, Dict[str, float]], parents_order: List[str], g_states: List[str], num_bins: int) -> TabularCPD:
    # Vectorized construction using numpy
    # Shape: [card(p1), card(p2), ..., card(pk), card(child)]
    
    # 1. Determine cardinality for each parent
    parent_cards = []
    for p in parents_order:
        if p == "g":
            parent_cards.append(len(g_states))
        else:
            parent_cards.append(num_bins)
            
    # 2. Initialize CPD array and Count array
    shape = tuple(parent_cards) + (num_bins,)
    cpd_tensor = np.zeros(shape, dtype=np.float64)
    # Using small epsilon to avoid divide by zero if we init with zeros? 
    # Better: Initialize with uniform prior to avoid holes
    cpd_tensor.fill(1.0 / num_bins) 
    
    # Track counts for averaging collisions
    count_tensor = np.zeros(shape[:-1], dtype=np.float64) # Same shape minus child axis

    # 3. Fill from table
    g_map = {g: i for i, g in enumerate(g_states)}
    
    # ... (Key Parsing Logic Remains the same) ...
    for key_str, probs in table.items():
        parts = key_str.split("|")
        # Check if we have mismatch in length
        # Standard: len(parts) == original_parents_count
        # We only want to extract values for our current 'parents_order'.
        
        # ... (Indices Logic) ...
        indices = []
        valid = True
        
        # Mapping logic:
        # If len(parts) == len(parents_order), map 1:1.
        # If len(parts) > len(parents_order), map:
        #   0, 1, 2 (k,q,c) -> 0, 1, 2
        #   Last (g) -> Last part
        #   Pax: indices 3 to (3 + num_pax - 1). 
        #   We kept 'num_pax' parents.
        
        num_kept = len(parents_order)
        num_parts = len(parts)
        
        if num_kept == num_parts:
             # Standard 1:1
             kept_indices_in_key = range(num_parts)
        else:
             # Pruning happened. Assume standard [k, q, c, ...pax..., g] structure.
             # We kept k, q, c (3 items). g (1 item). Pax (num_kept - 4 items).
             num_pax_kept = max(0, num_kept - 4)
             # We take first indices 0, 1, 2.
             # Then we take next num_pax_kept.
             # Then we take the LAST index for g.
             kept_indices_in_key = list(range(3 + num_pax_kept)) + [num_parts - 1]
             # Safety check: if num_parts is small (no pax?), handle gracefullly
             if num_parts < 4: kept_indices_in_key = range(num_parts) # Fallback

        current_p_idx = 0
        for i in kept_indices_in_key:
            if i >= len(parts): break # Safety
            val_str = parts[i]
            # Use current parent definition
            p_name = parents_order[current_p_idx]
            
            if p_name == "g" or p_name.endswith("_g"): # Was "g" or "F001:g", now "F001_g"
                 if val_str not in g_map:
                     valid = False
                     break
                 indices.append(g_map[val_str])
            else:
                 indices.append(int(float(val_str)))
            current_p_idx += 1
            
        if not valid:
            continue
            
        # Write probabilities...
        # Convert probs dict to array
        prob_arr = np.zeros(num_bins)
        for cval, p in probs.items():
            prob_arr[int(cval)] = p
            
        # Normalize source row
        s = prob_arr.sum()
        if s > 0: prob_arr /= s
        else: prob_arr.fill(1.0/num_bins)
        
        # If this cell was pristine (count=0), clear the default uniform value first
        idx_tuple = tuple(indices)
        if count_tensor[idx_tuple] == 0:
            cpd_tensor[idx_tuple] = 0.0
            
        cpd_tensor[idx_tuple] += prob_arr 
        count_tensor[idx_tuple] += 1
        
    # Average based on counts
    # Avoid division by zero
    # We expand count_tensor to match cpd_tensor shape for broadcasting
    c_expanded = count_tensor[..., np.newaxis]
    
    
    # Where count > 0, divide. Where count == 0, keep the initialized uniform value.
    # Use np.where to create a safe divisor (replace 0 with 1 to avoid NaN, though we don't use those values)
    # Where count > 0, divide. Where count == 0, we apply Physics Fallback.
    safe_divisor = np.where(c_expanded > 0, c_expanded, 1.0)
    cpd_tensor /= safe_divisor
    
    # Physics Fallback for Unseen Combinations (Count == 0)
    # If we never saw (K=4, Q=0), we shouldn't assume Uniform.
    # We assume Flight Delay = Max(Resource Delays).
    zero_indices = np.argwhere(count_tensor == 0)
    for idx in zero_indices:
        idx_t = tuple(idx)
        # Determine likely bin from Resource Parents (exclude 'g')
        res_bins = []
        for i, pname in enumerate(parents_order):
            if pname != "g": # It's a resource/bin parent
                 res_bins.append(idx[i])
        
        # Max logic: Flight delay is driven by the worst resource
        target_bin = max(res_bins) if res_bins else 0
        if target_bin >= num_bins: target_bin = num_bins - 1
        
        # Set Identity-like Distribution (High confidence)
        dist = np.zeros(num_bins)
        dist[target_bin] = 0.9
        if target_bin > 0: dist[target_bin-1] = 0.05
        if target_bin < num_bins-1: dist[target_bin+1] = 0.05
        # Normalize
        s_d = dist.sum()
        if s_d > 0: dist /= s_d
        
        cpd_tensor[idx_t] = dist
    
    # Normalize final tensor to be sure
    s_tensor = cpd_tensor.sum(axis=-1, keepdims=True)
    s_tensor[s_tensor == 0] = 1.0
    cpd_tensor /= s_tensor

    # 4. Reshape for pgmpy: (child, product_parents)
    # Move child axis to front: [child, p1, p2, ...]
    cpd_tensor_t = np.moveaxis(cpd_tensor, -1, 0)
    
    # Flatten parent dimensions
    flat_values = cpd_tensor_t.reshape(num_bins, -1)
    
    evidence_nodes = [f"{flight_id}_{p}" for p in parents_order]
    evidence_card = parent_cards
    state_names = {f"{flight_id}_{p}": (g_states if p == "g" else list(range(num_bins))) for p in parents_order}
    state_names[f"{flight_id}_t"] = list(range(num_bins))

    return TabularCPD(
        variable=f"{flight_id}_t",
        variable_card=num_bins,
        values=flat_values,
        evidence=evidence_nodes,
        evidence_card=evidence_card,
        state_names=state_names,
    )


def build_model(cpts_path: str) -> VariableElimination:
    _ensure_pgmpy()
    data = json.load(open(cpts_path, "r"))
    cpts_raw = data["cpts"]
    res_cpds = data["resource_cpds"]
    bins = data["metadata"]["delay_bins"]
    g_states = data["metadata"].get("g_states") or _infer_g_states_global(cpts_raw)
    g_priors = data.get("g_priors", {})
    num_bins = len(bins) + 1

    edges = []
    nodes = set()

    # Ensure every flight has a resource entry
    # Ensure every flight has a resource entry
    for fid in cpts_raw.keys():
        if fid not in res_cpds:
            res_cpds[fid] = {"k": {}, "q": {}, "c": {}, "pax": {}, "pax_parents": [], "parents_order": ["k", "q", "c", "g"]}

    # Pre-process: Apply Memory Safety Pruning & Sanitization to res_cpds IN PLACE
    # This ensures both Edge creation (Loop 1) and CPD creation (Loop 2) see the same parents.
    MAX_PARENTS = 8
    for fid in list(res_cpds.keys()):
        res = res_cpds[fid]
        original_parents = res.get("parents_order", [])
        
        # 1. Pruning
        if len(original_parents) > MAX_PARENTS:
            # Structure is [k, q, c, ...pax..., g]
            num_keep_pax = max(0, MAX_PARENTS - 4)
            pax_ref = original_parents[3:-1]
            kept_pax_ref = pax_ref[:num_keep_pax]
            
            # Construct pruned parents_order
            pruned_order = original_parents[:3] + kept_pax_ref + [original_parents[-1]]
            
            # We must also update the 'pax_parents' list and 'pax' dict to match!
            # The 'pax_parents' list in res_cpds usually contains IDs.
            # 'pax_ref' are strings like 'p_F123_bin'.
            # We need to map back or just filter res['pax_parents'] based on what we kept.
            
            # Simpler: We know 'pax_parents' aligns with the middle slice.
            # kept_pax_ref contains 'p_Fxxx_bin'.
            # We extract Fxxx from it to filter res['pax_parents'].
            kept_pax_ids = []
            for p_str in kept_pax_ref:
                # p_F123_bin -> F123
                pid = p_str.replace("p_", "").replace("_bin", "")
                kept_pax_ids.append(pid)
                
            print(f"  WARNING: Pruning {fid} parents from {len(original_parents)} to {len(pruned_order)}. Kept {len(kept_pax_ids)} pax parents.")
            
            # Apply to res structure
            res["parents_order"] = pruned_order
            res["pax_parents"] = kept_pax_ids
            # Remove dropped pax from 'pax' dict to prevent edge creation for them
            if "pax" in res:
                res["pax"] = {pid: m for pid, m in res["pax"].items() if pid in kept_pax_ids}

    # 2. Sanitize Variable Names
    # Replace '_bin' in parents_order for all.
    for fid, res in res_cpds.items():
        res["parents_order"] = [p.replace("_bin", "") for p in res["parents_order"]]

    # Ensure all nodes exist
    for fid, res in res_cpds.items():
        nodes.add(f"{fid}_t")
        nodes.add(f"{fid}_g")
        nodes.add(f"{fid}_k")
        nodes.add(f"{fid}_q")
        nodes.add(f"{fid}_c")
        for pfid in res.get("pax_parents", []):
            nodes.add(f"{fid}_p_{pfid}")
        # upstream t nodes
        for inbound in list(res.get("k", {}).keys()) + list(res.get("q", {}).keys()) + list(res.get("c", {}).keys()) + list(res.get("pax", {}).keys()):
            nodes.add(f"{inbound}_t")

    # Build edges and CPDs
    cpds = []

    # Build resource CPDs and edges
    for fid, res in res_cpds.items():
        child_t = f"{fid}_t"

        def _normalize_mapping(mapping: Dict[Any, Any]) -> Dict[int, Dict[int, float]]:
            norm = {}
            for tbin, dist in mapping.items():
                tbin_i = int(tbin)
                if isinstance(dist, dict):
                    norm[tbin_i] = {int(k): float(v) for k, v in dist.items()}
                else:
                    # treat scalar as deterministic to that bin
                    norm[tbin_i] = {int(dist): 1.0}
            return norm

        # aircraft
        if res.get("k"):
            for inbound, mapping in res.get("k", {}).items():
                res_node = f"{fid}_k"
                edges.append((f"{inbound}_t", res_node))
                edges.append((res_node, child_t))
                cpds.append(_probabilistic_cpd(res_node, f"{inbound}_t", _normalize_mapping(mapping), num_bins))
        else:
            res_node = f"{fid}_k"
            edges.append((res_node, child_t))
            cpds.append(
                TabularCPD(
                    variable=res_node,
                    variable_card=num_bins,
                    values=[[1.0 / num_bins] for _ in range(num_bins)],
                    state_names={res_node: list(range(num_bins))},
                )
            )
        # pilot
        if res.get("q"):
            for inbound, mapping in res.get("q", {}).items():
                res_node = f"{fid}_q"
                edges.append((f"{inbound}_t", res_node))
                edges.append((res_node, child_t))
                cpds.append(_probabilistic_cpd(res_node, f"{inbound}_t", _normalize_mapping(mapping), num_bins))
        else:
            res_node = f"{fid}_q"
            edges.append((res_node, child_t))
            cpds.append(
                TabularCPD(
                    variable=res_node,
                    variable_card=num_bins,
                    values=[[1.0 / num_bins] for _ in range(num_bins)],
                    state_names={res_node: list(range(num_bins))},
                )
            )
        # cabin
        if res.get("c"):
            for inbound, mapping in res.get("c", {}).items():
                res_node = f"{fid}_c"
                edges.append((f"{inbound}_t", res_node))
                edges.append((res_node, child_t))
                cpds.append(_probabilistic_cpd(res_node, f"{inbound}_t", _normalize_mapping(mapping), num_bins))
        else:
            res_node = f"{fid}_c"
            edges.append((res_node, child_t))
            cpds.append(
                TabularCPD(
                    variable=res_node,
                    variable_card=num_bins,
                    values=[[1.0 / num_bins] for _ in range(num_bins)],
                    state_names={res_node: list(range(num_bins))},
                )
            )
        # pax
        if res.get("pax"):
            for inbound, mapping in res.get("pax", {}).items():
                res_node = f"{fid}_p_{inbound}"
                edges.append((f"{inbound}_t", res_node))
                edges.append((res_node, child_t))
                cpds.append(_probabilistic_cpd(res_node, f"{inbound}_t", _normalize_mapping(mapping), num_bins))
        else:
            for inbound in res.get("pax_parents", []):
                res_node = f"{fid}_p_{inbound}"
                edges.append((res_node, child_t))
                cpds.append(
                    TabularCPD(
                        variable=res_node,
                        variable_card=num_bins,
                        values=[[1.0 / num_bins] for _ in range(num_bins)],
                        state_names={res_node: list(range(num_bins))},
                    )
                )
        # g root uniform
        card_g = len(g_states)
        prior = g_priors.get(fid)
        if prior and card_g:
            vals = [float(prior.get(g, 0.0)) for g in g_states]
            s = sum(vals)
            if s == 0:
                vals = [1.0 / card_g] * card_g
            else:
                vals = [v / s for v in vals]
            cpds.append(
                TabularCPD(
                    variable=f"{fid}_g",
                    variable_card=card_g,
                    values=[[v] for v in vals],
                    state_names={f"{fid}_g": g_states},
                )
            )
        else:
            cpds.append(
                TabularCPD(
                    variable=f"{fid}_g",
                    variable_card=card_g,
                    values=[[1.0 / card_g] for _ in range(card_g)],
                    state_names={f"{fid}_g": g_states},
                )
            )
        edges.append((f"{fid}_g", child_t))

    # Child CPDs
    # Parallelize CPD creation
    # ProcessPoolExecutor removed per user request (instability)
    print(f"Building {len(cpts_raw)} CPDs (Sequential)...")
    child_cpds = []
    for fid, table in cpts_raw.items():
        parents_order = res_cpds[fid]["parents_order"]
        # Direct call
        cpd = _expand_t_cpd(fid, table, parents_order, g_states, num_bins)
        child_cpds.append(cpd)
    
    cpds.extend(child_cpds)
            
    print(f"Building CPDs complete. Assembling network...        ")
    
    model = ModelClass(edges)
    model.add_cpds(*cpds)
    # model.check_model()  # Too slow for large networks (300+ nodes)
    model.g_states = g_states
    
    # Return the model directly. 
    # Do NOT wrap in VariableElimination here, as it triggers expensive pre-computation 
    # and makes pickling slow/large.
    return model


def query_posterior(infer: VariableElimination, flight_id: str, evidence: Dict[str, Any]) -> Dict[int, float]:
    ev_pgmpy = {}
    g_states = getattr(infer.model, "g_states", None)
    for k, v in evidence.items():
        if k == "g" and g_states:
            ev_pgmpy[f"{flight_id}_{k}"] = v
        else:
            ev_pgmpy[f"{flight_id}_{k}"] = v
    child_t = f"{flight_id}_t"
    
    query_var = f"{flight_id}_t"
    result = infer.query([query_var], evidence=ev_pgmpy, show_progress=False)
    dist = result[query_var] if not hasattr(result, "values") else result
    vals = dist.values
    return {i: float(vals[i]) for i in range(len(vals))}
