# import time
# from typing import Any, Dict, List, Optional, Tuple

# from fastapi import HTTPException

# from dptbn.config import default_config
# from dptbn.graph_algo import _batch_query
# from dptbn.math_engine import calculate_expected_delay, calculate_prob_delay
# from dptbn.schemas import (
#     EdgeStyle,
#     FlightState,
#     GlobalResponse,
#     MetaInfo,
#     NodeStyle,
#     SummaryPanel,
#     Visuals,
# )


# from collections import defaultdict, deque
# from typing import Dict, List, Set, Tuple

# # -------------------------
# # Speed knobs (tune safely)
# # -------------------------
# SAMPLE_EDGES_PER_CAT = 25      # Stage-1 sampling per category
# CANDIDATES_PER_CAT = 8         # Stage-2 do() evaluations per category
# TOP_K_PER_CAT = 10             # final displayed per category
# BASE_QUERY_CHUNK = 200         # chunk for baseline resource query
# FLIGHT_QUERY_CHUNK = 30        # chunk for baseline flight query
# HOP_LIMIT = 5                   # max hops for flight descendants
# MAX_SCOPE_TARGETS = 120       # max targets for flight descendants


# def _build_flight_adj_from_inventory(inventory: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
#     """
#     Flight-to-flight adjacency using all inbound arcs we can infer.
#     Includes pax edges too (for propagation), even if we don't score pax.
#     """
#     adj: Dict[str, List[str]] = defaultdict(list)
#     for cat, edges in inventory.items():
#         for e in edges:
#             src = e.get("src_fid")
#             dst = e.get("dst_fid")
#             if src and dst and src != dst:
#                 adj[src].append(dst)
#     return adj

# def _descendants_flights(start: List[str], adj: Dict[str, List[str]], hop_limit: int) -> Set[str]:
#     seen: Set[str] = set()
#     q = deque([(s, 0) for s in start if s])
#     while q:
#         node, d = q.popleft()
#         if node in seen or d > hop_limit:
#             continue
#         seen.add(node)
#         for nxt in adj.get(node, []):
#             if nxt not in seen:
#                 q.append((nxt, d + 1))
#     return seen


# def _score_edges_hybrid(
#     model,
#     infer_engine,
#     inventory: Dict[str, List[Dict[str, Any]]],
# ) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     Hybrid scoring:
#       - pax: local do-delta on dst flight (original behavior)
#       - ac/pilot/cabin: cascade do-delta summed over descendants scope (hop-limited)
#     Output fields remain:
#       prob_bad, effect_minutes, score (+ optional shock_minutes/scope_size)
#     """
#     bins = default_config.delay_bins

#     # Build adjacency once (includes pax edges for propagation scope)
#     adj = _build_flight_adj_from_inventory(inventory)

#     # ---- Stage 1: baseline risk for ALL categories (including pax) ----
#     sampled_inventory: Dict[str, List[Dict[str, Any]]] = {
#         cat: edges[:SAMPLE_EDGES_PER_CAT] for cat, edges in inventory.items()
#     }

#     all_resource_nodes = [e["resource_node"] for edges in sampled_inventory.values() for e in edges]
#     base_resource_results = _batch_query(infer_engine, all_resource_nodes, evidence={}, chunk_size=BASE_QUERY_CHUNK)
#     base_prob = {r: _prob_bad(dist) for r, dist in base_resource_results.items()}

#     # Candidate selection
#     top_candidates: Dict[str, List[Dict[str, Any]]] = {}
#     for cat, edges in sampled_inventory.items():
#         edges_with_prob = [{**e, "prob_bad": base_prob.get(e["resource_node"], 0.0)} for e in edges]

#         # For cascade cats, use a proxy that prefers wide scopes (faster than full do)
#         if cat in ("ac", "pilot", "cabin"):
#             for x in edges_with_prob:
#                 dst = x.get("dst_fid")
#                 scope = _descendants_flights([dst], adj, HOP_LIMIT) if dst else set()
#                 x["scope_size"] = len(scope)
#                 x["_proxy"] = float(x["prob_bad"]) * (1.0 + float(x["scope_size"]))
#             edges_with_prob.sort(key=lambda x: x["_proxy"], reverse=True)
#         else:
#             # pax: just by prob_bad
#             edges_with_prob.sort(key=lambda x: x["prob_bad"], reverse=True)

#         top_candidates[cat] = edges_with_prob[:CANDIDATES_PER_CAT]

#     # ---- Stage 2: do() evaluations ----
#     scored: Dict[str, List[Dict[str, Any]]] = {}
#     do_engine_cache: Dict[str, Any] = {}
#     exp_cache: Dict[Tuple[str, str, int], float] = {}

#     for cat, edges in top_candidates.items():
#         out = []
#         for e in edges:
#             res_node = e["resource_node"]

#             try:
#                 card = int(model.get_cardinality(res_node))
#             except Exception:
#                 card = 5

#             good_state = 0
#             bad_state = max(0, card - 1)

#             # -------------------------
#             # pax: ORIGINAL local scoring
#             # -------------------------
#             if cat == "pax":
#                 dst_t = e["dst_t"]

#                 e_good = _expected_delay_under_do(
#                     model, res_node, good_state, dst_t, bins, exp_cache, do_engine_cache
#                 )
#                 e_bad = _expected_delay_under_do(
#                     model, res_node, bad_state, dst_t, bins, exp_cache, do_engine_cache
#                 )
#                 effect_minutes = max(0.0, float(e_bad - e_good))
#                 score = float(e["prob_bad"]) * effect_minutes

#                 out.append({**e, "effect_minutes": effect_minutes, "score": score})
#                 continue

#             # --------------------------------------------
#             # ac/pilot/cabin: CASCADE scoring over a scope
#             # --------------------------------------------
#             dst_fid = e.get("dst_fid")
#             scope_flights = list(_descendants_flights([dst_fid], adj, HOP_LIMIT)) if dst_fid else []
#             if len(scope_flights) > MAX_SCOPE_TARGETS:
#                 scope_flights = scope_flights[:MAX_SCOPE_TARGETS]

#             shock = 0.0
#             for fid in scope_flights:
#                 tnode = f"{fid}_t"
#                 if tnode not in model.nodes():
#                     continue

#                 e_good = _expected_delay_under_do(
#                     model, res_node, good_state, tnode, bins, exp_cache, do_engine_cache
#                 )
#                 e_bad = _expected_delay_under_do(
#                     model, res_node, bad_state, tnode, bins, exp_cache, do_engine_cache
#                 )
#                 shock += max(0.0, float(e_bad - e_good))

#             # IMPORTANT: keep the same field name the frontend expects: "effect_minutes"
#             effect_minutes = float(shock)
#             score = float(e["prob_bad"]) * effect_minutes

#             out.append({
#                 **e,
#                 "effect_minutes": effect_minutes,   # <-- frontend parses "Effect:"
#                 "shock_minutes": effect_minutes,    # optional extra
#                 "scope_size": len(scope_flights),    # optional extra
#                 "score": score,
#             })

#         out.sort(key=lambda x: x["score"], reverse=True)
#         scored[cat] = out[:TOP_K_PER_CAT]

#     # Ensure all keys exist for UI stability
#     for k in ["ac", "pilot", "cabin", "pax"]:
#         scored.setdefault(k, [])

#     return scored

# def _score_edges_cascade(
#     model,
#     infer_engine,
#     inventory: Dict[str, List[Dict[str, Any]]],
# ) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     Swap-like cascade score per edge (aircraft/pilot/cabin):
#       score = P(resource bad) * sum_{f in scope} max(0, E[f|do(bad)] - E[f|do(good)])
#     Scope = descendants of dst flight in the full flight adjacency graph.
#     """
#     bins = default_config.delay_bins

#     # Build adjacency once (includes pax for propagation)
#     adj = _build_flight_adj_from_inventory(inventory)

#     # ---- Stage 1: baseline risk (same as your code) ----
#     sampled_inventory = {
#         cat: edges[:SAMPLE_EDGES_PER_CAT]
#         for cat, edges in inventory.items()
#         if cat in ("ac", "pilot", "cabin")  # <-- only score these
#     }

#     all_resource_nodes = [e["resource_node"] for edges in sampled_inventory.values() for e in edges]
#     base_resource_results = _batch_query(infer_engine, all_resource_nodes, evidence={}, chunk_size=BASE_QUERY_CHUNK)
#     base_prob = {r: _prob_bad(dist) for r, dist in base_resource_results.items()}

#     top_candidates: Dict[str, List[Dict[str, Any]]] = {}
#     for cat, edges in sampled_inventory.items():
#         edges_with_prob = [{**e, "prob_bad": base_prob.get(e["resource_node"], 0.0)} for e in edges]
#         # quick proxy: prob_bad * (1 + descendants count) to pick candidates fast
#         for x in edges_with_prob:
#             dst = x.get("dst_fid")
#             scope = _descendants_flights([dst], adj, HOP_LIMIT) if dst else set()
#             x["scope_size"] = len(scope)
#             x["_proxy"] = float(x["prob_bad"]) * (1.0 + float(x["scope_size"]))
#         edges_with_prob.sort(key=lambda x: x["_proxy"], reverse=True)
#         top_candidates[cat] = edges_with_prob[:CANDIDATES_PER_CAT]

#     # ---- Stage 2: causal cascade shock ----
#     scored: Dict[str, List[Dict[str, Any]]] = {}
#     do_engine_cache: Dict[str, Any] = {}
#     exp_cache: Dict[Tuple[str, str, int], float] = {}

#     for cat, edges in top_candidates.items():
#         out = []
#         for e in edges:
#             res_node = e["resource_node"]
#             dst_fid = e.get("dst_fid")

#             # descendants scope on flight ids, then map to *_t nodes
#             scope_flights = list(_descendants_flights([dst_fid], adj, HOP_LIMIT)) if dst_fid else []
#             if len(scope_flights) > MAX_SCOPE_TARGETS:
#                 scope_flights = scope_flights[:MAX_SCOPE_TARGETS]

#             try:
#                 card = int(model.get_cardinality(res_node))
#             except Exception:
#                 card = 5

#             good_state = 0
#             bad_state = max(0, card - 1)

#             shock = 0.0
#             for fid in scope_flights:
#                 tnode = f"{fid}_t"
#                 if tnode not in model.nodes():
#                     continue

#                 e_good = _expected_delay_under_do(
#                     model, res_node, good_state, tnode, bins, exp_cache, do_engine_cache
#                 )
#                 e_bad = _expected_delay_under_do(
#                     model, res_node, bad_state, tnode, bins, exp_cache, do_engine_cache
#                 )
#                 shock += max(0.0, float(e_bad - e_good))

#             score = float(e["prob_bad"]) * float(shock)

#             out.append({
#                 **e,
#                 "shock_minutes": float(shock),
#                 "scope_size": len(scope_flights),
#                 "score": float(score),
#             })

#         out.sort(key=lambda x: x["score"], reverse=True)
#         scored[cat] = out[:TOP_K_PER_CAT]

#     return scored


# def _edge_palette(edge_type: str) -> str:
#     return {
#         "ac": "#2563eb",     # aircraft
#         "pilot": "#0ea5e9",  # pilot
#         "cabin": "#ef4444",  # cabin crew
#         "pax": "#8b5cf6",    # passenger
#     }.get(edge_type, "#6b7280")


# def _edge_inventory(model) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     Extract edges grouped by category.
#     Returns dict: {category: [{resource_node, src_fid, dst_fid, dst_t, type}, ...]}
#     """
#     inventory = {"ac": [], "pilot": [], "cabin": [], "pax": []}

#     for node in model.nodes():
#         # Aircraft edge: Fprev_t -> Fchild_k -> Fchild_t
#         if node.endswith("_k"):
#             child = next((c for c in model.get_children(node) if c.endswith("_t")), None)
#             parent = next((p for p in model.get_parents(node) if p.endswith("_t")), None)
#             if child and parent:
#                 inventory["ac"].append(
#                     {
#                         "resource_node": node,
#                         "src_fid": parent[:-2],
#                         "dst_fid": child[:-2],
#                         "dst_t": child,
#                         "type": "ac",
#                     }
#                 )

#         # Pilot edge: Fprev_t -> Fchild_q -> Fchild_t
#         elif node.endswith("_q"):
#             child = next((c for c in model.get_children(node) if c.endswith("_t")), None)
#             parent = next((p for p in model.get_parents(node) if p.endswith("_t")), None)
#             if child and parent:
#                 inventory["pilot"].append(
#                     {
#                         "resource_node": node,
#                         "src_fid": parent[:-2],
#                         "dst_fid": child[:-2],
#                         "dst_t": child,
#                         "type": "pilot",
#                     }
#                 )

#         # Cabin edge: Fprev_t -> Fchild_c -> Fchild_t
#         elif node.endswith("_c"):
#             child = next((c for c in model.get_children(node) if c.endswith("_t")), None)
#             parent = next((p for p in model.get_parents(node) if p.endswith("_t")), None)
#             if child and parent:
#                 inventory["cabin"].append(
#                     {
#                         "resource_node": node,
#                         "src_fid": parent[:-2],
#                         "dst_fid": child[:-2],
#                         "dst_t": child,
#                         "type": "cabin",
#                     }
#                 )

#         # Pax edge: node looks like Fchild_p_Fparent
#         elif "_p_" in node:
#             child_id = node.split("_p_")[0]
#             child_t = f"{child_id}_t"
#             parent = next((p for p in model.get_parents(node) if p.endswith("_t")), None)
#             if parent:
#                 inventory["pax"].append(
#                     {
#                         "resource_node": node,
#                         "src_fid": parent[:-2],
#                         "dst_fid": child_id,
#                         "dst_t": child_t,
#                         "type": "pax",
#                     }
#                 )

#     # Deterministic ordering for reproducibility
#     for k in inventory:
#         inventory[k] = sorted(inventory[k], key=lambda e: (e["resource_node"], e["src_fid"], e["dst_fid"]))
#     return inventory


# def _prob_bad(dist: Optional[Dict[Any, float]]) -> float:
#     if not dist:
#         return 0.0
#     # "bad" = any state >= 1
#     return calculate_prob_delay(dist, threshold_bin_idx=1)


# # -------------------------
# # BIG SPEED FIX STARTS HERE
# # -------------------------

# def _get_do_infer(model, resource_node: str, do_engine_cache: Dict[str, Any]):
#     """
#     Cache: VariableElimination(model.do([resource_node])) per resource_node.
#     Building VE is expensive; do it once per resource.
#     """
#     if resource_node in do_engine_cache:
#         return do_engine_cache[resource_node]

#     from pgmpy.inference import VariableElimination
#     active = model.do([resource_node])
#     infer = VariableElimination(active)
#     do_engine_cache[resource_node] = infer
#     return infer


# def _expected_delay_under_do(
#     model,
#     resource_node: str,
#     state_val: int,
#     target: str,
#     bins,
#     result_cache: Dict[Tuple[str, str, int], float],
#     do_engine_cache: Dict[str, Any],
# ) -> float:
#     """
#     Compute E[target] under do(resource_node=state_val).
#     Uses two caches:
#       - do_engine_cache: resource_node -> VE engine on do-mutilated model
#       - result_cache: (resource_node, target, state) -> expected delay
#     """
#     key = (resource_node, target, int(state_val))
#     if key in result_cache:
#         return result_cache[key]

#     infer_do = _get_do_infer(model, resource_node, do_engine_cache)

#     try:
#         q = infer_do.query(
#             [target],
#             evidence={resource_node: int(state_val)},
#             show_progress=False,
#             joint=False,
#         )
#         factor = q[target]
#         dist = {i: float(v) for i, v in enumerate(factor.values)}
#         val = float(calculate_expected_delay(dist, bins))
#     except Exception:
#         val = 0.0

#     result_cache[key] = val
#     return val


# def _score_edges(model, infer_engine, inventory: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     Two-stage scoring:
#       Stage 1: baseline prob_bad(resource)
#       Stage 2: do() effect minutes on downstream dst flight
#       Score = prob_bad * effect_minutes
#     """
#     bins = default_config.delay_bins

#     # ---- Stage 1: baseline risk ----
#     sampled_inventory: Dict[str, List[Dict[str, Any]]] = {
#         cat: edges[:SAMPLE_EDGES_PER_CAT] for cat, edges in inventory.items()
#     }

#     all_resource_nodes = [e["resource_node"] for edges in sampled_inventory.values() for e in edges]
#     base_resource_results = _batch_query(infer_engine, all_resource_nodes, evidence={}, chunk_size=BASE_QUERY_CHUNK)
#     base_prob = {r: _prob_bad(dist) for r, dist in base_resource_results.items()}

#     top_candidates: Dict[str, List[Dict[str, Any]]] = {}
#     for cat, edges in sampled_inventory.items():
#         edges_with_prob = [{**e, "prob_bad": base_prob.get(e["resource_node"], 0.0)} for e in edges]
#         edges_with_prob.sort(key=lambda x: x["prob_bad"], reverse=True)
#         top_candidates[cat] = edges_with_prob[:CANDIDATES_PER_CAT]

#     # ---- Stage 2: causal effect ----
#     scored: Dict[str, List[Dict[str, Any]]] = {}
#     do_engine_cache: Dict[str, Any] = {}
#     exp_cache: Dict[Tuple[str, str, int], float] = {}

#     for cat, edges in top_candidates.items():
#         scored_list = []
#         for e in edges:
#             res_node = e["resource_node"]
#             dst_t = e["dst_t"]

#             try:
#                 card = int(model.get_cardinality(res_node))
#             except Exception:
#                 card = 5

#             good_state = 0
#             bad_state = max(0, card - 1)

#             e_good = _expected_delay_under_do(
#                 model, res_node, good_state, dst_t, bins, exp_cache, do_engine_cache
#             )
#             e_bad = _expected_delay_under_do(
#                 model, res_node, bad_state, dst_t, bins, exp_cache, do_engine_cache
#             )

#             effect_minutes = max(0.0, float(e_bad - e_good))
#             score = float(e["prob_bad"]) * effect_minutes

#             scored_list.append({**e, "effect_minutes": effect_minutes, "score": score})

#         scored_list.sort(key=lambda x: x["score"], reverse=True)
#         scored[cat] = scored_list[:TOP_K_PER_CAT]

#     return scored


# def run_connection_risk(req, model_wrapper, infer_engine) -> GlobalResponse:
#     """
#     Connection Risk Tool:
#       - Rank top risky edges per category (pax/crew/pilot/aircraft).
#       - Score = P(resource bad) * (E[dst|do(worst)] - E[dst|do(good)]).
#     """
#     t0 = time.time()
#     if not infer_engine:
#         raise HTTPException(500, "Inference engine not ready")

#     model = model_wrapper.model
#     bins = default_config.delay_bins

#     # 1) Build inventory and score
#     inventory = _edge_inventory(model)
#     scored = _score_edges_hybrid(model, infer_engine, inventory)

#     # 2) Build visuals (edges + involved nodes)
#     visual_edges: List[EdgeStyle] = []
#     involved_flights = set()

#     for cat, edges in scored.items():
#         for e in edges:
#             involved_flights.update([e["src_fid"], e["dst_fid"]])

#             color = _edge_palette(e["type"])
#             # thickness scaled by score; clamp for visuals
#             thickness = 5 #max(2, min(8, int(2 + e["score"])))
#             is_cascade = cat in ("ac", "pilot", "cabin")

#             tooltip = (
#                 f"{cat.upper()} edge\n"
#                 f"P(bad): {e['prob_bad']:.0%}\n"
#                 + (f"Scope: {e.get('scope_size', 0)} flights\n" if is_cascade else "")
#                 + f"Effect: {e['effect_minutes']:.1f}m\n"
#                 f"Score: {e['score']:.2f}\n"
#                 f"{e['src_fid']} → {e['dst_fid']}\n"
#                 f"Node: {e['resource_node']}"
#             )

#             visual_edges.append(
#                 EdgeStyle(
#                     **{"from": e["src_fid"], "to": e["dst_fid"]},
#                     color=color,
#                     thickness=thickness,
#                     tooltip=tooltip,
#                     type=cat,
#                 )
#             )

#     # 3) Node states for involved flights (baseline only)
#     flight_nodes = [f"{fid}_t" for fid in involved_flights]
#     base_flight_results = (
#         _batch_query(infer_engine, flight_nodes, evidence={}, chunk_size=FLIGHT_QUERY_CHUNK)
#         if flight_nodes
#         else {}
#     )

#     net_state: Dict[str, FlightState] = {}
#     visual_nodes: Dict[str, NodeStyle] = {}

#     for fnode in flight_nodes:
#         dist = base_flight_results.get(fnode, {}) or {}
#         prob_delay = calculate_prob_delay(dist, threshold_bin_idx=1)
#         exp_delay = float(calculate_expected_delay(dist, bins)) if dist else 0.0

#         ml_state = max(dist, key=dist.get) if dist else 0
#         ml_prob = float(dist.get(ml_state, 0.0)) if dist else 0.0

#         net_state[fnode] = FlightState(
#             prob_delay=float(prob_delay),
#             expected_delay=float(exp_delay),
#             most_likely_bin=int(ml_state) if str(ml_state).isdigit() else 0,
#             most_likely_bin_prob=ml_prob,
#             is_affected=True,
#         )

#         visual_nodes[fnode[:-2]] = NodeStyle(
#             color="#f8fafc",
#             border="2px solid #111111",
#             tooltip=f"P(delay>=bin1): {prob_delay:.0%}\nE[D]: {exp_delay:.1f}m",
#         )

#     # 4) Summary overlay
#     summary_stats = []
#     for cat in ["ac", "pilot", "cabin", "pax"]:
#         edges = scored.get(cat, [])
#         if edges:
#             top = edges[0]
#             summary_stats.append(
#                 {
#                     "label": f"Top {cat} edge",
#                     "value": f"{top['src_fid']}→{top['dst_fid']} (Score {top['score']:.2f})",
#                 }
#             )
#         summary_stats.append({"label": f"{cat} edges listed", "value": len(edges)})

#     calc_ms = (time.time() - t0) * 1000.0
#     visuals = Visuals(
#         heatmap_mode="prob_delay",
#         edges=visual_edges,
#         nodes=visual_nodes,
#         summary_panel=SummaryPanel(title="Connection Risk", stats=summary_stats),
#     )

#     return GlobalResponse(
#         meta_info=MetaInfo(tool_exec="connection_risk", calc_time_ms=calc_ms, model_name=req.model),
#         network_state=net_state,
#         visuals=visuals,
#     )


import time
import json
from pathlib import Path
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import HTTPException

from dptbn.config import default_config
from dptbn.graph_algo import _batch_query
from dptbn.math_engine import calculate_expected_delay, calculate_prob_delay
from dptbn.schemas import (
    EdgeStyle,
    FlightState,
    GlobalResponse,
    MetaInfo,
    NodeStyle,
    SummaryPanel,
    Visuals,
)

# -------------------------
# Speed knobs (tune safely)
# -------------------------
SAMPLE_EDGES_PER_CAT = 25      # Stage-1 sampling per category
CANDIDATES_PER_CAT = 8         # Stage-2 do() evaluations per category
TOP_K_PER_CAT = 10             # final displayed per category
BASE_QUERY_CHUNK = 200         # chunk for baseline resource query
FLIGHT_QUERY_CHUNK = 30        # chunk for baseline flight query
HOP_LIMIT = 10                  # max hops for flight descendants
MAX_SCOPE_TARGETS = 120        # max targets for flight descendants

# -------------------------
# "Cascade across lines" knob
# -------------------------
ALPHA_TAIL_SPREAD = 0.25      # multiplies score by (1 + alpha * new_tails)


def _data_dir_guess() -> Path:
    """
    Best-effort path to bn-viz/public/data (matches your server startup logic).
    Adjust if your repo layout differs.
    """
    return Path(__file__).resolve().parent.parent / "bn-viz" / "public" / "data"


def _load_tail_map(model_name: str) -> Dict[str, str]:
    """
    flight_id -> tail_id, from *_network_opt.json (preferred) else *_network.json.
    Used to upweight connections that spread across multiple tails ("lines of flying").
    """
    data_dir = _data_dir_guess()
    opt_path = data_dir / f"{model_name}_network_opt.json"
    base_path = data_dir / f"{model_name}_network.json"
    path = opt_path if opt_path.exists() else base_path

    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text())
        flights = payload.get("flights", []) or []
        return {f.get("flight_id", ""): (f.get("tail_id") or "") for f in flights if f.get("flight_id")}
    except Exception:
        return {}


def _build_flight_adj_from_inventory(inventory: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
    """
    Flight-to-flight adjacency using all inbound arcs we can infer.
    Includes pax edges too (for propagation scope), even if pax isn't cascade-scored.
    """
    adj: Dict[str, List[str]] = defaultdict(list)
    for _cat, edges in inventory.items():
        for e in edges:
            src = e.get("src_fid")
            dst = e.get("dst_fid")
            if src and dst and src != dst:
                adj[src].append(dst)
    return adj


def _descendants_flights(start: List[str], adj: Dict[str, List[str]], hop_limit: int) -> Set[str]:
    seen: Set[str] = set()
    q = deque([(s, 0) for s in start if s])
    while q:
        node, d = q.popleft()
        if node in seen or d > hop_limit:
            continue
        seen.add(node)
        for nxt in adj.get(node, []):
            if nxt not in seen:
                q.append((nxt, d + 1))
    return seen


def _edge_palette(edge_type: str) -> str:
    return {
        "ac": "#2563eb",     # aircraft
        "pilot": "#0ea5e9",  # pilot
        "cabin": "#ef4444",  # cabin crew
        "pax": "#8b5cf6",    # passenger
    }.get(edge_type, "#6b7280")


def _edge_inventory(model) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract edges grouped by category.
    Returns dict: {category: [{resource_node, src_fid, dst_fid, dst_t, type}, ...]}
    """
    inventory = {"ac": [], "pilot": [], "cabin": [], "pax": []}

    for node in model.nodes():
        # Aircraft edge: Fprev_t -> Fchild_k -> Fchild_t
        if node.endswith("_k"):
            child = next((c for c in model.get_children(node) if c.endswith("_t")), None)
            parent = next((p for p in model.get_parents(node) if p.endswith("_t")), None)
            if child and parent:
                inventory["ac"].append(
                    {
                        "resource_node": node,
                        "src_fid": parent[:-2],
                        "dst_fid": child[:-2],
                        "dst_t": child,
                        "type": "ac",
                    }
                )

        # Pilot edge: Fprev_t -> Fchild_q -> Fchild_t
        elif node.endswith("_q"):
            child = next((c for c in model.get_children(node) if c.endswith("_t")), None)
            parent = next((p for p in model.get_parents(node) if p.endswith("_t")), None)
            if child and parent:
                inventory["pilot"].append(
                    {
                        "resource_node": node,
                        "src_fid": parent[:-2],
                        "dst_fid": child[:-2],
                        "dst_t": child,
                        "type": "pilot",
                    }
                )

        # Cabin edge: Fprev_t -> Fchild_c -> Fchild_t
        elif node.endswith("_c"):
            child = next((c for c in model.get_children(node) if c.endswith("_t")), None)
            parent = next((p for p in model.get_parents(node) if p.endswith("_t")), None)
            if child and parent:
                inventory["cabin"].append(
                    {
                        "resource_node": node,
                        "src_fid": parent[:-2],
                        "dst_fid": child[:-2],
                        "dst_t": child,
                        "type": "cabin",
                    }
                )

        # Pax edge: node looks like Fchild_p_Fparent
        elif "_p_" in node:
            child_id = node.split("_p_")[0]
            child_t = f"{child_id}_t"
            parent = next((p for p in model.get_parents(node) if p.endswith("_t")), None)
            if parent:
                inventory["pax"].append(
                    {
                        "resource_node": node,
                        "src_fid": parent[:-2],
                        "dst_fid": child_id,
                        "dst_t": child_t,
                        "type": "pax",
                    }
                )

    # Deterministic ordering for reproducibility
    for k in inventory:
        inventory[k] = sorted(inventory[k], key=lambda e: (e["resource_node"], e["src_fid"], e["dst_fid"]))
    return inventory


def _prob_bad(dist: Optional[Dict[Any, float]]) -> float:
    if not dist:
        return 0.0
    # "bad" = any state >= 1
    return calculate_prob_delay(dist, threshold_bin_idx=1)


def _get_do_infer(model, resource_node: str, do_engine_cache: Dict[str, Any]):
    """
    Cache: VariableElimination(model.do([resource_node])) per resource_node.
    Building VE is expensive; do it once per resource.
    """
    if resource_node in do_engine_cache:
        return do_engine_cache[resource_node]

    from pgmpy.inference import VariableElimination
    active = model.do([resource_node])
    infer = VariableElimination(active)
    do_engine_cache[resource_node] = infer
    return infer


def _expected_delay_under_do(
    model,
    resource_node: str,
    state_val: int,
    target: str,
    bins,
    result_cache: Dict[Tuple[str, str, int], float],
    do_engine_cache: Dict[str, Any],
) -> float:
    """
    Compute E[target] under do(resource_node=state_val).
    Uses two caches:
      - do_engine_cache: resource_node -> VE engine on do-mutilated model
      - result_cache: (resource_node, target, state) -> expected delay
    """
    key = (resource_node, target, int(state_val))
    if key in result_cache:
        return result_cache[key]

    infer_do = _get_do_infer(model, resource_node, do_engine_cache)

    try:
        q = infer_do.query(
            [target],
            evidence={resource_node: int(state_val)},
            show_progress=False,
            joint=False,
        )
        factor = q[target]
        dist = {i: float(v) for i, v in enumerate(factor.values)}
        val = float(calculate_expected_delay(dist, bins))
    except Exception:
        val = 0.0

    result_cache[key] = val
    return val


def _score_edges_hybrid(
    model,
    infer_engine,
    inventory: Dict[str, List[Dict[str, Any]]],
    tail_map: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Hybrid scoring:
      - pax: local do-delta on dst flight (usual behavior)
      - ac/pilot/cabin: cascade do-delta summed over descendants scope (hop-limited)
        AND upweighted if the cascade spreads across multiple tails ("lines of flying").

    Fields returned (stable for frontend):
      prob_bad, effect_minutes, score
    Extra fields (optional):
      scope_size, shock_minutes, new_tails, tail_span, bridge_mult
    """
    bins = default_config.delay_bins
    tail_map = tail_map or {}

    # Build adjacency once (includes pax edges for propagation scope)
    adj = _build_flight_adj_from_inventory(inventory)

    # ---- Stage 1: baseline risk for ALL categories (including pax) ----
    sampled_inventory: Dict[str, List[Dict[str, Any]]] = {
        cat: edges[:SAMPLE_EDGES_PER_CAT] for cat, edges in inventory.items()
    }

    all_resource_nodes = [e["resource_node"] for edges in sampled_inventory.values() for e in edges]
    base_resource_results = _batch_query(infer_engine, all_resource_nodes, evidence={}, chunk_size=BASE_QUERY_CHUNK)
    base_prob = {r: _prob_bad(dist) for r, dist in base_resource_results.items()}

    # Candidate selection
    top_candidates: Dict[str, List[Dict[str, Any]]] = {}
    for cat, edges in sampled_inventory.items():
        edges_with_prob = [{**e, "prob_bad": base_prob.get(e["resource_node"], 0.0)} for e in edges]

        if cat in ("ac", "pilot", "cabin"):
            # Proxy prefers: high prob_bad + big scope + cross-tail spread
            for x in edges_with_prob:
                dst = x.get("dst_fid")
                scope = _descendants_flights([dst], adj, HOP_LIMIT) if dst else set()
                scope_list = list(scope)
                x["scope_size"] = len(scope_list)

                # approximate cross-tail spread on scope (cheap)
                dst_tail = tail_map.get(dst or "", "")
                tails_in_scope = {tail_map.get(fid, "") for fid in scope_list if tail_map.get(fid, "")}
                tails_in_scope.discard("")
                new_tails = len({t for t in tails_in_scope if t and t != dst_tail})

                x["tail_span"] = len(tails_in_scope)
                x["new_tails"] = new_tails

                x["_proxy"] = float(x["prob_bad"]) * (1.0 + float(x["scope_size"])) * (1.0 + ALPHA_TAIL_SPREAD * float(new_tails))

            edges_with_prob.sort(key=lambda x: x["_proxy"], reverse=True)
        else:
            # pax: usual selection by prob_bad
            edges_with_prob.sort(key=lambda x: x["prob_bad"], reverse=True)

        top_candidates[cat] = edges_with_prob[:CANDIDATES_PER_CAT]

    # ---- Stage 2: do() evaluations ----
    scored: Dict[str, List[Dict[str, Any]]] = {}
    do_engine_cache: Dict[str, Any] = {}
    exp_cache: Dict[Tuple[str, str, int], float] = {}

    for cat, edges in top_candidates.items():
        out: List[Dict[str, Any]] = []

        for e in edges:
            res_node = e["resource_node"]

            try:
                card = int(model.get_cardinality(res_node))
            except Exception:
                card = 5

            good_state = 0
            bad_state = max(0, card - 1)

            # -------------------------
            # pax: USUAL local scoring
            # -------------------------
            if cat == "pax":
                dst_t = e["dst_t"]

                e_good = _expected_delay_under_do(
                    model, res_node, good_state, dst_t, bins, exp_cache, do_engine_cache
                )
                e_bad = _expected_delay_under_do(
                    model, res_node, bad_state, dst_t, bins, exp_cache, do_engine_cache
                )
                effect_minutes = max(0.0, float(e_bad - e_good))
                score = float(e["prob_bad"]) * effect_minutes

                out.append({**e, "effect_minutes": effect_minutes, "score": score})
                continue

            # --------------------------------------------
            # ac/pilot/cabin: CASCADE scoring over a scope
            # + emphasize cross-tail spread ("line jumping")
            # --------------------------------------------
            dst_fid = e.get("dst_fid")
            scope_flights = list(_descendants_flights([dst_fid], adj, HOP_LIMIT)) if dst_fid else []
            if len(scope_flights) > MAX_SCOPE_TARGETS:
                scope_flights = scope_flights[:MAX_SCOPE_TARGETS]

            shock = 0.0
            for fid in scope_flights:
                tnode = f"{fid}_t"
                if tnode not in model.nodes():
                    continue

                e_good = _expected_delay_under_do(
                    model, res_node, good_state, tnode, bins, exp_cache, do_engine_cache
                )
                e_bad = _expected_delay_under_do(
                    model, res_node, bad_state, tnode, bins, exp_cache, do_engine_cache
                )
                shock += max(0.0, float(e_bad - e_good))

            # Cross-tail spread metrics
            dst_tail = tail_map.get(dst_fid or "", "")
            tails_in_scope = {tail_map.get(fid, "") for fid in scope_flights if tail_map.get(fid, "")}
            tails_in_scope.discard("")
            tail_span = len(tails_in_scope)
            new_tails = len({t for t in tails_in_scope if t and t != dst_tail})

            bridge_mult = 1.0 + ALPHA_TAIL_SPREAD * float(new_tails)

            # Frontend expects "Effect:" -> keep effect_minutes
            effect_minutes = float(shock)
            score = float(e["prob_bad"]) * effect_minutes * bridge_mult

            out.append({
                **e,
                "effect_minutes": effect_minutes,
                "shock_minutes": effect_minutes,
                "scope_size": len(scope_flights),
                "tail_span": int(tail_span),
                "new_tails": int(new_tails),
                "bridge_mult": float(bridge_mult),
                "score": float(score),
            })

        out.sort(key=lambda x: x["score"], reverse=True)
        scored[cat] = out[:TOP_K_PER_CAT]

    # Ensure all keys exist for UI stability
    for k in ["ac", "pilot", "cabin", "pax"]:
        scored.setdefault(k, [])

    return scored


def run_connection_risk(req, model_wrapper, infer_engine) -> GlobalResponse:
    """
    Connection Risk Tool:
      - pax: usual local do-delta scoring
      - ac/pilot/cabin: cascade do-delta over hop-limited descendants,
        upweighted for cross-tail spread (line jumping).
    """
    t0 = time.time()
    if not infer_engine:
        raise HTTPException(500, "Inference engine not ready")

    model = model_wrapper.model
    bins = default_config.delay_bins

    # 1) Build inventory and score
    inventory = _edge_inventory(model)
    tail_map = _load_tail_map(req.model)
    scored = _score_edges_hybrid(model, infer_engine, inventory, tail_map=tail_map)

    # 2) Build visuals (edges + involved nodes)
    visual_edges: List[EdgeStyle] = []
    involved_flights: Set[str] = set()

    for cat, edges in scored.items():
        for e in edges:
            involved_flights.update([e["src_fid"], e["dst_fid"]])

            color = _edge_palette(e["type"])
            thickness = 5

            is_cascade = cat in ("ac", "pilot", "cabin")
            extra = ""
            if is_cascade:
                extra = (
                    f"Scope: {e.get('scope_size', 0)} flights\n"
                    f"Cross-tails: +{e.get('new_tails', 0)} (span {e.get('tail_span', 0)})\n"
                )

            # NOTE: no "Link Risk" wording
            tooltip = (
                f"{cat.upper()} edge\n"
                f"P(bad): {e['prob_bad']:.0%}\n"
                f"{extra}"
                f"Effect: {e['effect_minutes']:.1f}m\n"
                f"Score: {e['score']:.2f}\n"
                f"{e['src_fid']} → {e['dst_fid']}\n"
                f"Node: {e['resource_node']}"
            )

            visual_edges.append(
                EdgeStyle(
                    **{"from": e["src_fid"], "to": e["dst_fid"]},
                    color=color,
                    thickness=thickness,
                    tooltip=tooltip,
                    type=cat,
                )
            )

    # 3) Node states for involved flights (baseline only)
    flight_nodes = [f"{fid}_t" for fid in involved_flights]
    base_flight_results = (
        _batch_query(infer_engine, flight_nodes, evidence={}, chunk_size=FLIGHT_QUERY_CHUNK)
        if flight_nodes
        else {}
    )

    net_state: Dict[str, FlightState] = {}
    visual_nodes: Dict[str, NodeStyle] = {}

    for fnode in flight_nodes:
        dist = base_flight_results.get(fnode, {}) or {}
        prob_delay = calculate_prob_delay(dist, threshold_bin_idx=1)
        exp_delay = float(calculate_expected_delay(dist, bins)) if dist else 0.0

        ml_state = max(dist, key=dist.get) if dist else 0
        ml_prob = float(dist.get(ml_state, 0.0)) if dist else 0.0

        net_state[fnode] = FlightState(
            prob_delay=float(prob_delay),
            expected_delay=float(exp_delay),
            most_likely_bin=int(ml_state) if str(ml_state).isdigit() else 0,
            most_likely_bin_prob=ml_prob,
            is_affected=True,
        )

        visual_nodes[fnode[:-2]] = NodeStyle(
            color="#f8fafc",
            border="2px solid #111111",
            tooltip=f"P(delay>=bin1): {prob_delay:.0%}\nE[D]: {exp_delay:.1f}m",
        )

    # 4) Summary overlay
    summary_stats = []
    for cat in ["ac", "pilot", "cabin", "pax"]:
        edges = scored.get(cat, [])
        if edges:
            top = edges[0]
            summary_stats.append(
                {
                    "label": f"Top {cat} edge",
                    "value": f"{top['src_fid']}→{top['dst_fid']} (Score {top['score']:.2f})",
                }
            )
        summary_stats.append({"label": f"{cat} edges listed", "value": len(edges)})

    calc_ms = (time.time() - t0) * 1000.0
    visuals = Visuals(
        heatmap_mode="prob_delay",
        edges=visual_edges,
        nodes=visual_nodes,
        summary_panel=SummaryPanel(title="Connection Risk", stats=summary_stats),
    )

    return GlobalResponse(
        meta_info=MetaInfo(tool_exec="connection_risk", calc_time_ms=calc_ms, model_name=req.model),
        network_state=net_state,
        visuals=visuals,
    )
